import argparse
import asyncio
import json
import os
import queue
import re
import shutil
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

from image_to_talking_avatar import (
    apply_realtime_preset,
    copy_file_with_retry,
    detect_text_emotion,
    find_builtin_driving_template,
    pick_template_for_emotion,
    prepare_liveportrait_base,
    run_cmd,
    run_wav2lip,
    synthesize_tts,
)

try:
    import cv2
except Exception:
    cv2 = None

try:
    import numpy as np
except Exception:
    np = None

try:
    import torch
except Exception:
    torch = None

try:
    from aiortc import RTCPeerConnection, RTCSessionDescription
    from aiortc.contrib.media import MediaPlayer

    WEBRTC_IMPORT_ERROR = ""
except Exception as _webrtc_exc:
    RTCPeerConnection = None
    RTCSessionDescription = None
    MediaPlayer = None
    WEBRTC_IMPORT_ERROR = str(_webrtc_exc)


PIPELINE_FLOW = "Text -> streaming TTS -> neural talking head frame generation -> direct frame stream -> WebRTC -> browser avatar"
STREAM_FIRST_CHUNK_MIN_CHARS = 28
STREAM_FIRST_CHUNK_MAX_CHARS = 64
STREAM_NEXT_CHUNK_MIN_CHARS = 28
STREAM_NEXT_CHUNK_MAX_CHARS = 72
STREAM_FIRST_CHUNK_MIN_WORDS = 6
STREAM_FIRST_CHUNK_MAX_WORDS = 14
STREAM_NEXT_CHUNK_MIN_WORDS = 6
STREAM_NEXT_CHUNK_MAX_WORDS = 12
STREAM_AUDIO_FALLBACK_MS = 450
STREAM_POLL_INTERVAL_MS = 80
STREAM_FILE_RETENTION_SEC = 300
STREAM_SEMANTIC_TARGET_MS = 1000
STREAM_SEMANTIC_MIN_MS = 650
STREAM_SEMANTIC_MAX_MS = 1800
STREAM_PREFETCH_AHEAD = 2
STREAM_FORCE_CHUNK_CHARS = 72


def stage_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def log_stage(stage, message, turn_id="", chunk_idx=None):
    prefix = f"[{stage_timestamp()}] [{stage}]"
    if turn_id:
        prefix += f" [turn={turn_id}]"
    if chunk_idx is not None:
        prefix += f" [chunk={int(chunk_idx):04d}]"
    print(f"{prefix} {message}", flush=True)


def is_client_disconnect_error(exc):
    if isinstance(exc, (BrokenPipeError, ConnectionAbortedError, ConnectionResetError)):
        return True
    if isinstance(exc, OSError):
        if getattr(exc, "winerror", None) in {10053, 10054}:
            return True
        if getattr(exc, "errno", None) in {32, 104}:
            return True
    return False


def guess_audio_content_type(src, data=None):
    header = bytes(data[:12]) if data else b""
    if header.startswith(b"RIFF") and header[8:12] == b"WAVE":
        return "audio/wav"
    if header.startswith(b"ID3"):
        return "audio/mpeg"
    if len(header) >= 2 and header[0] == 0xFF and (header[1] & 0xE0) == 0xE0:
        return "audio/mpeg"
    suffix = str(getattr(src, "suffix", "") or "").lower()
    if suffix == ".wav":
        return "audio/wav"
    return "audio/mpeg"


def normalize_openai_api_url(value):
    url = str(value or "https://api.openai.com/v1/chat/completions").strip().rstrip("/")
    if not url:
        return "https://api.openai.com/v1/chat/completions"
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1"):
        return url + "/chat/completions"
    if url.endswith("/openai/v1"):
        return url + "/chat/completions"
    if "/chat/completions" not in url:
        return url + "/v1/chat/completions"
    return url


def safe_unlink(path):
    try:
        p = Path(path)
        if p.exists():
            p.unlink()
            return True
    except Exception:
        return False
    return False


def safe_rmtree(path):
    try:
        p = Path(path)
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)
            return True
    except Exception:
        return False
    return False


def wait_for_file_stable(path, timeout_sec=8.0, settle_sec=0.25, poll_sec=0.05):
    target = Path(path)
    deadline = time.time() + max(0.1, float(timeout_sec or 0))
    settle_window = max(float(settle_sec or 0), float(poll_sec or 0))
    last_size = -1
    stable_since = None

    while time.time() <= deadline:
        try:
            if target.exists():
                size_now = int(target.stat().st_size)
                if size_now > 0:
                    if size_now == last_size:
                        if stable_since is None:
                            stable_since = time.time()
                        elif (time.time() - stable_since) >= settle_window:
                            return target
                    else:
                        last_size = size_now
                        stable_since = time.time()
        except Exception:
            stable_since = None
        time.sleep(max(0.01, float(poll_sec or 0.05)))

    raise RuntimeError(f"Timed out waiting for stable file: {target}")


def load_env_file(path=".env"):
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue
        if len(value) >= 2 and ((value[0] == '"' and value[-1] == '"') or (value[0] == "'" and value[-1] == "'")):
            value = value[1:-1]
        os.environ[key] = value


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Live Talking Avatar</title>
  <style>
    :root {
      --bg: #f1efe7;
      --ink: #11201e;
      --accent: #0d7a67;
      --panel: #fffdf8;
      --muted: #5f6d6b;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", Tahoma, sans-serif;
      background: radial-gradient(circle at 20% 0%, #fffaf0, #f1efe7 52%, #e4ebe6 100%);
      color: var(--ink);
      min-height: 100vh;
      display: grid;
      place-items: center;
      padding: 20px;
    }
    .app {
      width: min(1100px, 100%);
      display: grid;
      grid-template-columns: 1.1fr 0.9fr;
      gap: 16px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid #d8ddd8;
      border-radius: 16px;
      padding: 16px;
      box-shadow: 0 8px 24px rgba(17, 32, 30, 0.08);
    }
    h1 {
      margin: 0 0 10px 0;
      font-size: 24px;
    }
    .subtitle {
      margin: 0 0 14px 0;
      color: var(--muted);
      font-size: 14px;
    }
    #chat {
      height: 420px;
      overflow: auto;
      border: 1px solid #d9e0db;
      border-radius: 12px;
      padding: 10px;
      background: #fcfffd;
    }
    .msg {
      margin: 8px 0;
      padding: 9px 10px;
      border-radius: 10px;
      max-width: 88%;
      white-space: pre-wrap;
      word-wrap: break-word;
    }
    .user { background: #e3f7f3; margin-left: auto; }
    .bot { background: #f0f4ff; margin-right: auto; }
    .row {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
      margin-top: 10px;
    }
    #input {
      width: 100%;
      border: 1px solid #c7d3cd;
      border-radius: 10px;
      padding: 12px;
      font-size: 15px;
      outline: none;
    }
    button {
      border: 0;
      border-radius: 10px;
      background: var(--accent);
      color: white;
      padding: 0 18px;
      font-size: 15px;
      cursor: pointer;
      min-height: 42px;
    }
    button[disabled] { opacity: 0.6; cursor: wait; }
    .status {
      margin-top: 8px;
      color: var(--muted);
      font-size: 13px;
      min-height: 18px;
    }
    .avatar-wrap {
      position: relative;
      width: 100%;
      border-radius: 12px;
      border: 1px solid #d8ddd8;
      overflow: hidden;
      aspect-ratio: 9 / 12;
      background: #0f1211;
    }
    #avatarImage, #avatarMouthOverlay, video {
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
    }
    #avatarMouthOverlay {
      display: none !important;
    }
        #player {
      display: none;
    }
    .mouth {
      display: none !important;
    }
    .avatar-wrap.speaking .mouth {
      opacity: 0.98;
    }
    .avatar-wrap.video-mode .mouth,
    .avatar-wrap.video-mode #avatarMouthOverlay {
      opacity: 0;
    }
    .mic-row {
      display: grid;
      grid-template-columns: auto 1fr;
      gap: 10px;
      margin-top: 10px;
      align-items: center;
    }
    #mic {
      background: #124a41;
    }
    #mic.active {
      background: #a2382f;
    }
    .mic-status {
      font-size: 13px;
      color: var(--muted);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .meta {
      margin-top: 8px;
      font-size: 13px;
      color: var(--muted);
    }
    @media (max-width: 920px) {
      .app { grid-template-columns: 1fr; }
      #chat { height: 280px; }
    }
  </style>
</head>
<body>
  <div class="app">
    <section class="panel">
      <h1>Live Talking Avatar</h1>
      <p class="subtitle">Heygen-style live mode: type ya mic se bolo, LLM reply dega aur avatar real-time bolega.</p>
      <div id="chat"></div>
      <div class="row">
        <input id="input" placeholder="Type your message..." />
        <button id="send">Send</button>
      </div>
      <div class="mic-row">
        <button id="mic">Start Mic</button>
        <div id="micStatus" class="mic-status">Mic idle</div>
      </div>
      <div id="status" class="status"></div>
    </section>

    <section class="panel">
      <div id="avatarWrap" class="avatar-wrap">
        <img id="avatarImage" alt="Avatar image" />
        <img id="avatarMouthOverlay" alt="Avatar mouth overlay" />
        <video id="player" controls autoplay></video>
        <div class="mouth"></div>
      </div>
      <audio id="audioPlayer" controls autoplay style="width:100%;margin-top:10px;display:none;"></audio>
      <div class="meta">Latest response: <span id="mediaPath">not generated yet</span></div>
      <div class="meta">Tip: OpenAI key set hai to responses LLM se aayenge.</div>
    </section>
  </div>

    <script>
    const chat = document.getElementById("chat");
    const input = document.getElementById("input");
    const send = document.getElementById("send");
    const mic = document.getElementById("mic");
    const micStatus = document.getElementById("micStatus");
    const statusEl = document.getElementById("status");
    const player = document.getElementById("player");
    const audioPlayer = document.getElementById("audioPlayer");
    const mediaPath = document.getElementById("mediaPath");
    const avatarImage = document.getElementById("avatarImage");
    const avatarMouthOverlay = document.getElementById("avatarMouthOverlay");
    const avatarWrap = document.getElementById("avatarWrap");
    const streamAudioFallbackMs = 600;
    const streamPollIntervalMs = 160;

    function setAvatarImageSource(url) {
      const src = String(url || "").trim();
      if (!src) return;
      avatarImage.src = src;
      if (avatarMouthOverlay) avatarMouthOverlay.src = src;
    }

    function applyMouthAnchor(anchor) {
      const a = anchor && typeof anchor === "object" ? anchor : {};
      const x = Math.max(10, Math.min(90, Number(a.x_pct) || 50));
      const y = Math.max(10, Math.min(95, Number(a.y_pct) || 72));
      const w = Math.max(5, Math.min(20, Number(a.w_pct) || 8.8));
      const h = Math.max(2, Math.min(10, Number(a.h_pct) || 3.2));
      avatarWrap.style.setProperty("--mouth-x-pct", String(x));
      avatarWrap.style.setProperty("--mouth-y-pct", String(y));
      avatarWrap.style.setProperty("--mouth-w-pct", String(w));
      avatarWrap.style.setProperty("--mouth-h-pct", String(h));
      debugLog("LIP", `anchor x=${x.toFixed(1)} y=${y.toFixed(1)} w=${w.toFixed(1)} h=${h.toFixed(1)}`);
    }

    setAvatarImageSource("/avatar-image");
    applyMouthAnchor(null);

    let recognition = null;
    let micActive = false;
    let micBuffer = "";

    let audioCtx = null;
    let analyser = null;
    let analyserData = null;
    let sourceNode = null;
    let lipRaf = 0;

    let pendingTurnId = "";
    let videoPollTimer = null;

    let streamPollTimer = null;
    let streamEventSource = null;
    let streamFrameFallbackTimer = 0;
    let streamPollStartedAt = 0;
    let streamLastProgressAt = 0;
    let streamDoneAt = 0;
    let streamActiveTurnId = "";

    let streamSeenVideoIdx = new Set();
    let streamVideoQueue = [];
    let streamVideoPlaying = false;
    let streamLastVideoIdx = -1;

    let streamSeenFrameIdx = new Set();
    let streamFrameQueue = [];
    let streamFrameTimer = 0;
    let streamFrameDelayMs = 33;
    let streamUsingFrames = false;
    let streamLastFrameIdx = -1;

    let streamSeenAudioIdx = new Set();
    let streamAudioQueue = [];
    let streamAudioPlaying = false;
    let streamPendingAudio = new Map();
    let streamFrameCountByChunk = new Map();
    let streamLastAudioPollIdx = -1;
    let streamLastAudioReleaseIdx = -1;

    let streamChunkTexts = new Map();
    let streamChunkMaxIdx = -1;
    let streamBotNode = null;
    let streamLastPollSnapshot = "";

    let preferRealLipSync = false;
    let autoAudioFallbackEnabled = true;

    let useWebRTC = false;
    let useFrameStream = true;
    let activePeer = null;
    let audioPrimed = false;

    async function primeAudioPlayback() {
      if (audioPrimed) return true;
      try {
        const AC = window.AudioContext || window.webkitAudioContext;
        if (AC && !audioCtx) audioCtx = new AC();
        if (audioCtx && audioCtx.state !== "running") {
          await audioCtx.resume().catch(() => {});
        }
        audioPlayer.muted = true;
        audioPlayer.src = "data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEAESsAACJWAAACABAAZGF0YQAAAAA=";
        await audioPlayer.play();
        audioPlayer.pause();
        audioPlayer.currentTime = 0;
        audioPlayer.removeAttribute("src");
        audioPlayer.load();
        audioPlayer.muted = false;
        audioPrimed = true;
        debugLog("AUDIO", "Audio playback unlocked by user gesture");
        return true;
      } catch (err) {
        debugLog("AUDIO", "Audio unlock pending user interaction", err);
        audioPlayer.muted = false;
        return false;
      }
    }

    function debugLog(stage, message, extra) {
      const stamp = new Date().toISOString();
      if (typeof extra === "undefined") console.log(`[${stamp}] [${stage}] ${message}`);
      else console.log(`[${stamp}] [${stage}] ${message}`, extra);
    }

    function addMsg(role, text) {
      const div = document.createElement("div");
      div.className = "msg " + (role === "user" ? "user" : "bot");
      div.textContent = text;
      chat.appendChild(div);
      chat.scrollTop = chat.scrollHeight;
      return div;
    }

    function setSpeaking(isSpeaking) {
      avatarWrap.classList.toggle("speaking", Boolean(isSpeaking));
    }

    function setVideoMode(isVideo) {
      avatarWrap.classList.toggle("video-mode", Boolean(isVideo));
    }

    function setMouthLevel(level) {
      const clamped = Math.max(0, Math.min(1, Number(level) || 0));
      avatarWrap.style.setProperty("--mouth-open", (0.35 + clamped * 4.1).toFixed(3));
      setSpeaking(clamped > 0.008);
    }

    function stopAudioLipsync() {
      if (lipRaf) {
        cancelAnimationFrame(lipRaf);
        lipRaf = 0;
      }
      setMouthLevel(0);
      setSpeaking(false);
    }

    function startAudioLipsync() {
      const AC = window.AudioContext || window.webkitAudioContext;
      if (!AC) {
        setSpeaking(true);
        return;
      }
      if (!audioCtx) {
        audioCtx = new AC();
      }
      if (!analyser) {
        analyser = audioCtx.createAnalyser();
        analyser.fftSize = 512;
        analyser.smoothingTimeConstant = 0.55;
      }
      if (!sourceNode) {
        sourceNode = audioCtx.createMediaElementSource(audioPlayer);
        sourceNode.connect(analyser);
        analyser.connect(audioCtx.destination);
      }
      if (!analyserData || analyserData.length !== analyser.frequencyBinCount) {
        analyserData = new Uint8Array(analyser.frequencyBinCount);
      }
      audioCtx.resume().catch(() => {});
      if (lipRaf) {
        cancelAnimationFrame(lipRaf);
      }

      const tick = () => {
        if (audioPlayer.paused || audioPlayer.ended) {
          stopAudioLipsync();
          return;
        }
        analyser.getByteFrequencyData(analyserData);
        const usefulBins = Math.min(48, analyserData.length);
        let energy = 0;
        for (let i = 2; i < usefulBins; i++) {
          energy += analyserData[i];
        }
        const avg = usefulBins > 2 ? (energy / (usefulBins - 2)) : 0;
        const level = Math.min(1, Math.max(0, (avg - 8) / 42));
        setMouthLevel(level);
        lipRaf = requestAnimationFrame(tick);
      };

      tick();
    }

    function closeWebRtcPeer() {
      if (activePeer) {
        try { activePeer.close(); } catch (_) {}
        activePeer = null;
      }
    }

    async function playVideoViaWebRtc(videoUrl) {
      closeWebRtcPeer();
      const pc = new RTCPeerConnection();
      activePeer = pc;
      const remote = new MediaStream();

      pc.ontrack = (event) => {
        const tracks = event.streams && event.streams[0] ? event.streams[0].getTracks() : [event.track];
        for (const t of tracks) remote.addTrack(t);
        player.srcObject = remote;
      };

      pc.addTransceiver("video", { direction: "recvonly" });
      pc.addTransceiver("audio", { direction: "recvonly" });

      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);

      const res = await fetch("/api/webrtc-offer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sdp: pc.localDescription.sdp,
          type: pc.localDescription.type,
          media_url: videoUrl,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "WebRTC offer failed");

      await pc.setRemoteDescription({ type: data.type, sdp: data.sdp });
      return pc;
    }

    function stopMediaPlayers() {
      stopAudioLipsync();
      if (streamFrameTimer) {
        clearInterval(streamFrameTimer);
        streamFrameTimer = 0;
      }
      closeWebRtcPeer();
      try { player.pause(); } catch (_) {}
      try { player.srcObject = null; } catch (_) {}
      try { audioPlayer.pause(); } catch (_) {}
    }

    async function showVideoResponse(videoUrl) {
      stopAudioLipsync();
      try { audioPlayer.pause(); } catch (_) {}
      setVideoMode(true);
      avatarImage.style.display = "none";
      player.style.display = "block";
      audioPlayer.style.display = "none";
      setSpeaking(false);
      mediaPath.textContent = videoUrl;

      if (useWebRTC && window.RTCPeerConnection) {
        try {
          await playVideoViaWebRtc(videoUrl);
          await player.play().catch(() => {});
          return;
        } catch (err) {
          console.warn("WebRTC playback failed, falling back to file URL:", err);
          closeWebRtcPeer();
        }
      }

      try { player.srcObject = null; } catch (_) {}
      player.src = videoUrl;
      player.load();
      player.play().catch(() => {});
    }

    function stopVideoPolling() {
      pendingTurnId = "";
      if (videoPollTimer) {
        clearInterval(videoPollTimer);
        videoPollTimer = null;
      }
    }

    function startVideoPolling(turnId) {
      stopVideoPolling();
      pendingTurnId = turnId;
      const startedAt = Date.now();
      videoPollTimer = setInterval(async () => {
        if (!pendingTurnId || pendingTurnId !== turnId) return;
        try {
          const res = await fetch(`/api/turn-status?turn_id=${encodeURIComponent(turnId)}`);
          if (!res.ok) return;
          const data = await res.json();
          if (data.failed) {
            stopVideoPolling();
            statusEl.textContent = "Video render failed: " + (data.error || "unknown");
            return;
          }
          if (data.ready && data.video_url) {
            stopVideoPolling();
            showVideoResponse(data.video_url).catch(() => {});
            statusEl.textContent = "Done";
            return;
          }
          if (Date.now() - startedAt > 180000) {
            stopVideoPolling();
            statusEl.textContent = "Video still rendering. Send next text if needed.";
          }
        } catch (_) {}
      }, 700);
    }

    function setStreamFrameDelay(ms) {
      const n = Number(ms);
      if (!Number.isFinite(n)) return;
      const clamped = Math.max(20, Math.min(250, Math.round(n)));
      if (clamped === streamFrameDelayMs) return;
      streamFrameDelayMs = clamped;
      if (streamFrameTimer) {
        clearInterval(streamFrameTimer);
        streamFrameTimer = 0;
        startFramePlaybackLoop();
      }
    }

    function startFramePlaybackLoop() {
      if (streamFrameTimer) return;
      streamFrameTimer = setInterval(() => {
        if (!streamFrameQueue.length) return;
        const nextFrame = streamFrameQueue.shift();
        debugLog("FRAME", `Pushing frame to avatar queue_len=${streamFrameQueue.length} url=${nextFrame}`);
        setVideoMode(false);
        avatarImage.style.display = "block";
        player.style.display = "none";
        try { player.pause(); } catch (_) {}
        try { player.srcObject = null; } catch (_) {}
        player.removeAttribute("src");
        setAvatarImageSource(nextFrame);
        mediaPath.textContent = nextFrame;
      }, streamFrameDelayMs);
    }

    function queueStreamFrame(url) {
      if (!url) return;
      streamUsingFrames = true;
      streamFrameQueue.push(url);
      debugLog("FRAME", `Queued frame for playback queue_len=${streamFrameQueue.length} url=${url}`);
      startFramePlaybackLoop();
    }

    function playNextStreamVideo() {
      if (streamVideoPlaying || !streamVideoQueue.length) return;
      const nextVideo = streamVideoQueue.shift();
      streamVideoPlaying = true;
      stopAudioLipsync();
      try { audioPlayer.pause(); } catch (_) {}
      setVideoMode(true);
      avatarImage.style.display = "none";
      player.style.display = "block";
      audioPlayer.style.display = "none";
      closeWebRtcPeer();
      try { player.pause(); } catch (_) {}
      try { player.srcObject = null; } catch (_) {}
      player.src = nextVideo;
      player.load();
      player.play().catch(() => {
        streamVideoPlaying = false;
      });
      mediaPath.textContent = nextVideo;
      statusEl.textContent = "Talking...";
    }

    function queueStreamVideo(url) {
      if (!url) return;
      streamVideoQueue.push(url);
      debugLog("VIDEO", `Queued chunk video queue_len=${streamVideoQueue.length} url=${url}`);
      playNextStreamVideo();
    }

    function playNextStreamAudio() {
      if (streamAudioPlaying || !streamAudioQueue.length) return;
      const nextAudio = streamAudioQueue.shift();
      debugLog("AUDIO", `Starting queued audio playback queue_len=${streamAudioQueue.length} url=${nextAudio}`);
      streamAudioPlaying = true;
      setVideoMode(false);
      avatarImage.style.display = "block";
      player.style.display = "none";
      try { player.pause(); } catch (_) {}
      try { player.srcObject = null; } catch (_) {}
      player.removeAttribute("src");
      audioPlayer.style.display = "none";
      audioPlayer.src = nextAudio;
      audioPlayer.load();
      audioPlayer.play().catch(() => {
        streamAudioPlaying = false;
        statusEl.textContent = "Audio autoplay blocked. Click page once, then Send.";
      });
      mediaPath.textContent = nextAudio;
      statusEl.textContent = "Talking...";
    }

    function queueStreamAudio(url) {
      if (!url) return;
      streamAudioQueue.push(url);
      debugLog("AUDIO", `Queued audio chunk queue_len=${streamAudioQueue.length} url=${url}`);
      playNextStreamAudio();
    }

    function flushPendingAudioForLipsync(maxChunks = 1, force = false) {
      if (!streamPendingAudio.size) return 0;
      const pairs = Array.from(streamPendingAudio.entries()).sort((a, b) => a[0] - b[0]);
      const limit = Math.max(1, Number(maxChunks) || 1);
      let released = 0;
      for (const [idx, chunk] of pairs) {
        const url = chunk && chunk.audio_url ? chunk.audio_url : "";
        if (!url || streamSeenAudioIdx.has(idx)) {
          streamPendingAudio.delete(idx);
          continue;
        }
        if (streamLastAudioReleaseIdx >= 0 && idx !== (streamLastAudioReleaseIdx + 1)) {
          continue;
        }
        const expectedFrames = Math.max(0, Number(chunk.frame_count) || 0);
        const bufferedFrames = Math.max(0, Number(streamFrameCountByChunk.get(idx)) || 0);
        const enoughFrames = expectedFrames === 0 || bufferedFrames >= Math.min(expectedFrames, 6);
        if (!force && preferRealLipSync && !enoughFrames) {
          continue;
        }
        streamPendingAudio.delete(idx);
        streamSeenAudioIdx.add(idx);
        if (idx > streamLastAudioReleaseIdx) streamLastAudioReleaseIdx = idx;
        debugLog("SYNC", `Releasing audio chunk idx=${idx} expected_frames=${expectedFrames} buffered_frames=${bufferedFrames} force=${force}`);
        queueStreamAudio(url);
        released += 1;
        if (released >= limit) break;
      }
      return released;
    }

    audioPlayer.onplay = () => {
      debugLog("AUDIO", `Audio element playing src=${audioPlayer.currentSrc || audioPlayer.src || ""}`);
      setSpeaking(true);
      startAudioLipsync();
    };
    audioPlayer.onended = () => {
      debugLog("AUDIO", "Audio element ended");
      streamAudioPlaying = false;
      stopAudioLipsync();
      playNextStreamAudio();
    };
    audioPlayer.onerror = () => {
      debugLog("AUDIO", "Audio element error", audioPlayer.error || null);
      streamAudioPlaying = false;
      stopAudioLipsync();
      playNextStreamAudio();
    };
    player.onended = () => {
      debugLog("VIDEO", "Video element ended");
      streamVideoPlaying = false;
      if (streamVideoQueue.length) {
        playNextStreamVideo();
        return;
      }
      if (!useFrameStream) {
        setVideoMode(false);
        avatarImage.style.display = "block";
        player.style.display = "none";
      }
    };
    player.onerror = () => {
      debugLog("VIDEO", "Video element error", player.error || null);
      streamVideoPlaying = false;
      if (streamVideoQueue.length) playNextStreamVideo();
    };

    function stopStreamPolling(clearBotNode = true) {
      streamActiveTurnId = "";
      if (clearBotNode) streamBotNode = null;

      streamSeenVideoIdx = new Set();
      streamVideoQueue = [];
      streamVideoPlaying = false;
      streamLastVideoIdx = -1;

      streamSeenFrameIdx = new Set();
      streamFrameQueue = [];
      streamUsingFrames = false;
      streamFrameDelayMs = 33;
      if (streamFrameTimer) {
        clearInterval(streamFrameTimer);
        streamFrameTimer = 0;
      }

      streamSeenAudioIdx = new Set();
      streamAudioQueue = [];
      streamAudioPlaying = false;
      streamPendingAudio = new Map();
      streamFrameCountByChunk = new Map();
      streamLastAudioPollIdx = -1;
      streamLastAudioReleaseIdx = -1;
      streamLastFrameIdx = -1;

      streamChunkTexts = new Map();
      streamChunkMaxIdx = -1;
      streamLastPollSnapshot = "";

      streamPollStartedAt = 0;
      streamLastProgressAt = 0;
      streamDoneAt = 0;

      try { audioPlayer.pause(); } catch (_) {}
      if (streamPollTimer) {
        clearInterval(streamPollTimer);
        streamPollTimer = null;
      }
      if (streamEventSource) {
        try { streamEventSource.close(); } catch (_) {}
        streamEventSource = null;
      }
      if (streamFrameFallbackTimer) {
        clearTimeout(streamFrameFallbackTimer);
        streamFrameFallbackTimer = 0;
      }
    }

    function pickBestReplyText(reply, partial, chunkMap, maxIdx) {
      const finalText = String(reply || "").trim();
      if (finalText) return finalText;

      const partialText = String(partial || "").trim();
      let assembled = "";
      if (chunkMap && maxIdx >= 0) {
        const parts = [];
        for (let i = 0; i <= maxIdx; i++) {
          const t = chunkMap.get(i);
          if (t) parts.push(String(t).trim());
        }
        assembled = parts.join(" ").trim();
      }

      if (partialText && partialText.length >= assembled.length) return partialText;
      return assembled || partialText;
    }

    function handleStreamEvent(evt) {
      const type = String(evt && evt.type || "").trim();
      const payload = evt && evt.payload ? evt.payload : {};
      if (!type) return;

      if (type === "pipeline") {
        const stage = String(payload.stage || "").trim();
        if (stage) {
          debugLog("PIPELINE", `stage=${stage}`);
          statusEl.textContent = stage === "start" ? "Live avatar starting..." : statusEl.textContent;
        }
        return;
      }

      if (type === "partial") {
        const partial = String(payload.partial_reply || "").trim();
        if (streamBotNode && partial && partial.length >= String(streamBotNode.textContent || "").trim().length) {
          streamBotNode.textContent = partial;
        }
        streamLastProgressAt = Date.now();
        return;
      }

      if (type === "reply") {
        const reply = String(payload.reply || "").trim();
        if (streamBotNode && reply) streamBotNode.textContent = reply;
        streamLastProgressAt = Date.now();
        return;
      }

      if (type === "audio_chunk") {
        const idx = Number(payload.idx);
        if (Number.isNaN(idx) || streamSeenAudioIdx.has(idx)) return;
        setStreamFrameDelay(payload.frame_delay_ms);
        streamSeenAudioIdx.add(idx);
        if (idx > streamLastAudioPollIdx) streamLastAudioPollIdx = idx;
        if (idx > streamLastAudioReleaseIdx) streamLastAudioReleaseIdx = idx;
        debugLog("AUDIO", `push chunk idx=${idx} url=${payload.audio_url || ""}`);
        queueStreamAudio(payload.audio_url || "");
        streamLastProgressAt = Date.now();
        if (!streamFrameFallbackTimer) {
          streamFrameFallbackTimer = setTimeout(() => {
            if (!streamSeenFrameIdx.size) {
              debugLog("FRAME", "No frame produced within 1s, continuing audio-first fallback render path");
            }
            streamFrameFallbackTimer = 0;
          }, 1000);
        }
        return;
      }

      if (type === "frame_chunk") {
        const idx = Number(payload.idx);
        if (Number.isNaN(idx) || streamSeenFrameIdx.has(idx)) return;
        streamSeenFrameIdx.add(idx);
        if (idx > streamLastFrameIdx) streamLastFrameIdx = idx;
        if (streamFrameFallbackTimer) {
          clearTimeout(streamFrameFallbackTimer);
          streamFrameFallbackTimer = 0;
        }
        debugLog("WEBRTC", `frame_send idx=${idx} chunk_idx=${Number(payload.chunk_idx)} url=${payload.frame_url || ""}`);
        queueStreamFrame(payload.frame_url || "");
        streamLastProgressAt = Date.now();
        return;
      }

      if (type === "video_chunk") {
        const idx = Number(payload.idx);
        if (Number.isNaN(idx) || streamSeenVideoIdx.has(idx)) return;
        streamSeenVideoIdx.add(idx);
        if (idx > streamLastVideoIdx) streamLastVideoIdx = idx;
        queueStreamVideo(payload.video_url || "");
        streamLastProgressAt = Date.now();
        return;
      }

      if (type === "done") {
        const reply = String(payload.reply || "").trim();
        if (streamBotNode && reply) streamBotNode.textContent = reply;
        streamDoneAt = Date.now();
        statusEl.textContent = "Done";
        if (streamEventSource) {
          try { streamEventSource.close(); } catch (_) {}
          streamEventSource = null;
        }
        return;
      }

      if (type === "error") {
        statusEl.textContent = "Error: " + String(payload.error || "stream failed");
        if (streamEventSource) {
          try { streamEventSource.close(); } catch (_) {}
          streamEventSource = null;
        }
        return;
      }
    }

    function startStreamPolling(turnId) {
      stopStreamPolling(false);
      streamActiveTurnId = turnId;
      streamPollStartedAt = Date.now();
      streamLastProgressAt = streamPollStartedAt;
      streamDoneAt = 0;
      const url = `/api/stream-events?turn_id=${encodeURIComponent(turnId)}`;
      const es = new EventSource(url);
      streamEventSource = es;
      debugLog("WEBRTC", `event_stream_open ${url}`);
      es.onmessage = (messageEvent) => {
        try {
          const evt = JSON.parse(messageEvent.data || "{}");
          if (evt && evt.type === "keepalive") return;
          handleStreamEvent(evt || {});
        } catch (err) {
          debugLog("WEBRTC", "event_stream_parse_error", err);
        }
      };
      es.onerror = () => {
        debugLog("WEBRTC", "event_stream_error");
        if (streamEventSource === es) {
          try { es.close(); } catch (_) {}
          streamEventSource = null;
        }
      };
    }

    async function doSend(forcedMessage = "") {
      const message = (forcedMessage || input.value).trim();
      if (!message || send.disabled) return;

      await primeAudioPlayback();

      stopVideoPolling();
      stopStreamPolling();
      stopMediaPlayers();

      input.value = "";
      addMsg("user", message);
      streamBotNode = addMsg("bot", "...");
      send.disabled = true;
      mic.disabled = true;
      statusEl.textContent = "Live avatar starting...";

      try {
        const res = await fetch("/api/talk-stream", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Request failed");

        const turnId = data.turn_id || "";
        if (!turnId) throw new Error("missing turn id");
        streamActiveTurnId = turnId;
        debugLog("POLL", `Starting polling for session=${turnId}`);
        if (streamBotNode && streamBotNode.textContent === "...") streamBotNode.textContent = "Thinking...";
        startStreamPolling(turnId);
      } catch (err) {
        stopStreamPolling();
        stopAudioLipsync();
        statusEl.textContent = "Error: " + err.message;
      } finally {
        send.disabled = false;
        mic.disabled = false;
        input.focus();
      }
    }

    function initMic() {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (!SpeechRecognition) {
        mic.disabled = true;
        micStatus.textContent = "Mic not supported in this browser";
        return;
      }

      recognition = new SpeechRecognition();
      recognition.lang = "hi-IN";
      recognition.interimResults = true;
      recognition.continuous = true;

      recognition.onresult = (event) => {
        let interim = "";
        for (let i = event.resultIndex; i < event.results.length; i++) {
          const transcript = (event.results[i][0].transcript || "").trim();
          if (!transcript) continue;
          if (event.results[i].isFinal) {
            micBuffer += (micBuffer ? " " : "") + transcript;
          } else {
            interim += (interim ? " " : "") + transcript;
          }
        }
        micStatus.textContent = interim ? `Listening: ${interim}` : "Listening...";
      };

      recognition.onerror = (event) => {
        micActive = false;
        mic.classList.remove("active");
        mic.textContent = "Start Mic";
        micStatus.textContent = `Mic error: ${event.error || "unknown"}`;
      };

      recognition.onend = () => {
        if (micActive) {
          try { recognition.start(); } catch (_) {}
          return;
        }
        const msg = micBuffer.trim();
        micBuffer = "";
        if (msg) {
          micStatus.textContent = "Sending...";
          doSend(msg);
        } else {
          micStatus.textContent = "Mic idle";
        }
      };

      mic.addEventListener("click", () => {
        if (!recognition) return;
        if (!micActive) {
          micActive = true;
          micBuffer = "";
          mic.classList.add("active");
          mic.textContent = "Stop Mic";
          micStatus.textContent = "Listening...";
          try { recognition.start(); } catch (_) {}
          return;
        }
        micActive = false;
        mic.classList.remove("active");
        mic.textContent = "Start Mic";
        micStatus.textContent = "Processing...";
        try { recognition.stop(); } catch (_) {}
      });
    }

    async function initRuntime() {
      try {
        const res = await fetch("/health");
        if (!res.ok) return;
        const data = await res.json();
        useWebRTC = Boolean(data.webrtc_enabled);
        useFrameStream = Boolean(data.frame_stream);
        preferRealLipSync = Boolean(data.frame_stream);
        applyMouthAnchor(data.mouth_anchor || null);
        if (useWebRTC) {
          statusEl.textContent = "WebRTC playback enabled (strict real lipsync)";
        } else if (useFrameStream) {
          statusEl.textContent = "Real-time frame lipsync enabled";
        } else {
          statusEl.textContent = "Real-time chunk video lipsync enabled";
        }
      } catch (_) {}
    }

    send.addEventListener("click", () => doSend());
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") doSend();
    });

    initMic();
    initRuntime();
    input.focus();
  </script>
</body>
</html>
"""


class OpenAIResponder:
    def __init__(self, model, api_key, system_prompt, api_url="https://api.openai.com/v1/chat/completions", max_history=10, request_timeout_sec=20, stream_timeout_sec=25):
        self.model = model
        self.api_key = api_key
        self.system_prompt = system_prompt
        self.api_url = normalize_openai_api_url(api_url)
        self.max_history = max_history
        self.request_timeout_sec = int(request_timeout_sec)
        self.stream_timeout_sec = int(stream_timeout_sec)
        self.history = []
        self.lock = threading.Lock()

    def _build_messages(self, user_text):
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.history[-self.max_history :])
        messages.append({"role": "user", "content": user_text})
        return messages

    def _append_history(self, user_text, assistant_text):
        with self.lock:
            self.history.append({"role": "user", "content": user_text})
            self.history.append({"role": "assistant", "content": assistant_text})
            self.history = self.history[-(self.max_history * 2) :]

    def _rate_limit_wait_sec(self, exc, attempt):
        if isinstance(exc, urllib.error.HTTPError) and exc.code == 429:
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            if retry_after:
                try:
                    return max(1.0, float(retry_after))
                except Exception:
                    pass
            return min(8.0, 1.5 * max(1, attempt))
        return 0.0

    def _request_chat_completion(self, messages, timeout_sec=None):
        log_stage("LLM", f"Starting completion request model={self.model} messages={len(messages)} timeout={timeout_sec or self.request_timeout_sec}")
        url = self.api_url
        payload = {"model": self.model, "messages": messages, "temperature": 0.6}
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout_sec or self.request_timeout_sec) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
        content = data["choices"][0]["message"]["content"].strip()
        log_stage("LLM", f"Completion finished chars={len(content)}")
        return content

    def stream_reply(self, user_text):
        if not self.api_key:
            yield "OpenAI API key missing hai, isliye LLM reply available nahi hai."
            return

        log_stage("LLM", f"Streaming reply started model={self.model} prompt_chars={len(user_text or '')}")

        with self.lock:
            messages = self._build_messages(user_text)

        url = self.api_url
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.6,
            "stream": True,
        }

        pieces = []
        for attempt in range(1, 3):
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.stream_timeout_sec) as resp:
                    for raw in resp:
                        line = raw.decode("utf-8", errors="ignore").strip()
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if not data:
                            continue
                        if data == "[DONE]":
                            break
                        try:
                            obj = json.loads(data)
                            delta = obj["choices"][0].get("delta", {}).get("content", "")
                        except (KeyError, IndexError, json.JSONDecodeError):
                            delta = ""
                        if delta:
                            pieces.append(delta)
                            preview = delta.replace(chr(10), " ").replace(chr(13), " ")[:80]
                            log_stage("LLM", f"Streaming delta chars={len(delta)} total_chars={sum(len(p) for p in pieces)} preview={preview!r}")
                            yield delta
                break
            except Exception as exc:
                wait_sec = self._rate_limit_wait_sec(exc, attempt)
                print(f"[WARN] OpenAI streaming failed (attempt {attempt}): {exc}")
                if pieces:
                    break
                if wait_sec > 0 and attempt < 2:
                    time.sleep(wait_sec)
                    continue
                try:
                    assistant_text = self._request_chat_completion(messages, timeout_sec=max(self.request_timeout_sec, 30))
                except Exception as fallback_exc:
                    print(f"[WARN] OpenAI fallback request failed: {fallback_exc}")
                    yield "OpenAI rate limit aa raha hai. API quota/billing check karo ya thoda wait karke dubara try karo."
                    return
                assistant_text = (assistant_text or "").strip()
                if not assistant_text:
                    assistant_text = "OpenAI ne empty response diya. API/model config check karo."
                log_stage("LLM", f"Fallback completion returned chars={len(assistant_text)}")
                self._append_history(user_text, assistant_text)
                yield assistant_text
                return

        assistant_text = "".join(pieces).strip()
        if not assistant_text:
            assistant_text = "OpenAI ne empty response diya. API/model config check karo."

        log_stage("LLM", f"Streaming reply finished chars={len(assistant_text)}")
        self._append_history(user_text, assistant_text)

    def reply(self, user_text):
        if not self.api_key:
            return "OpenAI API key missing hai, isliye LLM reply available nahi hai."

        log_stage("LLM", f"Sync reply started model={self.model} prompt_chars={len(user_text or '')}")

        with self.lock:
            messages = self._build_messages(user_text)

        for attempt in range(1, 3):
            try:
                assistant_text = self._request_chat_completion(messages)
                assistant_text = (assistant_text or "").strip()
                if not assistant_text:
                    assistant_text = "OpenAI ne empty response diya. API/model config check karo."
                self._append_history(user_text, assistant_text)
                log_stage("LLM", f"Sync reply finished chars={len(assistant_text)}")
                return assistant_text
            except Exception as exc:
                wait_sec = self._rate_limit_wait_sec(exc, attempt)
                print(f"[WARN] OpenAI request failed (attempt {attempt}): {exc}")
                if wait_sec > 0 and attempt < 2:
                    time.sleep(wait_sec)
                    continue
                return "OpenAI rate limit aa raha hai. API quota/billing check karo ya thoda wait karke dubara try karo."




def _prepare_wav2lip_face_image(src_image, cache_dir, max_dim=1536):
    src = Path(src_image)
    if (cv2 is None) or (max_dim is None) or (int(max_dim) <= 0):
        return src
    img = cv2.imread(str(src))
    if img is None:
        return src
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= int(max_dim):
        return src
    scale = float(max_dim) / float(m)
    nw = max(2, int(round(w * scale)))
    nh = max(2, int(round(h * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    try:
        # Mild unsharp mask keeps lip edges a bit cleaner after resize.
        blur = cv2.GaussianBlur(resized, (0, 0), sigmaX=1.0, sigmaY=1.0)
        resized = cv2.addWeighted(resized, 1.12, blur, -0.12, 0)
    except Exception:
        pass
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"w2l_face_{max_dim}.png"
    if cv2.imwrite(str(out), resized):
        return out
    return src


def _sanitize_face_box(img_shape, box):
    if not box:
        return None
    try:
        y1, y2, x1, x2 = [int(v) for v in box]
    except Exception:
        return None
    h, w = int(img_shape[0]), int(img_shape[1])
    y1 = max(0, min(h - 1, y1))
    y2 = max(1, min(h, y2))
    x1 = max(0, min(w - 1, x1))
    x2 = max(1, min(w, x2))
    if y2 <= y1 or x2 <= x1:
        return None
    max_y2 = int(h * 0.68)
    if y2 > max_y2:
        y2 = max(y1 + 8, max_y2)
    if y2 <= y1:
        return None
    return [y1, y2, x1, x2]

def _detect_face_box_sfd(image_path, pads=(0, 20, 0, 0)):
    try:
        import face_detection  # type: ignore
    except Exception:
        return None
    if cv2 is None or np is None or torch is None:
        return None
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = None
    try:
        detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, device=device)
        preds = detector.get_detections_for_batch(np.array([img]))
    except Exception:
        return None
    finally:
        try:
            del detector
        except Exception:
            pass
    if not preds or preds[0] is None:
        return None
    x1, y1, x2, y2 = [int(v) for v in preds[0]]
    pady1, pady2, padx1, padx2 = [int(v) for v in pads]
    box = [y1 - pady1, y2 + pady2, x1 - padx1, x2 + padx2]
    return _sanitize_face_box(img.shape[:2], box)

def _detect_face_box_haar(image_path, pads=(0, 20, 0, 0)):
    if cv2 is None:
        return None
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(cascade_path)
        if detector.empty():
            return None
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        if faces is None or len(faces) == 0:
            return None
        ranked = sorted(faces, key=lambda f: (int(f[1]), -int(f[2]) * int(f[3])))
        x, y, w, h = ranked[0]
        pady1, pady2, padx1, padx2 = [int(v) for v in pads]
        box = [int(y) - pady1, int(y + h) + pady2, int(x) - padx1, int(x + w) + padx2]
        return _sanitize_face_box(img.shape[:2], box)
    except Exception:
        return None

def _detect_face_box(image_path, pads=(0, 20, 0, 0)):
    box = _detect_face_box_sfd(image_path, pads=pads)
    if box:
        return box
    return _detect_face_box_haar(image_path, pads=pads)

class AvatarEngine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.lock = threading.Lock()

        self.image_path = Path(cfg.image).resolve()
        self.checkpoint_path = Path(cfg.checkpoint_path).resolve()
        self.session_dir = Path(cfg.session_dir).resolve()
        self.audio_dir = self.session_dir / "audio"
        self.video_dir = self.session_dir / "video"
        self.base_dir = self.session_dir / "base"
        self.latest_video = self.session_dir / "latest.mp4"
        self.latest_audio = self.session_dir / "latest.wav"
        self.template_cache = {}

        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        if not self.image_path.exists():
            raise FileNotFoundError(f"Image not found: {self.image_path}")
        if (not self.cfg.audio_only) and self.cfg.renderer == "wav2lip" and (not self.checkpoint_path.exists()):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        self.args = SimpleNamespace(
            python_bin=cfg.python_bin,
            liveportrait_repo=cfg.liveportrait_repo,
            checkpoint_path=str(self.checkpoint_path),
            tts_engine=cfg.tts_engine,
            tts_voice="",
            tts_rate=0,
            edge_voice=cfg.edge_voice,
            edge_rate="",
            face_det_batch_size=1,
            wav2lip_batch_size=32,
            wav2lip_resize_factor=1,
            wav2lip_pads=[0, 20, 0, 0],
            wav2lip_nosmooth=False,
            wav2lip_fps=25,
            wav2lip_box=None,
            lp_driving_option="pose-friendly",
            lp_driving_multiplier=0.9,
            lp_smooth_variance=5e-7,
            lp_animation_region=cfg.animation_region,
            lp_det_thresh=0.12,
            lp_scale=2.2,
            lp_source_max_dim=1536,
            lp_crop_driving_video=True,
            lp_no_stitching=False,
            lp_no_relative_motion=False,
            lp_no_pasteback=False,
            lp_force_cpu=cfg.force_cpu,
            realtime_preset=cfg.preset,
        )
        apply_realtime_preset(self.args)
        # Prepare a cached face image + constant box for faster Wav2Lip chunk inference.
        cache_dir = self.session_dir / "cache"
        self.wav2lip_face_image = _prepare_wav2lip_face_image(self.image_path, cache_dir=cache_dir, max_dim=1536)
        self.args.wav2lip_face_path = str(self.wav2lip_face_image)
        if float(getattr(self.args, "wav2lip_fps", 0) or 0) <= 0:
            self.args.wav2lip_fps = 25
        if cfg.wav2lip_box and len(cfg.wav2lip_box) == 4:
            self.args.wav2lip_box = [int(v) for v in cfg.wav2lip_box]
        else:
            self.args.wav2lip_box = _detect_face_box(self.wav2lip_face_image, pads=self.args.wav2lip_pads)
        if self.args.wav2lip_box:
            print(f"[INFO] Wav2Lip cached face box: {self.args.wav2lip_box}")

        self.mouth_anchor = {"x_pct": 50.0, "y_pct": 72.0, "w_pct": 8.8, "h_pct": 3.2}
        try:
            y1, y2, x1, x2 = [int(v) for v in (self.args.wav2lip_box or [0, 0, 0, 0])]
            if cv2 is not None:
                probe = cv2.imread(str(self.wav2lip_face_image))
                if probe is not None:
                    ih, iw = int(probe.shape[0]), int(probe.shape[1])
                else:
                    ih, iw = 960, 960
            else:
                ih, iw = 960, 960
            if (iw > 0) and (ih > 0) and (x2 > x1) and (y2 > y1):
                fx = (x1 + x2) / 2.0
                fy = y1 + (y2 - y1) * 0.78
                fw = max(24.0, (x2 - x1) * 0.42)
                fh = max(12.0, (y2 - y1) * 0.22)
                self.mouth_anchor = {
                    "x_pct": max(10.0, min(90.0, (fx / float(iw)) * 100.0)),
                    "y_pct": max(10.0, min(95.0, (fy / float(ih)) * 100.0)),
                    "w_pct": max(5.0, min(22.0, (fw / float(iw)) * 100.0)),
                    "h_pct": max(2.0, min(12.0, (fh / float(ih)) * 100.0)),
                }
        except Exception:
            pass

        self.stream_face_source = self.wav2lip_face_image
        if self.cfg.renderer == "wav2lip" and (not self.cfg.audio_only) and bool(getattr(self.cfg, "wav2lip_head_motion", False)):
            try:
                self.stream_face_source = self._build_liveportrait_video("d13", "stream_base")
                print(f"[INFO] Wav2Lip stream source (LivePortrait): {self.stream_face_source}")
            except Exception as exc:
                print(f"[WARN] LivePortrait stream base unavailable, using static face source: {exc}")


        self.musetalk_repo = Path(cfg.musetalk_repo).resolve()
        self.musetalk_python_bin = Path(cfg.musetalk_python_bin).resolve()
        self.musetalk_unet_model = self._resolve_musetalk_path(cfg.musetalk_unet_model)
        self.musetalk_unet_config = self._resolve_musetalk_path(cfg.musetalk_unet_config)
        self.musetalk_whisper_dir = self._resolve_musetalk_path(cfg.musetalk_whisper_dir)
        self.musetalk_runtime_ready = True
        self.musetalk_sd_vae_model = (self.musetalk_repo / "models" / "sd-vae" / "diffusion_pytorch_model.safetensors").resolve()

        if self.cfg.renderer == "musetalk":
            if not self.musetalk_repo.exists():
                raise FileNotFoundError(f"MuseTalk repo not found: {self.musetalk_repo}")
            if not self.musetalk_python_bin.exists():
                raise FileNotFoundError(f"MuseTalk python not found: {self.musetalk_python_bin}")
            if not self.musetalk_unet_model.exists():
                raise FileNotFoundError(f"MuseTalk model not found: {self.musetalk_unet_model}")
            if not self.musetalk_unet_config.exists():
                raise FileNotFoundError(f"MuseTalk config not found: {self.musetalk_unet_config}")
            if not self.musetalk_whisper_dir.exists():
                raise FileNotFoundError(f"MuseTalk whisper dir not found: {self.musetalk_whisper_dir}")
            if not self.musetalk_sd_vae_model.exists():
                self.musetalk_runtime_ready = False
                print(f"[WARN] MuseTalk sd-vae model missing: {self.musetalk_sd_vae_model}. Using Wav2Lip realtime fallback.")

        self.turn_counter = self._init_turn_counter()

    def _init_turn_counter(self):
        pattern = re.compile(r"^turn_(\d{4})_\d{8}_\d{6}$")
        max_idx = 0

        for folder in (self.video_dir, self.audio_dir):
            if not folder.exists():
                continue
            for fp in folder.glob("turn_*"):
                m = pattern.match(fp.stem)
                if m:
                    max_idx = max(max_idx, int(m.group(1)))

        return max_idx + 1

    def _build_liveportrait_video(self, template_name, turn_id):
        template_path = find_builtin_driving_template(template_name)
        if template_path is None:
            raise RuntimeError(f"No compatible template found for: {template_name}")

        if self.cfg.liveportrait_each_turn:
            target_dir = self.base_dir / "per_turn_motion" / turn_id
            return prepare_liveportrait_base(
                args=self.args,
                image=self.image_path,
                driving_video=template_path,
                base_dir=target_dir,
            )

        cached = self.base_dir / f"base_motion_{template_name}.mp4"
        if cached.exists() and cached.stat().st_size > 0:
            return cached

        target_dir = self.base_dir / f"text_motion_{template_name}"
        built = prepare_liveportrait_base(
            args=self.args,
            image=self.image_path,
            driving_video=template_path,
            base_dir=target_dir,
        )
        if built != cached:
            copy_file_with_retry(built, cached)
        return cached

    def _resolve_musetalk_path(self, value):
        p = Path(value)
        if p.is_absolute():
            return p
        return (self.musetalk_repo / p).resolve()

    def _run_musetalk_stage(self, audio_path, turn_key, frames_only=False):
        log_stage("MUSETALK", f"Starting inference frames_only={bool(frames_only)} audio={Path(audio_path).name}", turn_id=turn_key)
        job_dir = self.session_dir / "musetalk_jobs" / turn_key
        result_dir = job_dir / "results"
        result_dir.mkdir(parents=True, exist_ok=True)

        inference_cfg = job_dir / "inference.yaml"
        cfg_text = "".join(
            [
                "task_0:\n",
                f" video_path: {json.dumps(str(self.image_path))}\n",
                f" audio_path: {json.dumps(str(Path(audio_path).resolve()))}\n",
            ]
        )
        inference_cfg.write_text(cfg_text, encoding="utf-8")

        output_name = f"{turn_key}.mp4"
        cmd = [
            str(self.musetalk_python_bin),
            "-m",
            "scripts.inference",
            "--inference_config",
            str(inference_cfg),
            "--result_dir",
            str(result_dir),
            "--unet_model_path",
            str(self.musetalk_unet_model),
            "--unet_config",
            str(self.musetalk_unet_config),
            "--whisper_dir",
            str(self.musetalk_whisper_dir),
            "--version",
            str(self.cfg.musetalk_version),
            "--fps",
            str(self.cfg.musetalk_fps),
            "--gpu_id",
            str(self.cfg.musetalk_gpu_id),
            "--batch_size",
            str(self.cfg.musetalk_batch_size),
            "--use_saved_coord",
            "--saved_coord",
        ]
        if frames_only:
            cmd.append("--frames_only")
        else:
            cmd.extend(["--output_vid_name", output_name])
        if self.cfg.musetalk_use_float16:
            cmd.append("--use_float16")

        def _run_once(force_cpu=False):
            prev_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
            try:
                if force_cpu or self.cfg.force_cpu:
                    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
                out = run_cmd(cmd, cwd=str(self.musetalk_repo), timeout_sec=max(5, int(self.cfg.musetalk_timeout)))
            finally:
                if prev_cuda is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = prev_cuda
            return out

        try:
            out_text = _run_once(force_cpu=False)
            log_stage("MUSETALK", "Inference subprocess finished", turn_id=turn_key)
        except RuntimeError as exc:
            if (not self.cfg.force_cpu) and ("no kernel image" in str(exc).lower()):
                print("[WARN] MuseTalk CUDA kernel mismatch detected. Retrying on CPU...")
                out_text = _run_once(force_cpu=True)
            else:
                raise

        low = out_text.lower()
        if ("no kernel image is available" in low) or ("not compatible with the current pytorch installation" in low):
            if not self.cfg.force_cpu:
                print("[WARN] MuseTalk CUDA build incompatible with this GPU. Retrying on CPU...")
                out_text = _run_once(force_cpu=True)
                log_stage("MUSETALK", "CPU retry finished", turn_id=turn_key)
                low = out_text.lower()

        if "error occurred during processing:" in low:
            lines = [ln.strip() for ln in out_text.splitlines() if ln.strip()]
            tail = "\n".join(lines[-10:])
            raise RuntimeError(f"MuseTalk processing failed:\n{tail}")

        version_dir = result_dir / str(self.cfg.musetalk_version)
        if frames_only:
            frame_dirs = [d for d in version_dir.iterdir() if d.is_dir()] if version_dir.exists() else []
            frame_dirs = sorted(frame_dirs, key=lambda fp: fp.stat().st_mtime, reverse=True)
            if frame_dirs:
                log_stage("MUSETALK", f"Frame directory ready frames={len(list(frame_dirs[0].glob('*.png')))}", turn_id=turn_key)
                return frame_dirs[0]
            raise RuntimeError("MuseTalk frame output directory not found")

        out = version_dir / output_name
        if out.exists() and out.stat().st_size > 0:
            log_stage("MUSETALK", f"Video ready file={out.name} bytes={out.stat().st_size}", turn_id=turn_key)
            return out

        candidates = sorted(version_dir.glob("*.mp4"), key=lambda fp: fp.stat().st_mtime, reverse=True) if version_dir.exists() else []
        if candidates:
            return candidates[0]
        raise RuntimeError("MuseTalk output video not found")

    def _run_musetalk_frames_stage(self, audio_path, turn_key):
        log_stage("FRAME", f"Requesting MuseTalk frame generation audio={Path(audio_path).name}", turn_id=turn_key)
        frame_dir = self._run_musetalk_stage(audio_path=audio_path, turn_key=turn_key, frames_only=True)
        frames = sorted(frame_dir.glob("*.png"), key=lambda fp: fp.name)
        if not frames:
            raise RuntimeError(
                f"No MuseTalk frames found in: {frame_dir}. "
                "If your GPU is newer (e.g. RTX 50 series), use --force_cpu or install a CUDA build of PyTorch that supports your GPU architecture."
            )
        log_stage("FRAME", f"MuseTalk produced {len(frames)} frames from {frame_dir.name}", turn_id=turn_key)
        return frames


    def render_video_from_audio(self, audio_path, output_path, turn_key, template_name):
        log_stage("PIPELINE", f"Render video from audio renderer={self.cfg.renderer} template={template_name} audio={Path(audio_path).name}", turn_id=turn_key)
        if (not self.cfg.strict_pipeline) and self.cfg.renderer == "musetalk":
            musetalk_video = self._run_musetalk_stage(audio_path=audio_path, turn_key=turn_key)
            lp_dir = self.base_dir / "musetalk_liveportrait" / turn_key
            lp_video = prepare_liveportrait_base(
                args=self.args,
                image=self.image_path,
                driving_video=musetalk_video,
                base_dir=lp_dir,
            )
            if not copy_file_with_retry(lp_video, output_path):
                raise RuntimeError(f"Failed to copy rendered video to: {output_path}")
            log_stage("LIVEPORTRAIT", f"Head-motion video ready file={Path(output_path).name}", turn_id=turn_key)
            return output_path

        # Strict exact pipeline: Audio -> Face Animation (LivePortrait) -> Lip Sync (Wav2Lip)
        face_video = self._build_liveportrait_video(template_name, turn_key)
        log_stage("WAV2LIP", f"Starting lipsync source={Path(face_video).name} audio={Path(audio_path).name}", turn_id=turn_key)
        run_wav2lip(self.args, face_video=face_video, audio=audio_path, output=output_path)
        log_stage("WAV2LIP", f"Video ready file={Path(output_path).name}", turn_id=turn_key)
        return output_path

    def _make_turn_metadata(self, text):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        turn_id = f"turn_{self.turn_counter:04d}_{stamp}"
        self.turn_counter += 1
        emotion = detect_text_emotion(text)
        template_name = pick_template_for_emotion(emotion) if self.cfg.text_driven_motion else "d13"
        video_path = None if self.cfg.audio_only else (self.video_dir / f"{turn_id}.mp4")
        return {
            "turn_id": turn_id,
            "emotion": emotion,
            "template": template_name,
            "audio_path": self.audio_dir / f"{turn_id}.wav",
            "video_path": video_path,
            "video_pending": bool(video_path),
            "video_ready": self.cfg.audio_only,
        }

    def prepare_turn(self, text):
        turn = self._make_turn_metadata(text)
        log_stage("TTS", f"Preparing turn audio chars={len(text or '')}", turn_id=turn["turn_id"])
        synthesize_tts(self.args, text=text, out_media=turn["audio_path"])
        wait_for_file_stable(turn["audio_path"])
        audio_size = turn["audio_path"].stat().st_size if turn["audio_path"].exists() else 0
        log_stage("AUDIO", f"Turn audio ready file={turn['audio_path'].name} bytes={audio_size}", turn_id=turn["turn_id"])
        copy_file_with_retry(turn["audio_path"], self.latest_audio)
        return turn

    def render_video_for_turn(self, turn):
        if self.cfg.audio_only:
            turn["video_pending"] = False
            turn["video_ready"] = True
            turn["video_path"] = None
            return turn

        out_path = turn["video_path"]
        self.render_video_from_audio(
            audio_path=turn["audio_path"],
            output_path=out_path,
            turn_key=turn["turn_id"],
            template_name=turn["template"],
        )
        copy_file_with_retry(out_path, self.latest_video)
        turn["video_pending"] = False
        turn["video_ready"] = True
        return turn

    def render_turn(self, text):
        turn = self.prepare_turn(text)
        if self.cfg.audio_only:
            return turn
        return self.render_video_for_turn(turn)


def parse_args():
    p = argparse.ArgumentParser(description="Local web frontend for MuseTalk + LivePortrait talking avatar")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7861)
    p.add_argument("--image", default="apna bhai.jpg")
    p.add_argument("--checkpoint_path", default="Wav2Lip/checkpoints/wav2lip_gan.pth")
    p.add_argument("--renderer", choices=["wav2lip", "musetalk"], default="musetalk")
    p.add_argument("--wav2lip_head_motion", action=argparse.BooleanOptionalAction, default=False, help="Use LivePortrait motion video as Wav2Lip source for natural head motion (higher latency)")
    p.add_argument("--wav2lip_box", nargs=4, type=int, default=None, metavar=("Y1","Y2","X1","X2"), help="Manual Wav2Lip face box override")
    p.add_argument(
        "--strict_pipeline",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Legacy toggle; MuseTalk mode keeps this disabled.",
    )
    p.add_argument("--session_dir", default="output/realtime_avatar")
    p.add_argument("--python_bin", default="envs/py310/python.exe")
    p.add_argument("--liveportrait_repo", default="LivePortrait")
    p.add_argument("--musetalk_python_bin", default="envs/py310/python.exe")
    p.add_argument("--musetalk_repo", default="MuseTalk")
    p.add_argument("--musetalk_unet_model", default="models/musetalkV15/unet.pth")
    p.add_argument("--musetalk_unet_config", default="models/musetalkV15/musetalk.json")
    p.add_argument("--musetalk_whisper_dir", default="models/whisper")
    p.add_argument("--musetalk_version", choices=["v1", "v15"], default="v15")
    p.add_argument("--musetalk_fps", type=int, default=25)
    p.add_argument("--musetalk_gpu_id", type=int, default=0)
    p.add_argument("--musetalk_batch_size", type=int, default=4)
    p.add_argument("--musetalk_use_float16", action="store_true")
    p.add_argument("--musetalk_timeout", type=int, default=8, help="MuseTalk subprocess timeout per chunk in seconds")
    p.add_argument(
        "--musetalk_wav2lip_fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow Wav2Lip MP4 fallback when MuseTalk frame generation fails in frame-stream mode.",
    )
    p.add_argument("--tts_engine", choices=["auto", "edge", "windows"], default="auto")
    p.add_argument("--edge_voice", default="en-US-AriaNeural")
    p.add_argument("--preset", choices=["quality", "balanced", "fast"], default="quality")
    p.add_argument("--animation_region", choices=["exp", "pose", "lip", "eyes", "all"], default="all")
    p.add_argument("--text_driven_motion", action="store_true", default=False)
    p.add_argument("--liveportrait_each_turn", action="store_true")
    p.add_argument("--force_cpu", action="store_true")
    p.add_argument("--audio_only", action="store_true", help="Skip video generation, reply with TTS audio only")
    p.add_argument("--sync_video", action="store_true", help="Block until per-turn video is fully rendered")
    p.add_argument("--prewarm_base", action="store_true", help="Build default base motion at startup to reduce first-turn latency")
    p.add_argument("--use_openai", action="store_true")
    p.add_argument("--openai_model", default="gpt-4o-mini")
    p.add_argument("--openai_system_prompt", default="Reply in short, natural Hinglish for a talking avatar.")
    p.add_argument("--openai_timeout", type=int, default=20, help="OpenAI request timeout in seconds")
    p.add_argument("--webrtc", action="store_true", help="Serve generated clips through WebRTC when aiortc is installed")
    p.add_argument(
        "--stream_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="HeyGen-like mode: stream chunk frames/lipsync only and skip full end-of-turn video render",
    )
    p.add_argument(
        "--frame_stream",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Extract and stream individual frames from chunk videos for live avatar playback.",
    )
    return p.parse_args()



def pop_complete_sentences(buffer):
    completed = []
    if not buffer:
        return completed, ""

    last_end = 0
    for match in re.finditer(r"[.!?]+", buffer):
        end = match.end()
        sentence = buffer[last_end:end].strip()
        if sentence:
            completed.append(sentence)
        last_end = end

    remaining = buffer[last_end:].lstrip()
    return completed, remaining


def pop_stream_chunks(buffer, min_chars=60, max_chars=220):
    chunks = []
    remaining = (buffer or "").strip()

    while remaining:
        punct_match = re.search(r"[.!?,;:]+", remaining)
        if punct_match and punct_match.end() >= min_chars:
            cut = punct_match.end()
            chunk = remaining[:cut].strip()
            if chunk:
                chunks.append(chunk)
            remaining = remaining[cut:].lstrip()
            continue

        if len(remaining) >= max_chars:
            cut = remaining.rfind(" ", min_chars, max_chars)
            if cut <= 0:
                cut = max_chars
            chunk = remaining[:cut].strip()
            if chunk:
                chunks.append(chunk)
            remaining = remaining[cut:].lstrip()
            continue

        # Low-latency streaming split: if text already crossed min_chars, emit up to last
        # available word boundary instead of waiting for punctuation or max_chars.
        if len(remaining) >= min_chars:
            cut = remaining.rfind(" ", min_chars, len(remaining))
            if cut > 0:
                chunk = remaining[:cut].strip()
                if chunk:
                    chunks.append(chunk)
                    remaining = remaining[cut:].lstrip()
                    continue

        break

    return chunks, remaining


def build_tts_chunks(
    text,
    first_min_words=STREAM_FIRST_CHUNK_MIN_WORDS,
    first_max_words=STREAM_FIRST_CHUNK_MAX_WORDS,
    next_min_words=STREAM_NEXT_CHUNK_MIN_WORDS,
    next_max_words=STREAM_NEXT_CHUNK_MAX_WORDS,
):
    src = re.sub(r"\s+", " ", str(text or "")).strip()
    if not src:
        return []

    def _take_chunk(words, min_words, max_words):
        if not words:
            return "", []
        if len(words) <= max_words:
            return " ".join(words).strip(), []

        cut = min(len(words), max_words)
        for idx in range(max(0, min_words - 1), min(len(words), max_words)):
            if re.search(r"[.!?,;:]$", words[idx]):
                cut = idx + 1
        return " ".join(words[:cut]).strip(), words[cut:]

    words = src.split(" ")
    chunks = []
    is_first = True
    while words:
        min_words = first_min_words if is_first else next_min_words
        max_words = first_max_words if is_first else next_max_words
        chunk, words = _take_chunk(words, min_words=min_words, max_words=max_words)
        if chunk:
            chunks.append(chunk)
        is_first = False

    if len(chunks) >= 2:
        tail_words = chunks[-1].split()
        if len(tail_words) <= 6:
            chunks[-2] = (chunks[-2] + " " + chunks[-1]).strip()
            chunks.pop()

    return chunks


def estimate_text_duration_ms(text, min_ms=320):
    words = re.findall(r"\S+", str(text or ""))
    if not words:
        return int(min_ms)
    words_per_sec = 2.7
    duration_ms = int(round((len(words) / words_per_sec) * 1000.0))
    return max(int(min_ms), duration_ms)


def get_audio_duration_ms(path):
    src = Path(path)
    try:
        with wave.open(str(src), "rb") as wf:
            frames = int(wf.getnframes())
            rate = int(wf.getframerate())
            if frames > 0 and rate > 0:
                return max(STREAM_AUDIO_FALLBACK_MS, int(round((frames / float(rate)) * 1000.0)))
    except Exception:
        pass
    return estimate_text_duration_ms(src.stem.replace("_", " "), min_ms=STREAM_AUDIO_FALLBACK_MS)


class SemanticChunkBuffer:
    def __init__(self, target_ms=STREAM_SEMANTIC_TARGET_MS, min_ms=STREAM_SEMANTIC_MIN_MS, max_ms=STREAM_SEMANTIC_MAX_MS, force_chars=STREAM_FORCE_CHUNK_CHARS):
        self.target_ms = int(target_ms)
        self.min_ms = int(min_ms)
        self.max_ms = int(max_ms)
        self.force_chars = max(1, int(force_chars))
        self.buffer = ""

    def push(self, delta, is_final=False):
        if delta:
            self.buffer += str(delta)
        source = re.sub(r"\s+", " ", self.buffer).strip()
        log_stage("SEMANTIC", f"push delta_chars={len(str(delta or ''))} buffer_chars={len(source)} final={str(bool(is_final)).lower()}")
        emitted = []

        while True:
            chunk, rest = self._pop_ready_chunk(force=is_final)
            if not chunk:
                break
            emitted.append(chunk)
            log_stage("SEMANTIC", f"emitted chunk chars={len(chunk)} remaining_chars={len(rest)} final={str(bool(is_final)).lower()}")
            self.buffer = rest
            if not self.buffer.strip():
                self.buffer = ""
                break
            if is_final:
                continue

        if not emitted:
            forced_chunk, forced_rest = self._force_fallback_chunk(force=is_final)
            if forced_chunk:
                emitted.append(forced_chunk)
                self.buffer = forced_rest
                log_stage("SEMANTIC", f"forced fallback chunk chars={len(forced_chunk)} remaining_chars={len(forced_rest)} final={str(bool(is_final)).lower()}")

        return emitted

    def flush(self):
        return self.push("", is_final=True)

    def _force_fallback_chunk(self, force=False):
        source = re.sub(r"\s+", " ", self.buffer).strip()
        if not source:
            return "", ""
        if (not force) and len(source) < self.force_chars:
            return "", source

        cut = -1
        if force:
            cut = len(source)
        else:
            window_end = min(len(source), max(self.force_chars + 40, self.force_chars))
            cut = source.rfind(" ", self.force_chars, window_end)
            if cut <= 0:
                cut = source.find(" ", self.force_chars)
            if cut <= 0:
                cut = min(len(source), self.force_chars)
        chunk = source[:cut].strip()
        rest = source[cut:].lstrip() if cut < len(source) else ""
        return chunk, rest

    def _pop_ready_chunk(self, force=False):
        source = re.sub(r"\s+", " ", self.buffer).strip()
        if not source:
            return "", ""

        candidates = []
        for match in re.finditer(r"[.!?,;:]", source):
            idx = match.end()
            candidates.append((idx, match.group(0)))
        for match in re.finditer(r"\s+", source):
            idx = match.start()
            if idx > 0:
                candidates.append((idx, " "))
        if len(source) not in [idx for idx, _ in candidates]:
            candidates.append((len(source), "eof"))
        candidates = sorted(set(candidates), key=lambda item: item[0])

        best_ready = None
        best_force = None
        for idx, marker in candidates:
            candidate = source[:idx].strip()
            if not candidate:
                continue
            dur_ms = estimate_text_duration_ms(candidate)
            score = abs(dur_ms - self.target_ms)
            is_punct = marker != " "
            if dur_ms >= self.min_ms and (is_punct or dur_ms >= self.target_ms):
                rank = (score, 0 if is_punct else 1, -idx)
                if best_ready is None or rank < best_ready[0]:
                    best_ready = (rank, candidate, source[idx:].lstrip())
            if dur_ms <= self.max_ms:
                rank = (0 if is_punct else 1, score, -idx)
                if best_force is None or rank < best_force[0]:
                    best_force = (rank, candidate, source[idx:].lstrip())
            elif best_force is None:
                best_force = ((2, score, -idx), candidate, source[idx:].lstrip())
                break

        if best_ready is not None:
            return best_ready[1], best_ready[2]
        if force and best_force is not None:
            return best_force[1], best_force[2]
        return "", source


def main():
    load_env_file()
    cfg = parse_args()
    frame_stream_flag_explicit = any(arg in ("--frame_stream", "--no-frame_stream") for arg in sys.argv[1:])
    # Keep live avatar mode on by default; allow explicit CLI override with --no-frame_stream.
    # Respect selected renderer; only keep strict pipeline disabled for MuseTalk live mode.
    if cfg.renderer == "musetalk":
        cfg.strict_pipeline = False
    if cfg.renderer == "wav2lip" and (not cfg.audio_only) and bool(getattr(cfg, "wav2lip_head_motion", False)) and (not cfg.text_driven_motion):
        # Enable emotion/template-driven motion so Wav2Lip + LivePortrait does not look static.
        cfg.text_driven_motion = True
    if cfg.renderer == "wav2lip" and bool(getattr(cfg, "wav2lip_head_motion", False)) and cfg.animation_region == "lip":
        cfg.animation_region = "all"
    if not cfg.use_openai and os.environ.get("OPENAI_API_KEY"):
        cfg.use_openai = True
    env_openai_model = str(os.environ.get("OPENAI_MODEL", "") or "").strip()
    if env_openai_model and cfg.openai_model == "gpt-4o-mini":
        cfg.openai_model = env_openai_model
    env_openai_prompt = str(os.environ.get("OPENAI_SYSTEM_PROMPT", "") or "").strip()
    if env_openai_prompt and cfg.openai_system_prompt == "Reply in short, natural Hinglish for a talking avatar.":
        cfg.openai_system_prompt = env_openai_prompt
    openai_api_url = normalize_openai_api_url(os.environ.get("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions"))
    engine = AvatarEngine(cfg)
    if cfg.renderer == "musetalk" and (not bool(getattr(engine, "musetalk_runtime_ready", True))):
        cfg.renderer = "wav2lip"
        cfg.frame_stream = True
        cfg.wav2lip_head_motion = False
        print("[WARN] MuseTalk runtime unavailable. Auto-switching to Wav2Lip frame-stream lipsync (--renderer wav2lip --frame_stream).")
    if cfg.prewarm_base and (not cfg.audio_only):
        try:
            print("Prewarming base motion (d13) for lower first-turn latency...")
            with engine.lock:
                engine._build_liveportrait_video("d13", "warmup")
            print("Prewarm complete.")
        except Exception as exc:
            print(f"[WARN] Prewarm failed: {exc}")
    api_key = os.environ.get("OPENAI_API_KEY", "") if cfg.use_openai else ""
    responder = OpenAIResponder(
        model=cfg.openai_model,
        api_key=api_key,
        system_prompt=cfg.openai_system_prompt,
        api_url=openai_api_url,
        request_timeout_sec=cfg.openai_timeout,
        stream_timeout_sec=max(25, cfg.openai_timeout),
    )

    render_jobs = {}
    render_jobs_lock = threading.Lock()
    stream_jobs = {}
    stream_jobs_lock = threading.RLock()
    stream_event_subscribers = {}
    stream_event_subscribers_lock = threading.Lock()

    def publish_stream_event(turn_id, event_type, payload):
        event = {
            "type": str(event_type),
            "turn_id": str(turn_id),
            "ts": int(time.time() * 1000),
            "payload": dict(payload or {}),
        }
        with stream_jobs_lock:
            job = stream_jobs.get(turn_id)
            if job is not None:
                history = job.setdefault("event_history", [])
                history.append(event)
                if len(history) > 400:
                    del history[:-400]
        with stream_event_subscribers_lock:
            subscribers = list(stream_event_subscribers.get(turn_id, []))
        for q in subscribers:
            try:
                q.put_nowait(event)
            except Exception:
                pass

    def register_stream_event_subscriber(turn_id):
        q = queue.Queue()
        with stream_jobs_lock:
            job = stream_jobs.get(turn_id)
            history = list(job.get("event_history", [])) if job else []
        with stream_event_subscribers_lock:
            stream_event_subscribers.setdefault(turn_id, []).append(q)
        return q, history

    def unregister_stream_event_subscriber(turn_id, q):
        with stream_event_subscribers_lock:
            arr = stream_event_subscribers.get(turn_id)
            if not arr:
                return
            stream_event_subscribers[turn_id] = [item for item in arr if item is not q]
            if not stream_event_subscribers[turn_id]:
                stream_event_subscribers.pop(turn_id, None)

    def mark_stream_file(job, kind, idx, path):
        if not job:
            return
        files = job.setdefault("files", {})
        key = (str(kind), int(idx))
        bucket = files.setdefault(key, [])
        p = str(Path(path))
        if p not in bucket:
            bucket.append(p)

    def cleanup_stream_chunk_files(job, kind, idx):
        if not job:
            return
        files = job.setdefault("files", {})
        key = (str(kind), int(idx))
        for fp in files.pop(key, []):
            safe_unlink(fp)

    def cleanup_stream_turn(turn_id):
        with stream_jobs_lock:
            job = stream_jobs.get(turn_id)
            if not job:
                return
            if job.get("cleaned_up"):
                return
            job["cleaned_up"] = True
        base = engine.session_dir
        safe_rmtree(base / "stream_audio" / turn_id)
        safe_rmtree(base / "stream_video" / turn_id)
        safe_rmtree(base / "stream_frames" / turn_id)
        safe_rmtree(base / "musetalk_jobs" / turn_id)

    def cleanup_expired_stream_jobs():
        while True:
            time.sleep(15)
            now = time.time()
            expired = []
            with stream_jobs_lock:
                for turn_id, job in list(stream_jobs.items()):
                    if not job.get("done"):
                        continue
                    if job.get("render_pending") or (not job.get("render_done")):
                        continue
                    if job.get("video_pending") and (not job.get("video_ready")):
                        continue
                    finished_at = float(job.get("finished_at", job.get("created_at", now)) or now)
                    if now - finished_at >= STREAM_FILE_RETENTION_SEC:
                        expired.append(turn_id)
            for turn_id in expired:
                cleanup_stream_turn(turn_id)

    threading.Thread(target=cleanup_expired_stream_jobs, daemon=True).start()

    webrtc_enabled = bool(cfg.webrtc and RTCPeerConnection and RTCSessionDescription and MediaPlayer)
    webrtc_pcs = set()
    webrtc_pcs_lock = threading.Lock()

    def resolve_video_media_path(media_url):
        req_path = urllib.parse.urlparse(str(media_url or "")).path
        if req_path.startswith("/latest.mp4"):
            if engine.latest_video.exists():
                return engine.latest_video
            return None

        if req_path.startswith("/media/video/"):
            turn_id = req_path[len("/media/video/") :].strip()
            if (not turn_id) or (turn_id != Path(turn_id).name):
                return None
            src = engine.video_dir / f"{turn_id}.mp4"
            return src if src.exists() else None

        if req_path.startswith("/media/stream-video/"):
            rel = req_path[len("/media/stream-video/") :].strip()
            parts = [p for p in rel.split("/") if p]
            if len(parts) != 2:
                return None
            stream_turn, idx_part = parts
            if (stream_turn != Path(stream_turn).name) or (not idx_part.isdigit()):
                return None
            src = engine.session_dir / "stream_video" / stream_turn / f"chunk_{int(idx_part):04d}.mp4"
            return src if src.exists() else None

        return None

    async def build_webrtc_answer(offer_sdp, offer_type, media_url):
        video_path = resolve_video_media_path(media_url)
        if video_path is None:
            raise RuntimeError("Video source not found for WebRTC")

        pc = RTCPeerConnection()
        player = MediaPlayer(str(video_path))

        if player.video:
            pc.addTrack(player.video)
        if player.audio:
            pc.addTrack(player.audio)

        @pc.on("connectionstatechange")
        async def _on_connectionstatechange():
            state = pc.connectionState
            if state in ("closed", "failed", "disconnected"):
                with webrtc_pcs_lock:
                    if pc in webrtc_pcs:
                        webrtc_pcs.discard(pc)
                await pc.close()

        with webrtc_pcs_lock:
            webrtc_pcs.add(pc)

        await pc.setRemoteDescription(RTCSessionDescription(sdp=offer_sdp, type=offer_type))
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
        }

    def start_async_video_render(turn_payload):
        turn_id = turn_payload["turn_id"]
        with render_jobs_lock:
            render_jobs[turn_id] = {
                "status": "pending",
                "error": "",
                "created_at": time.time(),
            }

        def _worker(payload):
            try:
                with engine.lock:
                    engine.render_video_for_turn(payload)
                with render_jobs_lock:
                    render_jobs[turn_id] = {
                        "status": "ready",
                        "error": "",
                        "created_at": render_jobs.get(turn_id, {}).get("created_at", time.time()),
                        "finished_at": time.time(),
                    }
            except Exception as worker_exc:
                with render_jobs_lock:
                    render_jobs[turn_id] = {
                        "status": "error",
                        "error": str(worker_exc),
                        "created_at": render_jobs.get(turn_id, {}).get("created_at", time.time()),
                        "finished_at": time.time(),
                    }

        threading.Thread(target=_worker, args=(dict(turn_payload),), daemon=True).start()

    def _scan_stream_chunk_files(base_dir, pattern, group_idx=1):
        found = {}
        if not base_dir.exists():
            return found
        rx = re.compile(pattern)
        for fp in sorted(base_dir.iterdir(), key=lambda item: item.name):
            if not fp.is_file():
                continue
            match = rx.match(fp.name)
            if not match:
                continue
            idx = int(match.group(group_idx))
            prev = found.get(idx)
            if prev is None or fp.suffix.lower() == ".wav" or fp.suffix.lower() == ".png":
                found[idx] = fp
        return found

    def _frame_chunk_index_lookup(job):
        lookup = {}
        for frame in list(job.get("frame_chunks", [])):
            try:
                idx = int(frame.get("idx", -1))
                chunk_idx = int(frame.get("chunk_idx", -1))
            except Exception:
                continue
            if idx >= 0 and chunk_idx >= 0:
                lookup[idx] = chunk_idx
        if lookup:
            return lookup

        current_idx = 0
        for audio_chunk in sorted(list(job.get("audio_chunks", [])), key=lambda item: int(item.get("idx", -1))):
            try:
                chunk_idx = int(audio_chunk.get("idx", -1))
                frame_count = max(0, int(audio_chunk.get("frame_count", 0) or 0))
            except Exception:
                continue
            for _ in range(frame_count):
                lookup[current_idx] = chunk_idx
                current_idx += 1
        return lookup

    def _build_live_status_payload(turn_id, after_audio, after_video, after_frame):
        with stream_jobs_lock:
            job = dict(stream_jobs.get(turn_id) or {})
        if not job:
            return None

        audio_meta = {}
        for chunk in list(job.get("audio_chunks", [])):
            try:
                audio_meta[int(chunk.get("idx", -1))] = dict(chunk)
            except Exception:
                continue
        video_meta = {}
        for chunk in list(job.get("video_chunks", [])):
            try:
                video_meta[int(chunk.get("idx", -1))] = dict(chunk)
            except Exception:
                continue
        frame_meta = {}
        for frame in list(job.get("frame_chunks", [])):
            try:
                frame_meta[int(frame.get("idx", -1))] = dict(frame)
            except Exception:
                continue
        frame_chunk_lookup = _frame_chunk_index_lookup(job)

        stream_audio_dir = engine.session_dir / "stream_audio" / turn_id
        stream_video_dir = engine.session_dir / "stream_video" / turn_id
        stream_frame_dir = engine.session_dir / "stream_frames" / turn_id

        audio_files = _scan_stream_chunk_files(stream_audio_dir, r"^chunk_(\d+)\.(wav|mp3)$")
        video_files = _scan_stream_chunk_files(stream_video_dir, r"^chunk_(\d+)\.mp4$")
        frame_files = _scan_stream_chunk_files(stream_frame_dir, r"^frame_(\d+)\.(png|jpg|jpeg)$")

        audio_chunks = []
        for idx, fp in sorted(audio_files.items()):
            if after_audio >= 0 and idx <= after_audio:
                continue
            payload = dict(audio_meta.get(idx) or {})
            payload.setdefault("idx", idx)
            payload.setdefault("text", "")
            payload["audio_url"] = f"/media/stream-audio/{urllib.parse.quote(turn_id)}/{idx}?ts={int(fp.stat().st_mtime_ns // 1000000)}"
            payload.setdefault("frame_count", 0)
            payload.setdefault("ready", True)
            audio_chunks.append(payload)

        video_chunks = []
        for idx, fp in sorted(video_files.items()):
            if after_video >= 0 and idx <= after_video:
                continue
            payload = dict(video_meta.get(idx) or {})
            payload.setdefault("idx", idx)
            payload.setdefault("text", (audio_meta.get(idx) or {}).get("text", ""))
            payload["video_url"] = f"/media/stream-video/{urllib.parse.quote(turn_id)}/{idx}?ts={int(fp.stat().st_mtime_ns // 1000000)}"
            if idx in audio_files:
                payload.setdefault("audio_url", f"/media/stream-audio/{urllib.parse.quote(turn_id)}/{idx}?ts={int(audio_files[idx].stat().st_mtime_ns // 1000000)}")
            video_chunks.append(payload)

        frame_chunks = []
        for idx, fp in sorted(frame_files.items()):
            if after_frame >= 0 and idx <= after_frame:
                continue
            payload = dict(frame_meta.get(idx) or {})
            payload.setdefault("idx", idx)
            payload.setdefault("chunk_idx", int(frame_chunk_lookup.get(idx, -1)))
            payload["frame_url"] = f"/media/stream-frame/{urllib.parse.quote(turn_id)}/{idx}?ts={int(fp.stat().st_mtime_ns // 1000000)}"
            frame_chunks.append(payload)

        job["audio_count"] = len(audio_files)
        job["video_count"] = len(video_files)
        job["frame_count"] = max(len(frame_files), len(frame_meta))
        job["audio_chunks"] = audio_chunks
        job["video_chunks"] = video_chunks
        job["frame_chunks"] = frame_chunks
        if not audio_chunks and not frame_chunks and not video_chunks and not job.get("done"):
            return job

        return job

    def start_stream_talk_job(turn_id, user_msg):
        log_stage("PIPELINE", f"Starting realtime stream prompt_chars={len(user_msg or '')}", turn_id=turn_id)
        stream_audio_dir = engine.session_dir / "stream_audio" / turn_id
        stream_video_dir = engine.session_dir / "stream_video" / turn_id
        stream_frame_dir = engine.session_dir / "stream_frames" / turn_id
        stream_audio_dir.mkdir(parents=True, exist_ok=True)
        stream_video_dir.mkdir(parents=True, exist_ok=True)
        stream_frame_dir.mkdir(parents=True, exist_ok=True)

        if cfg.renderer == "wav2lip":
            base_frame_fps = float(getattr(engine.args, "wav2lip_fps", 12) or 12)
        elif cfg.renderer == "musetalk":
            base_frame_fps = float(cfg.musetalk_fps or 25)
        else:
            base_frame_fps = 25.0
        if base_frame_fps <= 0:
            base_frame_fps = 25.0
        frame_delay_ms = int(round(1000.0 / base_frame_fps))

        with stream_jobs_lock:
            stream_jobs[turn_id] = {
                "status": "pending",
                "error": "",
                "partial_reply": "",
                "reply": "",
                "audio_chunks": [],
                "video_chunks": [],
                "frame_chunks": [],
                "frame_idx": 0,
                "created_at": time.time(),
                "done": False,
                "render_pending": True,
                "render_done": False,
                "video_turn_id": "",
                "video_pending": False,
                "video_ready": False,
                "video_url": "",
                "pipeline": PIPELINE_FLOW,
                "frame_fps": base_frame_fps,
                "frame_delay_ms": frame_delay_ms,
                "semantic_target_ms": STREAM_SEMANTIC_TARGET_MS,
                "prefetch_ahead": STREAM_PREFETCH_AHEAD,
                "scheduler_clock_ms": 0,
                "next_release_ms": 0,
                "prefetched_until_idx": -1,
                "queued_chunks": 0,
                "last_live_status_snapshot": None,
                "event_history": [],
                "files": {},
                "cleaned_up": False,
                "musetalk_disabled": (cfg.renderer == "musetalk" and (not bool(getattr(engine, "musetalk_runtime_ready", True)))),
            }

        publish_stream_event(turn_id, "pipeline", {"stage": "start", "flow": PIPELINE_FLOW})

        chunk_queue = queue.Queue(maxsize=max(2, STREAM_PREFETCH_AHEAD + 1))
        producer_done = threading.Event()
        render_done = threading.Event()
        render_error = {"error": None}
        frame_tasks_lock = threading.Lock()
        frame_tasks_inflight = {"count": 0}

        def _try_complete_render():
            with frame_tasks_lock:
                inflight = int(frame_tasks_inflight.get("count", 0) or 0)
            if (not producer_done.is_set()) or chunk_queue.qsize() > 0 or inflight > 0:
                return False
            with stream_jobs_lock:
                job = stream_jobs.get(turn_id)
                if job:
                    job["render_pending"] = False
                    job["render_done"] = True
                    job["queued_chunks"] = 0
                    if not job.get("video_pending"):
                        job["finished_at"] = time.time()
            render_done.set()
            return True

        def _extract_frames_from_video(video_path, start_idx, chunk_idx):
            if cv2 is None:
                return [], start_idx

            cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
            if not cap.isOpened():
                return [], start_idx

            frame_entries = []
            idx_local = int(start_idx)
            ts = int(time.time() * 1000)
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        break
                    frame_file_png = stream_frame_dir / f"frame_{idx_local:06d}.png"
                    write_ok = False
                    try:
                        write_ok = bool(cv2.imwrite(str(frame_file_png), frame, [cv2.IMWRITE_PNG_COMPRESSION, 1]))
                    except Exception:
                        write_ok = False
                    if write_ok:
                        frame_file = frame_file_png
                    else:
                        frame_file = stream_frame_dir / f"frame_{idx_local:06d}.jpg"
                        if not cv2.imwrite(str(frame_file), frame, [cv2.IMWRITE_JPEG_QUALITY, 98]):
                            break
                    frame_entries.append(
                        {
                            "idx": idx_local,
                            "chunk_idx": chunk_idx,
                            "frame_url": f"/media/stream-frame/{urllib.parse.quote(turn_id)}/{idx_local}?ts={ts}",
                        }
                    )
                    log_stage("FRAME", "Extracted frame for stream playback", turn_id=turn_id, chunk_idx=chunk_idx)
                    idx_local += 1
            finally:
                cap.release()

            return frame_entries, idx_local

        def _reserve_schedule_window(text, audio_duration_ms):
            with stream_jobs_lock:
                job = stream_jobs.get(turn_id)
                if not job:
                    return 0, max(int(audio_duration_ms), STREAM_AUDIO_FALLBACK_MS)
                start_ms = int(job.get("scheduler_clock_ms", 0) or 0)
                duration_ms = max(int(audio_duration_ms or 0), estimate_text_duration_ms(text, min_ms=STREAM_AUDIO_FALLBACK_MS))
                end_ms = start_ms + duration_ms
                job["scheduler_clock_ms"] = end_ms
                job["next_release_ms"] = end_ms
                log_stage("SCHED", f"Reserved playback window start_ms={start_ms} duration_ms={duration_ms} text_chars={len(text or '')}", turn_id=turn_id)
                return start_ms, duration_ms

        def _mark_prefetched(idx):
            with stream_jobs_lock:
                job = stream_jobs.get(turn_id)
                if not job:
                    return
                job["queued_chunks"] = chunk_queue.qsize()
                if idx > int(job.get("prefetched_until_idx", -1)):
                    job["prefetched_until_idx"] = idx
                log_stage("PREFETCH", f"Chunk queued queued_chunks={job['queued_chunks']} prefetched_until={job['prefetched_until_idx']}", turn_id=turn_id, chunk_idx=idx)

        def _render_chunk_media(text, idx, audio_path, video_path, template_name):
            frame_entries = []
            next_frame_idx = None

            if cfg.audio_only:
                return frame_entries, next_frame_idx

            if cfg.frame_stream and cfg.renderer == "musetalk":
                with stream_jobs_lock:
                    job = stream_jobs.get(turn_id)
                    if not job:
                        return frame_entries, next_frame_idx
                    start_frame_idx = int(job.get("frame_idx", 0))
                    musetalk_disabled = bool(job.get("musetalk_disabled", False))

                used_fallback = False
                try:
                    if musetalk_disabled:
                        raise RuntimeError("MuseTalk disabled for this turn after previous failure")
                    with engine.lock:
                        frame_paths = engine._run_musetalk_frames_stage(
                            audio_path=audio_path,
                            turn_key=f"{turn_id}_{idx:04d}",
                        )

                    ts = int(time.time() * 1000)
                    idx_local = start_frame_idx
                    for src_frame in frame_paths:
                        dst = stream_frame_dir / f"frame_{idx_local:06d}.png"
                        copy_file_with_retry(src_frame, dst)
                        with stream_jobs_lock:
                            job = stream_jobs.get(turn_id)
                            if job:
                                mark_stream_file(job, "frame", idx_local, dst)
                        frame_entries.append(
                            {
                                "idx": idx_local,
                                "chunk_idx": idx,
                                "frame_url": f"/media/stream-frame/{urllib.parse.quote(turn_id)}/{idx_local}?ts={ts}",
                            }
                        )
                        idx_local += 1
                    next_frame_idx = idx_local
                except Exception as mt_exc:
                    if not bool(getattr(cfg, "musetalk_wav2lip_fallback", False)):
                        log_stage(
                            "FRAME",
                            f"MuseTalk frame render failed; Wav2Lip fallback disabled: {mt_exc}",
                            turn_id=turn_id,
                            chunk_idx=idx,
                        )
                        publish_stream_event(turn_id, "pipeline", {"stage": "frame_render_failed", "chunk_idx": idx})
                        return frame_entries, next_frame_idx

                    used_fallback = True
                    with stream_jobs_lock:
                        job = stream_jobs.get(turn_id)
                        if job is not None:
                            job["musetalk_disabled"] = True
                    log_stage("FRAME", f"MuseTalk unavailable for chunk; using Wav2Lip fallback: {mt_exc}", turn_id=turn_id, chunk_idx=idx)
                    publish_stream_event(turn_id, "pipeline", {"stage": "frame_render_timeout_fallback", "chunk_idx": idx})
                    with engine.lock:
                        if not engine.checkpoint_path.exists():
                            raise RuntimeError(
                                f"MuseTalk failed and Wav2Lip fallback is unavailable. Missing checkpoint: {engine.checkpoint_path}"
                            )
                        run_wav2lip(engine.args, face_video=engine.wav2lip_face_image, audio=audio_path, output=video_path)
                        copy_file_with_retry(video_path, engine.latest_video)

                if used_fallback:
                    with stream_jobs_lock:
                        job = stream_jobs.get(turn_id)
                        if job:
                            mark_stream_file(job, "video", idx, video_path)
                    frame_entries, next_frame_idx = _extract_frames_from_video(video_path, start_frame_idx, idx)
                return frame_entries, next_frame_idx

            with engine.lock:
                if cfg.renderer == "wav2lip":
                    face_source = engine.wav2lip_face_image
                    prev_box = getattr(engine.args, "wav2lip_box", None)
                    prev_fps = getattr(engine.args, "wav2lip_fps", None)
                    try:
                        motion_template = template_name if cfg.text_driven_motion else "d13"
                        if bool(getattr(cfg, "wav2lip_head_motion", False)):
                            face_source = engine.stream_face_source
                            if idx > 0:
                                face_source = engine._build_liveportrait_video(motion_template, f"{turn_id}_{idx:04d}")
                        if str(face_source).lower().endswith(".mp4"):
                            engine.args.wav2lip_box = None
                            engine.args.wav2lip_fps = None
                        else:
                            engine.args.wav2lip_box = prev_box or engine.args.wav2lip_box
                        run_wav2lip(engine.args, face_video=face_source, audio=audio_path, output=video_path)
                    finally:
                        engine.args.wav2lip_box = prev_box
                        engine.args.wav2lip_fps = prev_fps
                else:
                    engine.render_video_from_audio(
                        audio_path=audio_path,
                        output_path=video_path,
                        turn_key=f"{turn_id}_{idx:04d}",
                        template_name=template_name,
                    )
                copy_file_with_retry(video_path, engine.latest_video)
                with stream_jobs_lock:
                    job = stream_jobs.get(turn_id)
                    if job:
                        mark_stream_file(job, "video", idx, video_path)

            if cfg.frame_stream:
                with stream_jobs_lock:
                    job = stream_jobs.get(turn_id)
                    if not job:
                        return frame_entries, next_frame_idx
                    start_frame_idx = int(job.get("frame_idx", 0))
                return _extract_frames_from_video(video_path, start_frame_idx, idx)
            return frame_entries, next_frame_idx

        def _publish_chunk(idx, text, audio_path, audio_duration_ms, schedule_start_ms, frame_entries, next_frame_idx, video_ready, publish_audio=True):
            ts = int(time.time() * 1000)
            chunk_payload = {
                "idx": idx,
                "text": text,
                "audio_url": f"/media/stream-audio/{urllib.parse.quote(turn_id)}/{idx}?ts={ts}",
                "frame_count": len(frame_entries),
                "ready": True,
                "release_at_ms": int(schedule_start_ms),
                "duration_ms": int(audio_duration_ms),
                "frame_delay_ms": int(frame_delay_ms),
                "prefetched": True,
            }
            with stream_jobs_lock:
                job = stream_jobs.get(turn_id)
                if not job:
                    return
                if frame_entries:
                    job["frame_chunks"].extend(frame_entries)
                    job["frame_idx"] = int(next_frame_idx or job.get("frame_idx", 0))
                    log_stage("FRAME", f"Queued {len(frame_entries)} frames for browser frame stream", turn_id=turn_id, chunk_idx=idx)
                if publish_audio and not any(int(c.get("idx", -1)) == int(idx) for c in job["audio_chunks"]):
                    job["audio_chunks"].append(chunk_payload)
                    log_stage("STREAM", f"Queued audio chunk for browser release_at_ms={chunk_payload['release_at_ms']} duration_ms={chunk_payload['duration_ms']}", turn_id=turn_id, chunk_idx=idx)
                    publish_stream_event(turn_id, "audio_chunk", chunk_payload)
                for frame_entry in frame_entries:
                    publish_stream_event(turn_id, "pipeline", {"stage": "webrtc_send", "chunk_idx": idx, "frame_idx": int(frame_entry.get("idx", -1))})
                    publish_stream_event(turn_id, "frame_chunk", frame_entry)
                job["queued_chunks"] = chunk_queue.qsize()
                if ((not cfg.frame_stream) or (cfg.frame_stream and (not frame_entries))) and video_ready:
                    video_payload = {
                            "idx": idx,
                            "text": text,
                            "video_url": f"/media/stream-video/{urllib.parse.quote(turn_id)}/{idx}?ts={ts}",
                            "audio_url": chunk_payload["audio_url"],
                            "release_at_ms": int(schedule_start_ms),
                            "duration_ms": int(audio_duration_ms),
                        }
                    job["video_chunks"].append(video_payload)
                    publish_stream_event(turn_id, "video_chunk", video_payload)

        def _process_chunk(item):
            idx = int(item["idx"])
            text = str(item["text"] or "").strip()
            if not text:
                return

            audio_path = stream_audio_dir / f"chunk_{idx:04d}.wav"
            video_path = stream_video_dir / f"chunk_{idx:04d}.mp4"
            template_name = "d13"
            if cfg.text_driven_motion:
                template_name = pick_template_for_emotion(detect_text_emotion(text))

            log_stage("TTS", f"Chunk TTS start chars={len(text)}", turn_id=turn_id, chunk_idx=idx)
            publish_stream_event(turn_id, "pipeline", {"stage": "tts", "chunk_idx": idx, "text": text[:80]})
            synthesize_tts(engine.args, text=text, out_media=audio_path)
            wait_for_file_stable(audio_path)
            audio_size = audio_path.stat().st_size if audio_path.exists() else 0
            log_stage("AUDIO", f"Chunk audio ready file={audio_path.name} bytes={audio_size}", turn_id=turn_id, chunk_idx=idx)
            copy_file_with_retry(audio_path, engine.latest_audio)
            audio_duration_ms = get_audio_duration_ms(audio_path)
            schedule_start_ms, scheduled_duration_ms = _reserve_schedule_window(text, audio_duration_ms)

            with stream_jobs_lock:
                job = stream_jobs.get(turn_id)
                if not job:
                    return
                mark_stream_file(job, "audio", idx, audio_path)

            publish_audio_immediately = bool(cfg.renderer != "wav2lip")
            if publish_audio_immediately:
                _publish_chunk(
                    idx=idx,
                    text=text,
                    audio_path=audio_path,
                    audio_duration_ms=scheduled_duration_ms,
                    schedule_start_ms=schedule_start_ms,
                    frame_entries=[],
                    next_frame_idx=None,
                    video_ready=False,
                    publish_audio=True,
                )

            if cfg.audio_only:
                if not publish_audio_immediately:
                    _publish_chunk(
                        idx=idx,
                        text=text,
                        audio_path=audio_path,
                        audio_duration_ms=scheduled_duration_ms,
                        schedule_start_ms=schedule_start_ms,
                        frame_entries=[],
                        next_frame_idx=None,
                        video_ready=False,
                        publish_audio=True,
                    )
                _try_complete_render()
                return

            with frame_tasks_lock:
                frame_tasks_inflight["count"] = int(frame_tasks_inflight.get("count", 0) or 0) + 1

            def _render_frame_task():
                try:
                    publish_stream_event(turn_id, "pipeline", {"stage": "frame_render", "chunk_idx": idx})
                    frame_entries, next_frame_idx = _render_chunk_media(text, idx, audio_path, video_path, template_name)
                    log_stage("PIPELINE", f"Chunk render finished frames={len(frame_entries)} video_ready={str((not cfg.audio_only)).lower()}", turn_id=turn_id, chunk_idx=idx)
                    if not frame_entries:
                        publish_stream_event(turn_id, "pipeline", {"stage": "frame_render_timeout_fallback", "chunk_idx": idx})
                    _publish_chunk(
                        idx=idx,
                        text=text,
                        audio_path=audio_path,
                        audio_duration_ms=scheduled_duration_ms,
                        schedule_start_ms=schedule_start_ms,
                        frame_entries=frame_entries,
                        next_frame_idx=next_frame_idx,
                        video_ready=(not cfg.audio_only),
                        publish_audio=(not publish_audio_immediately),
                    )
                except Exception as exc:
                    render_error["error"] = exc
                    with stream_jobs_lock:
                        job = stream_jobs.get(turn_id)
                        if job:
                            job["status"] = "error"
                            if not job.get("error"):
                                job["error"] = str(exc)
                            job["done"] = True
                            job["queued_chunks"] = 0
                            job["finished_at"] = time.time()
                    producer_done.set()
                    render_done.set()
                finally:
                    with frame_tasks_lock:
                        frame_tasks_inflight["count"] = max(0, int(frame_tasks_inflight.get("count", 0) or 0) - 1)
                    _try_complete_render()

            threading.Thread(target=_render_frame_task, daemon=True).start()

        def _enqueue_chunk(idx, text):
            clean = str(text or "").strip()
            if not clean:
                log_stage("SEMANTIC", "Chunk dropped because text is empty after trim", turn_id=turn_id, chunk_idx=idx)
                return False
            log_stage("SEMANTIC", f"Chunk accepted chars={len(clean)} preview={clean[:80]!r}", turn_id=turn_id, chunk_idx=idx)
            chunk_queue.put({"idx": int(idx), "text": clean})
            log_stage("TTSQ", f"Chunk queued for TTS/render queue_size={chunk_queue.qsize()}", turn_id=turn_id, chunk_idx=idx)
            _mark_prefetched(int(idx))
            return True

        def _render_worker():
            while True:
                try:
                    item = chunk_queue.get(timeout=0.05)
                except queue.Empty:
                    if producer_done.is_set():
                        if _try_complete_render() or render_done.is_set():
                            return
                    continue

                try:
                    log_stage("PIPELINE", "Dequeued chunk for TTS/render", turn_id=turn_id, chunk_idx=item.get("idx"))
                    _process_chunk(item)
                except Exception as exc:
                    render_error["error"] = exc
                    with stream_jobs_lock:
                        job = stream_jobs.get(turn_id)
                        if job:
                            job["status"] = "error"
                            if not job.get("error"):
                                job["error"] = str(exc)
                            job["render_pending"] = False
                            job["render_done"] = True
                            job["done"] = True
                            job["queued_chunks"] = 0
                            job["finished_at"] = time.time()
                    producer_done.set()
                    render_done.set()
                    return
                finally:
                    chunk_queue.task_done()
                    with stream_jobs_lock:
                        job = stream_jobs.get(turn_id)
                        if job:
                            job["queued_chunks"] = chunk_queue.qsize()
                    if producer_done.is_set():
                        _try_complete_render()

        def _worker():
            render_thread = threading.Thread(target=_render_worker, daemon=True)
            render_thread.start()

            try:
                semantic_buffer = SemanticChunkBuffer()
                parts = []
                next_chunk_idx = 0

                single_chunk_wav2lip = bool(cfg.renderer == "wav2lip")

                for delta in responder.stream_reply(user_msg):
                    if not delta:
                        continue
                    parts.append(delta)
                    with stream_jobs_lock:
                        job = stream_jobs.get(turn_id)
                        if job:
                            job["partial_reply"] = "".join(parts).strip()
                    publish_stream_event(turn_id, "partial", {"partial_reply": "".join(parts).strip()})
                    if single_chunk_wav2lip:
                        continue
                    ready_chunks = semantic_buffer.push(delta, is_final=False)
                    if ready_chunks:
                        log_stage("SEMANTIC", f"Buffer emitted {len(ready_chunks)} chunk(s) during stream", turn_id=turn_id)
                    for chunk_text in ready_chunks:
                        if _enqueue_chunk(next_chunk_idx, chunk_text):
                            next_chunk_idx += 1

                final_reply = "".join(parts).strip() or user_msg
                if single_chunk_wav2lip:
                    trailing_chunks = [final_reply] if final_reply else []
                    if trailing_chunks:
                        log_stage("SEMANTIC", "Wav2Lip mode: using single chunk for smoother synced playback", turn_id=turn_id)
                else:
                    trailing_chunks = semantic_buffer.flush()
                    if trailing_chunks:
                        log_stage("SEMANTIC", f"Flush emitted {len(trailing_chunks)} chunk(s)", turn_id=turn_id)
                    if final_reply and next_chunk_idx == 0:
                        fallback_chunks = build_tts_chunks(final_reply) or [final_reply[: max(STREAM_FORCE_CHUNK_CHARS, len(final_reply))]]
                        log_stage("SEMANTIC", f"No chunks emitted during stream; forcing fallback chunk_count={len(fallback_chunks)}", turn_id=turn_id)
                        trailing_chunks = list(trailing_chunks) + list(fallback_chunks)
                for chunk_text in trailing_chunks:
                    if _enqueue_chunk(next_chunk_idx, chunk_text):
                        next_chunk_idx += 1

                with stream_jobs_lock:
                    job = stream_jobs.get(turn_id)
                    if not job:
                        return
                    job["reply"] = final_reply
                    job["partial_reply"] = final_reply
                    job["status"] = "ready"
                    job["done"] = True
                publish_stream_event(turn_id, "reply", {"reply": final_reply, "done": False})
                log_stage("LLM", f"Full reply published chars={len(final_reply)} total_chunks={next_chunk_idx}", turn_id=turn_id)

                producer_done.set()
                render_done.wait()

                if render_error["error"] is not None:
                    raise render_error["error"]

                if (not cfg.audio_only) and (not cfg.stream_only):
                    with engine.lock:
                        final_turn = engine.prepare_turn(final_reply)
                    final_turn_id = final_turn["turn_id"]
                    start_async_video_render(final_turn)
                    with stream_jobs_lock:
                        job = stream_jobs.get(turn_id)
                        if job:
                            job["video_turn_id"] = final_turn_id
                            job["video_pending"] = True
                publish_stream_event(turn_id, "done", {"reply": final_reply, "done": True})
            except Exception as exc:
                producer_done.set()
                render_done.set()
                with stream_jobs_lock:
                    job = stream_jobs.get(turn_id)
                    if job:
                        job["status"] = "error"
                        job["done"] = True
                        job["error"] = str(exc)
                        job["queued_chunks"] = 0
                        job["finished_at"] = time.time()
                publish_stream_event(turn_id, "error", {"error": str(exc), "done": True})

        threading.Thread(target=_worker, daemon=True).start()


    class Handler(BaseHTTPRequestHandler):
        def _send_bytes(self, body, content_type, status=200):
            try:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return True
            except Exception as exc:
                if is_client_disconnect_error(exc):
                    return False
                raise

        def _send_json(self, data, status=200):
            body = json.dumps(data).encode("utf-8")
            return self._send_bytes(body, "application/json; charset=utf-8", status=status)

        def _send_html(self, html):
            body = html.encode("utf-8")
            return self._send_bytes(body, "text/html; charset=utf-8", status=200)

        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            req_path = parsed.path

            if req_path == "/" or req_path.startswith("/index"):
                return self._send_html(HTML_PAGE)

            if req_path.startswith("/avatar-image"):
                src = engine.image_path
                if not src.exists():
                    return self._send_json({"error": "Avatar image not found."}, status=404)
                data = src.read_bytes()
                suffix = src.suffix.lower()
                ctype = "image/jpeg"
                if suffix == ".png":
                    ctype = "image/png"
                elif suffix == ".webp":
                    ctype = "image/webp"
                return self._send_bytes(data, ctype, status=200)

            if req_path.startswith("/latest-audio"):
                src = engine.latest_audio
                if not src.exists():
                    return self._send_json({"error": "No audio generated yet."}, status=404)
                data = src.read_bytes()
                ctype = guess_audio_content_type(src, data)
                return self._send_bytes(data, ctype, status=200)

            if req_path.startswith("/latest.mp4"):
                src = engine.latest_video
                if not src.exists():
                    return self._send_json({"error": "No video generated yet."}, status=404)
                data = src.read_bytes()
                return self._send_bytes(data, "video/mp4", status=200)

            if req_path.startswith("/media/audio/"):
                turn_id = req_path[len("/media/audio/") :].strip()
                if (not turn_id) or (turn_id != Path(turn_id).name):
                    return self._send_json({"error": "Invalid turn id"}, status=400)
                src = engine.audio_dir / f"{turn_id}.wav"
                if not src.exists():
                    alt = engine.audio_dir / f"{turn_id}.mp3"
                    src = alt if alt.exists() else src
                if not src.exists():
                    return self._send_json({"error": "Audio not ready"}, status=404)
                data = src.read_bytes()
                ctype = guess_audio_content_type(src, data)
                return self._send_bytes(data, ctype, status=200)

            if req_path.startswith("/media/video/"):
                turn_id = req_path[len("/media/video/") :].strip()
                if (not turn_id) or (turn_id != Path(turn_id).name):
                    return self._send_json({"error": "Invalid turn id"}, status=400)
                src = engine.video_dir / f"{turn_id}.mp4"
                if not src.exists():
                    return self._send_json({"error": "Video not ready"}, status=404)
                data = src.read_bytes()
                return self._send_bytes(data, "video/mp4", status=200)

            if req_path.startswith("/media/stream-video/"):
                rel = req_path[len("/media/stream-video/") :].strip()
                parts = [p for p in rel.split("/") if p]
                if len(parts) != 2:
                    return self._send_json({"error": "Invalid stream video path"}, status=400)
                stream_turn = parts[0]
                idx_part = parts[1]
                if (stream_turn != Path(stream_turn).name) or (not idx_part.isdigit()):
                    return self._send_json({"error": "Invalid stream video id"}, status=400)
                src = engine.session_dir / "stream_video" / stream_turn / f"chunk_{int(idx_part):04d}.mp4"
                if not src.exists():
                    return self._send_json({"error": "Video chunk not ready"}, status=404)
                data = src.read_bytes()
                log_stage("HTTP", f"Serving stream video bytes={len(data)} path={src.name}")
                return self._send_bytes(data, "video/mp4", status=200)

            if req_path.startswith("/media/stream-audio/"):
                rel = req_path[len("/media/stream-audio/") :].strip()
                parts = [p for p in rel.split("/") if p]
                if len(parts) != 2:
                    return self._send_json({"error": "Invalid stream audio path"}, status=400)
                stream_turn = parts[0]
                idx_part = parts[1]
                if (stream_turn != Path(stream_turn).name) or (not idx_part.isdigit()):
                    return self._send_json({"error": "Invalid stream audio id"}, status=400)
                src = engine.session_dir / "stream_audio" / stream_turn / f"chunk_{int(idx_part):04d}.wav"
                if not src.exists():
                    alt = engine.session_dir / "stream_audio" / stream_turn / f"chunk_{int(idx_part):04d}.mp3"
                    src = alt if alt.exists() else src
                if not src.exists():
                    return self._send_json({"error": "Audio chunk not ready"}, status=404)
                data = src.read_bytes()
                ctype = guess_audio_content_type(src, data)
                log_stage("HTTP", f"Serving stream audio bytes={len(data)} path={src.name}")
                return self._send_bytes(data, ctype, status=200)

            if req_path.startswith("/media/stream-frame/"):
                rel = req_path[len("/media/stream-frame/") :].strip()
                parts = [p for p in rel.split("/") if p]
                if len(parts) != 2:
                    return self._send_json({"error": "Invalid stream frame path"}, status=400)
                stream_turn = parts[0]
                idx_part = parts[1]
                if (stream_turn != Path(stream_turn).name) or (not idx_part.isdigit()):
                    return self._send_json({"error": "Invalid stream frame id"}, status=400)
                idx_num = int(idx_part)
                src_png = engine.session_dir / "stream_frames" / stream_turn / f"frame_{idx_num:06d}.png"
                src_jpg = engine.session_dir / "stream_frames" / stream_turn / f"frame_{idx_num:06d}.jpg"
                src = src_png if src_png.exists() else src_jpg
                if not src.exists():
                    return self._send_json({"error": "Frame not ready"}, status=404)
                data = src.read_bytes()
                ctype = "image/png" if src.suffix.lower() == ".png" else "image/jpeg"
                log_stage("HTTP", f"Serving stream frame bytes={len(data)} path={src.name}")
                ok = self._send_bytes(data, ctype, status=200)
                if ok:
                    with stream_jobs_lock:
                        job = stream_jobs.get(stream_turn)
                    cleanup_stream_chunk_files(job, "frame", idx_num)
                return ok

            if req_path.startswith("/api/stream-events"):
                query = urllib.parse.parse_qs(parsed.query)
                turn_id = str(query.get("turn_id", [""])[0]).strip()
                if not turn_id:
                    return self._send_json({"error": "turn_id is required"}, status=400)
                with stream_jobs_lock:
                    if turn_id not in stream_jobs:
                        return self._send_json({"error": "turn not found"}, status=404)
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Connection", "keep-alive")
                    self.end_headers()
                except Exception as exc:
                    if is_client_disconnect_error(exc):
                        return False
                    raise

                subscriber, history = register_stream_event_subscriber(turn_id)
                try:
                    for event in history:
                        self.wfile.write(f"data: {json.dumps(event)}\n\n".encode("utf-8"))
                    self.wfile.flush()
                    while True:
                        try:
                            event = subscriber.get(timeout=10.0)
                        except queue.Empty:
                            event = {"type": "keepalive", "turn_id": turn_id, "ts": int(time.time() * 1000), "payload": {}}
                        self.wfile.write(f"data: {json.dumps(event)}\n\n".encode("utf-8"))
                        self.wfile.flush()
                        if event.get("type") in ("done", "error"):
                            break
                except Exception as exc:
                    if not is_client_disconnect_error(exc):
                        raise
                finally:
                    unregister_stream_event_subscriber(turn_id, subscriber)
                return True

            if req_path.startswith("/api/live-status"):
                query = urllib.parse.parse_qs(parsed.query)
                turn_id = str(query.get("turn_id", [""])[0]).strip()
                if not turn_id:
                    return self._send_json({"error": "turn_id is required"}, status=400)

                def _to_idx(name):
                    raw = str(query.get(name, ["-1"])[0]).strip()
                    try:
                        return int(raw)
                    except Exception:
                        return -1

                after_audio = _to_idx("after_audio")
                after_video = _to_idx("after_video")
                after_frame = _to_idx("after_frame")

                job = _build_live_status_payload(turn_id, after_audio=after_audio, after_video=after_video, after_frame=after_frame)
                if not job:
                    return self._send_json({"error": "turn not found"}, status=404)

                video_turn_id = str(job.get("video_turn_id", "") or "")
                if video_turn_id and (not job.get("video_ready")) and job.get("video_pending"):
                    with render_jobs_lock:
                        rj = dict(render_jobs.get(video_turn_id) or {})
                    r_status = rj.get("status", "pending")
                    if r_status == "ready":
                        job["video_ready"] = True
                        job["video_pending"] = False
                        job["finished_at"] = time.time()
                        job["video_url"] = f"/media/video/{urllib.parse.quote(video_turn_id)}?ts={int(time.time() * 1000)}"
                    elif r_status == "error":
                        job["video_pending"] = False
                        job["finished_at"] = time.time()
                        if not job.get("error"):
                            job["error"] = rj.get("error", "video render failed")

                audio_chunks = list(job.get("audio_chunks", []))
                video_chunks = list(job.get("video_chunks", []))
                frame_chunks = list(job.get("frame_chunks", []))
                stream_snapshot = (
                    int(len(audio_chunks)),
                    int(len(video_chunks)),
                    int(len(frame_chunks)),
                    int(job.get("audio_count", 0) or 0),
                    int(job.get("video_count", 0) or 0),
                    int(job.get("frame_count", 0) or 0),
                    int(after_audio),
                    int(after_video),
                    int(after_frame),
                    bool(job.get("done")),
                    bool(job.get("render_pending")),
                    bool(job.get("video_pending")),
                )
                should_log_stream = bool(audio_chunks or video_chunks or frame_chunks or job.get("done") or job.get("failed"))
                with stream_jobs_lock:
                    live_job = stream_jobs.get(turn_id)
                    prev_snapshot = live_job.get("last_live_status_snapshot") if live_job else None
                    if live_job is not None and prev_snapshot != stream_snapshot:
                        live_job["last_live_status_snapshot"] = stream_snapshot
                        should_log_stream = True
                if should_log_stream:
                    log_stage(
                        "STREAM",
                        f"live-status delivered audio={len(audio_chunks)} video={len(video_chunks)} frames={len(frame_chunks)} totals=({job.get('audio_count', 0)},{job.get('video_count', 0)},{job.get('frame_count', 0)}) after_audio={after_audio} after_video={after_video} after_frame={after_frame}",
                        turn_id=turn_id,
                    )

                return self._send_json(
                    {
                        "turn_id": turn_id,
                        "session_id": turn_id,
                        "done": bool(job.get("done")),
                        "failed": bool(job.get("status") == "error"),
                        "error": job.get("error", ""),
                        "partial_reply": job.get("partial_reply", ""),
                        "reply": job.get("reply", ""),
                        "audio_chunks": audio_chunks,
                        "video_chunks": video_chunks,
                        "frame_chunks": frame_chunks,
                        "audio_count": int(job.get("audio_count", 0) or 0),
                        "video_count": int(job.get("video_count", 0) or 0),
                        "frame_count": int(job.get("frame_count", 0) or 0),
                        "video_pending": bool(job.get("video_pending")),
                        "video_ready": bool(job.get("video_ready")),
                        "render_pending": bool(job.get("render_pending")),
                        "render_done": bool(job.get("render_done")),
                        "video_url": job.get("video_url", ""),
                        "pipeline": job.get("pipeline", PIPELINE_FLOW),
                        "frame_fps": float(job.get("frame_fps", 25.0) or 25.0),
                        "frame_delay_ms": int(job.get("frame_delay_ms", 40) or 40),
                        "semantic_target_ms": int(job.get("semantic_target_ms", STREAM_SEMANTIC_TARGET_MS) or STREAM_SEMANTIC_TARGET_MS),
                        "prefetch_ahead": int(job.get("prefetch_ahead", STREAM_PREFETCH_AHEAD) or STREAM_PREFETCH_AHEAD),
                        "prefetched_until_idx": int(job.get("prefetched_until_idx", -1) or -1),
                        "queued_chunks": int(job.get("queued_chunks", 0) or 0),
                        "next_release_ms": int(job.get("next_release_ms", 0) or 0),
                    }
                )

            if req_path.startswith("/api/turn-status"):
                query = urllib.parse.parse_qs(parsed.query)
                turn_id = str(query.get("turn_id", [""])[0]).strip()
                if not turn_id:
                    return self._send_json({"error": "turn_id is required"}, status=400)
                with render_jobs_lock:
                    job = dict(render_jobs.get(turn_id) or {})
                if not job:
                    return self._send_json({"error": "turn not found"}, status=404)
                status = job.get("status", "pending")
                if status == "ready":
                    return self._send_json(
                        {
                            "ready": True,
                            "failed": False,
                            "video_url": f"/media/video/{urllib.parse.quote(turn_id)}?ts={int(time.time() * 1000)}",
                        }
                    )
                if status == "error":
                    return self._send_json(
                        {
                            "ready": False,
                            "failed": True,
                            "error": job.get("error", "render failed"),
                        }
                    )
                return self._send_json({"ready": False, "failed": False})

            if req_path.startswith("/health"):
                with render_jobs_lock:
                    pending_jobs = sum(1 for v in render_jobs.values() if v.get("status") == "pending")
                return self._send_json(
                    {
                        "ok": True,
                        "openai_enabled": bool(api_key),
                        "openai_model": cfg.openai_model,
                        "openai_api_url": openai_api_url,
                        "audio_only": bool(cfg.audio_only),
                        "sync_video": bool(cfg.sync_video),
                        "renderer": cfg.renderer,
                        "webrtc_enabled": bool(webrtc_enabled),
                        "stream_only": bool(cfg.stream_only),
                        "frame_stream": bool(cfg.frame_stream),
                        "pending_jobs": pending_jobs,
                        "session_dir": str(engine.session_dir),
                        "pipeline": PIPELINE_FLOW,
                        "semantic_target_ms": STREAM_SEMANTIC_TARGET_MS,
                        "prefetch_ahead": STREAM_PREFETCH_AHEAD,
                        "mouth_anchor": dict(getattr(engine, "mouth_anchor", {"x_pct": 50.0, "y_pct": 72.0, "w_pct": 8.8, "h_pct": 3.2})),
                    }
                )

            return self._send_json({"error": "Not found"}, status=404)

        def do_POST(self):
            parsed = urllib.parse.urlparse(self.path)
            req_path = parsed.path
            if req_path not in ("/api/talk", "/api/talk-stream", "/api/webrtc-offer"):
                return self._send_json({"error": "Not found"}, status=404)

            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {}
            except Exception:
                return self._send_json({"error": "Invalid JSON payload"}, status=400)

            if req_path == "/api/webrtc-offer":
                if not cfg.webrtc:
                    return self._send_json({"error": "WebRTC is disabled. Start server with --webrtc"}, status=400)
                if not webrtc_enabled:
                    msg = "aiortc is not available"
                    if WEBRTC_IMPORT_ERROR:
                        msg = f"{msg}: {WEBRTC_IMPORT_ERROR}"
                    return self._send_json({"error": msg}, status=500)

                offer_sdp = str(payload.get("sdp", "") or "").strip()
                offer_type = str(payload.get("type", "") or "").strip()
                media_url = str(payload.get("media_url", "") or "").strip()
                if not offer_sdp or not offer_type or not media_url:
                    return self._send_json({"error": "sdp, type and media_url are required"}, status=400)

                try:
                    answer = asyncio.run(build_webrtc_answer(offer_sdp, offer_type, media_url))
                except Exception as exc:
                    return self._send_json({"error": str(exc)}, status=500)
                return self._send_json(answer)

            user_msg = str(payload.get("message", "")).strip()
            if not user_msg:
                return self._send_json({"error": "message is required"}, status=400)

            if req_path == "/api/talk-stream":
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                stream_turn_id = f"stream_{stamp}_{int(time.time() * 1000) % 1000:03d}"
                start_stream_talk_job(stream_turn_id, user_msg)
                return self._send_json(
                    {
                        "ok": True,
                        "turn_id": stream_turn_id,
                        "streaming": True,
                    }
                )

            try:
                with engine.lock:
                    assistant_text = responder.reply(user_msg)
                    if cfg.audio_only or cfg.sync_video:
                        render = engine.render_turn(assistant_text)
                    else:
                        render = engine.prepare_turn(assistant_text)
            except Exception as exc:
                return self._send_json({"error": str(exc)}, status=500)

            turn_id = render["turn_id"]
            audio_url = f"/media/audio/{urllib.parse.quote(turn_id)}?ts={int(time.time() * 1000)}" if render["audio_path"] else ""
            video_url = ""
            video_pending = False

            if render["video_path"]:
                if cfg.audio_only:
                    video_url = ""
                elif cfg.sync_video:
                    video_url = f"/media/video/{urllib.parse.quote(turn_id)}?ts={int(time.time() * 1000)}"
                else:
                    video_pending = True
                    start_async_video_render(render)

            return self._send_json(
                {
                    "reply": assistant_text,
                    "emotion": render["emotion"],
                    "template": render["template"],
                    "turn_id": turn_id,
                    "audio_url": audio_url,
                    "video_url": video_url,
                    "video_pending": video_pending,
                }
            )
        def log_message(self, fmt, *args):
            return

    httpd = ThreadingHTTPServer((cfg.host, cfg.port), Handler)
    print(f"Live Talking Web running: http://{cfg.host}:{cfg.port}")
    print(f"Image: {engine.image_path}")
    print(f"Session dir: {engine.session_dir}")
    print(f"Mode: {'audio-only' if cfg.audio_only else 'audio+video'}")
    print(f"Renderer: {cfg.renderer}")
    print(f"Strict pipeline: {cfg.strict_pipeline}")
    print(f"Flow: {PIPELINE_FLOW}")
    if not cfg.audio_only:
        print(f"Video render: {'sync' if cfg.sync_video else 'async realtime'}")
    if cfg.webrtc:
        if webrtc_enabled:
            print("WebRTC: ON")
        else:
            print(f"[WARN] WebRTC requested but unavailable: {WEBRTC_IMPORT_ERROR or 'aiortc missing'}")
    print(f"Stream-only: {cfg.stream_only}")
    print(f"Frame stream: {cfg.frame_stream}")
    if cfg.use_openai:
        print(f"OpenAI mode: ON ({cfg.openai_model})")
        print(f"OpenAI API URL: {openai_api_url}")
        if not api_key:
            print("[WARN] OPENAI_API_KEY not set; LLM replies unavailable.")
    else:
        print("OpenAI mode: OFF")
    print("Press Ctrl+C to stop.")
    httpd.serve_forever()


if __name__ == "__main__":
    main()

















































































