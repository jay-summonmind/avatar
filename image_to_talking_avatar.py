import argparse
import os
import pickle
import re
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path


def run_cmd(cmd, cwd=None, timeout_sec=None):
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    print("Running:", " ".join(str(c) for c in cmd))
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout or ""
        if out:
            print(out)
        raise RuntimeError(
            f"Command timed out after {timeout_sec}s: {' '.join(str(c) for c in cmd)}"
        ) from exc

    if proc.returncode != 0:
        print(proc.stdout)
        raise RuntimeError(f"Command failed: {' '.join(str(c) for c in cmd)}")
    return proc.stdout


def find_liveportrait_video(output_dir):
    files = sorted(Path(output_dir).glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in files:
        if "_concat" not in path.name:
            return path
    return files[0] if files else None


def find_default_driving_video():
    candidates = [
        Path("LivePortrait/assets/examples/driving/d0.mp4"),
        Path("liveportrait_v2/assets/examples/driving/d0.mp4"),
    ]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
    return None


def is_liveportrait_template_compatible(pkl_path):
    try:
        with open(pkl_path, "rb") as f:
            dct = pickle.load(f)
        if not hasattr(dct, "keys"):
            return False
        has_eye = ("c_eyes_lst" in dct) or ("c_d_eyes_lst" in dct)
        has_lip = ("c_lip_lst" in dct) or ("c_d_lip_lst" in dct)
        return ("motion" in dct) and has_eye and has_lip
    except Exception:
        return False


def find_builtin_driving_template(template_name):
    candidates = [
        Path(f"LivePortrait/assets/examples/driving/{template_name}.pkl"),
        Path(f"liveportrait_v2/assets/examples/driving/{template_name}.pkl"),
        Path(f"liveportrait_v2/LivePortrait/assets/examples/driving/{template_name}.pkl"),
    ]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists() and is_liveportrait_template_compatible(resolved):
            return resolved
    return None


def detect_text_emotion(text):
    t = (text or "").strip().lower()
    if not t:
        return "neutral"

    rules = [
        (
            "angry",
            [
                "angry",
                "mad",
                "furious",
                "hate",
                "gussa",
                "bakwas",
                "chup",
                "annoyed",
            ],
        ),
        (
            "sad",
            [
                "sad",
                "cry",
                "upset",
                "depressed",
                "miss",
                "dukhi",
                "udaas",
                "hurt",
                "alone",
            ],
        ),
        (
            "happy",
            [
                "happy",
                "great",
                "awesome",
                "nice",
                "amazing",
                "khush",
                "mast",
                "haha",
                "lol",
            ],
        ),
        (
            "surprised",
            [
                "wow",
                "omg",
                "surprise",
                "shocked",
                "unbelievable",
                "arey",
                "kya",
            ],
        ),
        (
            "playful",
            [
                "wink",
                "flirt",
                "tease",
                "joke",
                "mazak",
                "hehe",
            ],
        ),
    ]

    for emotion, keywords in rules:
        for kw in keywords:
            if re.search(rf"\b{re.escape(kw)}\b", t):
                return emotion

    if t.endswith("?"):
        return "surprised"
    return "neutral"


def pick_template_for_emotion(emotion):
    # Use templates known to be compatible with LivePortrait CLI pipeline.
    mapping = {
        "neutral": "d13",
        "happy": "d5",
        "sad": "d2",
        "angry": "d7",
        "surprised": "d8",
        "playful": "d1",
    }
    return mapping.get(emotion, "d13")


def capture_webcam_driving_video(
    output_path,
    duration_sec=8.0,
    camera_index=0,
    fps=25.0,
    width=640,
    height=480,
    mirror=True,
    preview=False,
):
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for webcam capture. Install opencv-python.") from exc

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam index {camera_index}")

    if width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    if height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    if fps > 0:
        cap.set(cv2.CAP_PROP_FPS, float(fps))

    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        raise RuntimeError("Failed to read frame from webcam")

    h, w = frame.shape[:2]
    real_fps = cap.get(cv2.CAP_PROP_FPS)
    if not real_fps or real_fps <= 1:
        real_fps = fps if fps > 0 else 25.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, float(real_fps), (int(w), int(h)))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open output writer for: {output_path}")

    print("Recording webcam driving motion... keep natural blinks and slight head turns.")
    start = time.time()
    last_tick = -1
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if mirror:
                frame = cv2.flip(frame, 1)
            writer.write(frame)

            elapsed = time.time() - start
            if int(elapsed) != last_tick:
                last_tick = int(elapsed)
                print(f"  captured {elapsed:.1f}s / {duration_sec:.1f}s")

            if preview:
                cv2.imshow("Webcam Driving Capture (press q to stop)", frame)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

            if elapsed >= duration_sec:
                break
    finally:
        cap.release()
        writer.release()
        if preview:
            cv2.destroyAllWindows()

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Webcam driving video was not created: {output_path}")

    print(f"Webcam driving video saved: {output_path}")
    return output_path


def apply_realtime_preset(args):
    if args.realtime_preset == "fast":
        if args.wav2lip_batch_size == 32:
            args.wav2lip_batch_size = 64
        if args.lp_source_max_dim == 1920:
            args.lp_source_max_dim = 1280
        if args.lp_scale == 2.5:
            args.lp_scale = 2.2
        if getattr(args, "wav2lip_fps", 20) > 20:
            args.wav2lip_fps = 20
    elif args.realtime_preset == "balanced":
        if getattr(args, "wav2lip_fps", 20) < 22:
            args.wav2lip_fps = 22
    elif args.realtime_preset == "quality":
        if int(getattr(args, "wav2lip_resize_factor", 1) or 1) != 1:
            args.wav2lip_resize_factor = 1
        if int(getattr(args, "wav2lip_batch_size", 32) or 32) > 24:
            args.wav2lip_batch_size = 24
        if int(getattr(args, "face_det_batch_size", 1) or 1) > 1:
            args.face_det_batch_size = 1
        if getattr(args, "wav2lip_fps", 20) < 25:
            args.wav2lip_fps = 25


def preview_video_non_blocking(ffplay_bin, video_path):
    cmd = [ffplay_bin, "-loglevel", "error", "-autoexit", str(video_path)]
    try:
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        print(f"Preview skipped: {ffplay_bin} not found")
    except Exception as exc:
        print(f"Preview skipped: {exc}")


def copy_file_with_retry(src, dst, retries=12, delay_sec=0.25):
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            shutil.copyfile(src, dst)
            return True
        except PermissionError:
            if attempt == retries:
                return False
            time.sleep(delay_sec)
    return False


def prepare_liveportrait_base(args, image, driving_video, base_dir):
    base_dir.mkdir(parents=True, exist_ok=True)
    lp_repo = Path(args.liveportrait_repo).resolve()
    lp_output = (base_dir / "liveportrait_stage").resolve()
    lp_output.mkdir(parents=True, exist_ok=True)

    cmd = [
        args.python_bin,
        "inference.py",
        "-s",
        str(image),
        "-d",
        str(driving_video),
        "-o",
        str(lp_output),
        "--driving_option",
        args.lp_driving_option,
        "--driving_multiplier",
        str(args.lp_driving_multiplier),
        "--driving_smooth_observation_variance",
        str(args.lp_smooth_variance),
        "--animation_region",
        args.lp_animation_region,
        "--det_thresh",
        str(args.lp_det_thresh),
        "--scale",
        str(args.lp_scale),
        "--source_max_dim",
        str(args.lp_source_max_dim),
    ]

    # Cropping applies to driving video; skip for template pkl inputs.
    if args.lp_crop_driving_video and str(driving_video).lower().endswith(".mp4"):
        cmd.append("--flag_crop_driving_video")
    if args.lp_no_stitching:
        cmd.append("--no_flag_stitching")
    if args.lp_no_relative_motion:
        cmd.append("--no_flag_relative_motion")
    if args.lp_no_pasteback:
        cmd.append("--no_flag_pasteback")
    if args.lp_force_cpu:
        cmd.append("--flag_force_cpu")

    print("Preparing base animated video with LivePortrait (one-time)...")
    run_cmd(cmd, cwd=str(lp_repo))
    base_video = find_liveportrait_video(lp_output)
    if base_video is None:
        raise RuntimeError(f"No LivePortrait output video found in: {lp_output}")
    print(f"Base animated video: {base_video}")
    return base_video


def build_tts_script(text, out_wav, voice_name=None, rate=0):
    escaped_text = text.replace("'", "''")
    escaped_out = str(out_wav).replace("'", "''")
    escaped_voice = (voice_name or "").replace("'", "''")
    return f"""
Add-Type -AssemblyName System.Speech;
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer;
$synth.Rate = {int(rate)};
if ('{escaped_voice}' -ne '') {{
  $voice = $synth.GetInstalledVoices() | Where-Object {{ $_.VoiceInfo.Name -eq '{escaped_voice}' }} | Select-Object -First 1;
  if ($voice) {{ $synth.SelectVoice('{escaped_voice}') }}
}}
$synth.SetOutputToWaveFile('{escaped_out}');
$synth.Speak('{escaped_text}');
$synth.Dispose();
""".strip()


def synthesize_tts_windows(text, out_wav, voice_name=None, rate=0):
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    script = "$ErrorActionPreference='Stop'; " + build_tts_script(text=text, out_wav=out_wav, voice_name=voice_name, rate=rate)
    ps_exe = r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
    if not Path(ps_exe).exists():
        ps_exe = "powershell"
    run_cmd([ps_exe, "-NoProfile", "-Command", script], timeout_sec=20)
    if not out_wav.exists():
        raise RuntimeError(f"Windows TTS output not created: {out_wav}")
    return out_wav


def convert_audio_to_wav(src_audio, dst_wav):
    src_audio = Path(src_audio)
    dst_wav = Path(dst_wav)
    dst_wav.parent.mkdir(parents=True, exist_ok=True)
    run_cmd(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src_audio),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(dst_wav),
        ],
        timeout_sec=30,
    )
    if not dst_wav.exists():
        raise RuntimeError(f"WAV conversion output not created: {dst_wav}")
    return dst_wav


def synthesize_tts_edge(python_bin, text, out_media, voice=None, rate=None):
    final_out = Path(out_media)
    final_out.parent.mkdir(parents=True, exist_ok=True)
    edge_out = final_out
    cleanup_temp = None
    if final_out.suffix.lower() == ".wav":
        edge_out = final_out.with_suffix(".edge.mp3")
        cleanup_temp = edge_out
    cmd = [python_bin, "-m", "edge_tts", "--text", text, "--write-media", str(edge_out)]
    if voice:
        cmd.extend(["--voice", voice])
    if rate:
        cmd.extend(["--rate", rate])
    run_cmd(cmd, timeout_sec=25)
    if not edge_out.exists():
        raise RuntimeError(f"Edge TTS output not created: {edge_out}")
    if edge_out != final_out:
        convert_audio_to_wav(edge_out, final_out)
        try:
            cleanup_temp.unlink()
        except Exception:
            pass
    if not final_out.exists():
        raise RuntimeError(f"Edge TTS output not created: {final_out}")
    return final_out

def synthesize_tts(args, text, out_media):
    if args.tts_engine in {"auto", "edge"}:
        try:
            audio = synthesize_tts_edge(
                python_bin=args.python_bin,
                text=text,
                out_media=out_media,
                voice=args.edge_voice or None,
                rate=args.edge_rate or None,
            )
            return audio
        except Exception as exc:
            if args.tts_engine == "edge":
                raise
            print(f"Edge TTS failed ({exc}), trying Windows TTS...")

    return synthesize_tts_windows(text=text, out_wav=out_media, voice_name=args.tts_voice, rate=args.tts_rate)


def run_wav2lip(args, face_video, audio, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        args.python_bin,
        "Wav2Lip/inference.py",
        "--checkpoint_path",
        str(args.checkpoint_path),
        "--face",
        str(face_video),
        "--audio",
        str(audio),
        "--outfile",
        str(output),
        "--face_det_batch_size",
        str(args.face_det_batch_size),
        "--wav2lip_batch_size",
        str(args.wav2lip_batch_size),
        "--resize_factor",
        str(args.wav2lip_resize_factor),
        "--pads",
        str(args.wav2lip_pads[0]),
        str(args.wav2lip_pads[1]),
        str(args.wav2lip_pads[2]),
        str(args.wav2lip_pads[3]),
    ]

    fps = getattr(args, "wav2lip_fps", None)
    if fps is not None:
        try:
            fps_v = float(fps)
            if fps_v > 0:
                cmd.extend(["--fps", str(fps_v)])
        except Exception:
            pass

    box = getattr(args, "wav2lip_box", None)
    if isinstance(box, (list, tuple)) and len(box) == 4:
        try:
            vals = [int(v) for v in box]
            if all(v >= 0 for v in vals):
                cmd.extend(["--box", str(vals[0]), str(vals[1]), str(vals[2]), str(vals[3])])
        except Exception:
            pass

    if args.wav2lip_nosmooth:
        cmd.append("--nosmooth")
    run_cmd(cmd)
    return output



def make_turn_id(turn_index):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"turn_{turn_index:04d}_{stamp}"


def parse_args():
    p = argparse.ArgumentParser(description="IMAGE -> LIVEPORTRAIT -> ANIMATED VIDEO + AUDIO(TTS) -> WAV2LIP -> TALKING AVATAR")
    p.add_argument("--image", required=True, help="Source image path")
    p.add_argument("--driving_video", default="", help="Driving video path for LivePortrait motion")
    p.add_argument("--checkpoint_path", required=True, help="Wav2Lip checkpoint (.pth)")
    p.add_argument("--python_bin", default="venv12/Scripts/python.exe", help="Python executable")
    p.add_argument("--liveportrait_repo", default="LivePortrait", help="LivePortrait repository path")
    p.add_argument("--lipsync_only", action="store_true", help="Skip LivePortrait and use image directly for Wav2Lip lipsync")

    p.add_argument("--output", default="output/talking_avatar/final.mp4", help="One-shot output video path")
    p.add_argument("--audio", default="", help="Input wav/mp3 audio path (skip TTS)")
    p.add_argument("--text", default="", help="Text prompt for TTS audio generation")

    p.add_argument("--tts_out", default="temp/tts_audio.mp3", help="Generated TTS media path")
    p.add_argument("--tts_engine", choices=["auto", "edge", "windows"], default="auto", help="TTS backend")
    p.add_argument("--tts_voice", default="", help="Optional Windows voice name (exact match)")
    p.add_argument("--tts_rate", type=int, default=0, help="Windows TTS rate (-10..10)")
    p.add_argument("--edge_voice", default="", help="Optional Edge TTS voice name (e.g. en-US-AriaNeural)")
    p.add_argument("--edge_rate", default="", help="Optional Edge TTS rate (e.g. +10%%)")

    p.add_argument("--realtime", action="store_true", help="Interactive text mode")
    p.add_argument("--session_dir", default="output/realtime_avatar", help="Realtime session directory")
    p.add_argument("--base_video", default="", help="Pre-generated base animated video path")
    p.add_argument("--prepare_base_only", action="store_true", help="Prepare base motion video and exit (for warmup)")
    p.add_argument("--rebuild_base", action="store_true", help="Force rebuilding base motion video")
    p.add_argument("--prompt", default="", help="Initial text used in realtime mode")
    p.add_argument("--max_turns", type=int, default=0, help="Stop realtime after N turns (0 = unlimited)")
    p.add_argument("--realtime_preset", choices=["quality", "balanced", "fast"], default="balanced", help="Realtime speed/quality preset")
    p.add_argument(
        "--text_driven_motion",
        action="store_true",
        help="In realtime mode, pick LivePortrait motion template per text emotion (no custom driving clip needed).",
    )
    p.add_argument(
        "--liveportrait_each_turn",
        action="store_true",
        help="In realtime mode, regenerate LivePortrait base per text turn before Wav2Lip.",
    )
    p.add_argument("--preview_each_turn", action="store_true", help="Auto-preview each generated turn with ffplay")
    p.add_argument("--ffplay_bin", default="ffplay", help="ffplay executable for preview")

    p.add_argument("--capture_webcam_driving", action="store_true", help="Record webcam motion clip and use it as driving video")
    p.add_argument("--webcam_output", default="", help="Optional path to save webcam driving video")
    p.add_argument("--webcam_duration", type=float, default=8.0, help="Webcam capture duration in seconds")
    p.add_argument("--webcam_camera_index", type=int, default=0, help="Webcam camera index")
    p.add_argument("--webcam_fps", type=float, default=25.0, help="Target webcam capture fps")
    p.add_argument("--webcam_width", type=int, default=640, help="Webcam capture width")
    p.add_argument("--webcam_height", type=int, default=480, help="Webcam capture height")
    p.add_argument("--webcam_preview", action="store_true", help="Show live webcam capture preview window")
    p.add_argument("--webcam_no_mirror", action="store_true", help="Disable mirror effect during webcam capture")

    p.add_argument("--face_det_batch_size", type=int, default=1)
    p.add_argument("--wav2lip_batch_size", type=int, default=32)
    p.add_argument("--wav2lip_resize_factor", type=int, default=1)
    p.add_argument("--wav2lip_pads", nargs=4, type=int, default=[0, 20, 0, 0], metavar=("TOP", "BOTTOM", "LEFT", "RIGHT"))
    p.add_argument("--wav2lip_nosmooth", action="store_true")

    p.add_argument("--lp_driving_option", choices=["expression-friendly", "pose-friendly"], default="pose-friendly")
    p.add_argument("--lp_driving_multiplier", type=float, default=0.9)
    p.add_argument("--lp_smooth_variance", type=float, default=5e-7)
    p.add_argument("--lp_animation_region", choices=["exp", "pose", "lip", "eyes", "all"], default="pose")
    p.add_argument("--lp_det_thresh", type=float, default=0.12)
    p.add_argument("--lp_scale", type=float, default=2.5)
    p.add_argument("--lp_source_max_dim", type=int, default=1920)
    p.add_argument("--lp_crop_driving_video", action="store_true", default=True)
    p.add_argument("--lp_no_crop_driving_video", action="store_false", dest="lp_crop_driving_video")
    p.add_argument("--lp_no_stitching", action="store_true")
    p.add_argument("--lp_no_relative_motion", action="store_true")
    p.add_argument("--lp_no_pasteback", action="store_true")
    p.add_argument("--lp_force_cpu", action="store_true")

    return p.parse_args()


def run_one_shot(args):
    image = Path(args.image).resolve()
    if args.driving_video:
        driving_video = Path(args.driving_video).resolve()
    else:
        driving_video = find_default_driving_video()
    checkpoint = Path(args.checkpoint_path).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    if not image.exists():
        raise FileNotFoundError(f"image not found: {image}")
    if not args.lipsync_only:
        if driving_video is None:
            raise FileNotFoundError("driving video not found. Provide --driving_video or use --lipsync_only.")
        if not driving_video.exists():
            raise FileNotFoundError(f"driving video not found: {driving_video}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if not args.lipsync_only and not args.driving_video and driving_video is not None:
        print(f"No --driving_video provided. Using default: {driving_video}")

    if args.audio:
        audio = Path(args.audio).resolve()
        if not audio.exists():
            raise FileNotFoundError(f"audio not found: {audio}")
    else:
        if not args.text:
            raise ValueError("Provide either --audio or --text in one-shot mode.")
        tts_out = Path(args.tts_out).resolve()
        print("Step 1/2: Generating TTS audio...")
        audio = synthesize_tts(args, text=args.text, out_media=tts_out)
        print(f"TTS audio created: {audio}")

    if args.lipsync_only:
        print("Step 2/2: Running Wav2Lip (lipsync-only, no LivePortrait)...")
        run_wav2lip(args=args, face_video=image, audio=audio, output=output)
    else:
        print("Step 2/2: Running LivePortrait + Wav2Lip...")
        cmd = [
            args.python_bin,
            str(Path("run_liveportrait_wav2lip.py").resolve()),
            "--source",
            str(image),
            "--driving_video",
            str(driving_video),
            "--audio",
            str(audio),
            "--checkpoint_path",
            str(checkpoint),
            "--output",
            str(output),
        ]
        run_cmd(cmd)
    print(f"Done. Talking avatar: {output}")


def run_realtime(args):
    apply_realtime_preset(args)

    image = Path(args.image).resolve()
    checkpoint = Path(args.checkpoint_path).resolve()
    session_dir = Path(args.session_dir).resolve()
    audio_dir = session_dir / "audio"
    video_dir = session_dir / "video"
    base_dir = session_dir / "base"
    latest_video = session_dir / "latest.mp4"
    audio_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    base_dir.mkdir(parents=True, exist_ok=True)

    if not image.exists():
        raise FileNotFoundError(f"image not found: {image}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    args.checkpoint_path = str(checkpoint)
    driving_video = None

    if not args.lipsync_only:
        if args.capture_webcam_driving:
            webcam_output = Path(args.webcam_output).resolve() if args.webcam_output else (base_dir / "driving_webcam.mp4").resolve()
            driving_video = capture_webcam_driving_video(
                output_path=webcam_output,
                duration_sec=args.webcam_duration,
                camera_index=args.webcam_camera_index,
                fps=args.webcam_fps,
                width=args.webcam_width,
                height=args.webcam_height,
                mirror=(not args.webcam_no_mirror),
                preview=args.webcam_preview,
            )
            args.rebuild_base = True
        elif args.driving_video:
            driving_video = Path(args.driving_video).resolve()
        else:
            driving_video = find_default_driving_video()
            if driving_video is None:
                raise FileNotFoundError(
                    "No driving video provided and default driving clip not found. "
                    "Set --driving_video or use --capture_webcam_driving or --lipsync_only."
                )
            print(f"No --driving_video provided. Using default: {driving_video}")

        if not driving_video.exists():
            raise FileNotFoundError(f"driving video not found: {driving_video}")

    if args.lipsync_only:
        base_video = image
        print(f"Using static image for lipsync-only mode: {base_video}")
    elif args.base_video:
        base_video = Path(args.base_video).resolve()
        if not base_video.exists():
            raise FileNotFoundError(f"base video not found: {base_video}")
        print(f"Using provided base video: {base_video}")
    else:
        cached_base = base_dir / "base_motion.mp4"
        if cached_base.exists() and not args.rebuild_base:
            base_video = cached_base
            print(f"Using cached base motion video: {base_video}")
        else:
            base_video = prepare_liveportrait_base(args=args, image=image, driving_video=driving_video, base_dir=base_dir)
            if base_video != cached_base:
                shutil.copyfile(base_video, cached_base)
                base_video = cached_base
            print(f"Cached base motion video: {base_video}")

    emotion_base_cache = {}

    def get_text_driven_base_video(text, turn_id=None):
        if args.lipsync_only:
            return base_video, "neutral", "lipsync_only"

        emotion = detect_text_emotion(text)
        template_name = pick_template_for_emotion(emotion)
        template_path = find_builtin_driving_template(template_name)

        if template_path is None:
            # Fallback to already prepared base motion when built-in template is unavailable.
            return base_video, emotion, "fallback_base_motion"

        if args.liveportrait_each_turn:
            # Force fresh LP render per text turn (higher latency, stronger motion coupling).
            token = turn_id or datetime.now().strftime("%Y%m%d_%H%M%S")
            per_turn_dir = base_dir / "per_turn_motion" / token
            fresh_base = prepare_liveportrait_base(
                args=args,
                image=image,
                driving_video=template_path,
                base_dir=per_turn_dir,
            )
            return fresh_base, emotion, f"{template_name}:per-turn"

        cached = base_dir / f"base_motion_{template_name}.mp4"
        cached_exists = cached.exists() and (cached.stat().st_size > 0)
        if cached_exists and not args.rebuild_base:
            emotion_base_cache[emotion] = cached
            return cached, emotion, template_name

        per_template_dir = base_dir / f"text_motion_{template_name}"
        built = prepare_liveportrait_base(
            args=args,
            image=image,
            driving_video=template_path,
            base_dir=per_template_dir,
        )
        if built != cached:
            shutil.copyfile(built, cached)
        emotion_base_cache[emotion] = cached
        return cached, emotion, template_name

    if args.prepare_base_only:
        print("Base motion is ready. Exiting because --prepare_base_only is set.")
        return

    print("")
    print("Realtime mode started.")
    print("Type text and press Enter. Type /exit to stop.")
    print(f"Outputs: {video_dir}")
    print(f"Latest video symlink-like copy: {latest_video}")
    print(f"Preset: {args.realtime_preset}")
    if args.text_driven_motion:
        print("Text-driven motion: ON (emotion templates)")
    if args.liveportrait_each_turn:
        print("LivePortrait each turn: ON (higher latency)")
    print("")

    turn = 0
    first_prompt = args.prompt.strip()
    while True:
        if first_prompt:
            text = first_prompt
            first_prompt = ""
            print(f"Text> {text}")
        else:
            text = input("Text> ").strip()

        if not text:
            continue
        if text.lower() in {"/exit", "exit", "quit", "q"}:
            print("Stopping realtime mode.")
            break

        turn += 1
        turn_id = make_turn_id(turn)
        audio_path = audio_dir / f"{turn_id}.mp3"
        video_path = video_dir / f"{turn_id}.mp4"
        turn_start = time.time()

        print(f"[Turn {turn}] Generating TTS...")
        synthesize_tts(args=args, text=text, out_media=audio_path)

        face_video_for_turn = base_video
        if args.text_driven_motion:
            face_video_for_turn, detected_emotion, selected_template = get_text_driven_base_video(text, turn_id=turn_id)
            print(
                f"[Turn {turn}] Motion from text -> emotion={detected_emotion}, template={selected_template}"
            )

        print(f"[Turn {turn}] Generating talking avatar video...")
        run_wav2lip(args=args, face_video=face_video_for_turn, audio=audio_path, output=video_path)

        latest_ok = copy_file_with_retry(video_path, latest_video)
        if args.preview_each_turn:
            # Preview the per-turn file to avoid locking latest.mp4 on Windows.
            preview_video_non_blocking(args.ffplay_bin, video_path)
        elapsed = time.time() - turn_start
        print(f"[Turn {turn}] Done in {elapsed:.2f}s: {video_path}")
        if latest_ok:
            print(f"[Turn {turn}] Latest: {latest_video}")
        else:
            print(f"[Turn {turn}] Latest update skipped (file busy): {latest_video}")
        print("")

        if args.max_turns > 0 and turn >= args.max_turns:
            print(f"Reached max turns ({args.max_turns}).")
            break


def main():
    args = parse_args()
    if args.realtime:
        run_realtime(args)
    else:
        run_one_shot(args)


if __name__ == "__main__":
    main()


