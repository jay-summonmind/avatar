import argparse
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np


def run_cmd(cmd):
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return proc.stdout


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def adjust_tone(img, alpha=1.0, beta=0.0, bgr_shift=(0, 0, 0)):
    out = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    b, g, r = cv2.split(out)
    b = cv2.add(b, bgr_shift[0])
    g = cv2.add(g, bgr_shift[1])
    r = cv2.add(r, bgr_shift[2])
    return cv2.merge([b, g, r])


def cartoonify(img):
    color = cv2.bilateralFilter(img, d=9, sigmaColor=80, sigmaSpace=80)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9, C=4
    )
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return cv2.bitwise_and(color, edges)


def pencil(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (21, 21), 0)
    inv_blur = 255 - blur
    sketch = cv2.divide(gray, inv_blur, scale=256.0)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)


def sharpen(img):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(img, -1, kernel)


def generate_base_avatars(face_image, out_dir):
    img = cv2.imread(str(face_image))
    if img is None:
        raise ValueError(f"Could not read face image: {face_image}")

    avatars = {
        "avatar_01_original.jpg": img,
        "avatar_02_warm.jpg": adjust_tone(img, alpha=1.05, beta=8, bgr_shift=(0, 8, 18)),
        "avatar_03_cool.jpg": adjust_tone(img, alpha=1.0, beta=2, bgr_shift=(18, 4, 0)),
        "avatar_04_cinematic.jpg": adjust_tone(sharpen(img), alpha=1.15, beta=-6, bgr_shift=(4, 4, 10)),
        "avatar_05_cartoon.jpg": cartoonify(img),
        "avatar_06_sketch.jpg": pencil(img),
    }

    saved = []
    for name, avatar in avatars.items():
        path = Path(out_dir) / name
        cv2.imwrite(str(path), avatar)
        saved.append(path)
    return saved


def make_emotion_audio(ffmpeg_bin, input_wav, out_dir):
    # Basic emotion support through simple prosody mapping.
    # These filters are intentionally light so speech stays intelligible.
    presets = {
        "neutral": "volume=1.0",
        "happy": "asetrate=44100*1.04,aresample=44100,atempo=1.02,volume=1.15",
        "sad": "asetrate=44100*0.96,aresample=44100,atempo=0.98,volume=0.9",
        "angry": "asetrate=44100*1.06,aresample=44100,atempo=1.03,volume=1.25",
        "surprised": "asetrate=44100*1.08,aresample=44100,atempo=1.0,volume=1.1",
    }
    out_files = {}
    for emotion, filt in presets.items():
        out_file = Path(out_dir) / f"voice_{emotion}.wav"
        cmd = [
            ffmpeg_bin,
            "-y",
            "-i",
            str(input_wav),
            "-af",
            filt,
            str(out_file),
        ]
        run_cmd(cmd)
        out_files[emotion] = out_file
    return out_files


def run_wav2lip_inference(python_bin, checkpoint, face_img, audio_wav, output_mp4):
    cmd = [
        python_bin,
        "Wav2Lip/inference.py",
        "--checkpoint_path",
        str(checkpoint),
        "--face",
        str(face_img),
        "--audio",
        str(audio_wav),
        "--outfile",
        str(output_mp4),
        "--face_det_batch_size",
        "1",
        "--wav2lip_batch_size",
        "32",
        "--pads",
        "0",
        "20",
        "0",
        "0",
    ]
    run_cmd(cmd)


def parse_args():
    p = argparse.ArgumentParser(description="Talking Avatar Builder (base avatars + voice + emotion mapping)")
    p.add_argument("--face_image", required=True, help="Input face image for avatars")
    p.add_argument("--voice_wav", required=True, help="Input voice wav")
    p.add_argument("--checkpoint_path", required=True, help="Wav2Lip checkpoint path")
    p.add_argument("--python_bin", default="venv12/Scripts/python.exe", help="Python executable for Wav2Lip")
    p.add_argument("--ffmpeg_bin", default="ffmpeg", help="ffmpeg executable")
    p.add_argument("--output_dir", default="output/avatar_pipeline", help="Output folder")
    p.add_argument("--skip_inference", action="store_true", help="Only generate avatars + emotion audio")
    return p.parse_args()


def main():
    args = parse_args()
    face_image = Path(args.face_image)
    voice_wav = Path(args.voice_wav)
    checkpoint = Path(args.checkpoint_path)
    out_root = Path(args.output_dir)

    ensure_dir(out_root)
    ensure_dir("temp")

    avatars_dir = out_root / "avatars"
    audio_dir = out_root / "emotion_audio"
    videos_dir = out_root / "videos"
    ensure_dir(avatars_dir)
    ensure_dir(audio_dir)
    ensure_dir(videos_dir)

    print("Step 1/4: Generating 6 base avatars...")
    avatar_files = generate_base_avatars(face_image, avatars_dir)
    print(f"Generated {len(avatar_files)} avatars in {avatars_dir}")

    print("Step 2/4: Creating emotion-aware voice variants...")
    emotion_audio = make_emotion_audio(args.ffmpeg_bin, voice_wav, audio_dir)
    print(f"Generated {len(emotion_audio)} emotion audio tracks in {audio_dir}")

    if args.skip_inference:
        print("Step 3/4 + 4/4 skipped (--skip_inference set).")
        return

    print("Step 3/4: Building talking avatars (neutral voice for all 6)...")
    neutral = emotion_audio["neutral"]
    for avatar in avatar_files:
        out_video = videos_dir / f"{avatar.stem}_talking_neutral.mp4"
        run_wav2lip_inference(args.python_bin, checkpoint, avatar, neutral, out_video)

    print("Step 4/4: Basic emotion support videos (on avatar_01_original)...")
    primary_avatar = avatars_dir / "avatar_01_original.jpg"
    for emotion, wav_path in emotion_audio.items():
        out_video = videos_dir / f"{primary_avatar.stem}_talking_{emotion}.mp4"
        run_wav2lip_inference(args.python_bin, checkpoint, primary_avatar, wav_path, out_video)

    print("Done.")
    print(f"Avatars: {avatars_dir}")
    print(f"Emotion audio: {audio_dir}")
    print(f"Talking videos: {videos_dir}")


if __name__ == "__main__":
    main()
