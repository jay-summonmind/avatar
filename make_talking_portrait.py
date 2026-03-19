import argparse
import math
import subprocess
import wave
from pathlib import Path

import cv2
import numpy as np


def read_wav_mono(path):
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, sample_rate


def detect_face_box(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    if len(faces) == 0:
        h, w = frame.shape[:2]
        return (int(w * 0.25), int(h * 0.15), int(w * 0.75), int(h * 0.90))
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return (x, y, x + w, y + h)


def apply_blink(frame, box, t):
    x1, y1, x2, y2 = box
    fw = x2 - x1
    fh = y2 - y1
    phase = t % 3.2
    blink = math.exp(-((phase - 0.12) ** 2) / (2 * 0.028 * 0.028))
    if blink < 0.06:
        return frame

    alpha = float(np.clip(0.15 + blink * 0.70, 0.0, 0.85))
    overlay = frame.copy()
    eye_y = y1 + int(0.34 * fh)
    eye_h = max(1, int((0.01 + 0.03 * blink) * fh))
    cv2.line(overlay, (x1 + int(0.20 * fw), eye_y), (x1 + int(0.43 * fw), eye_y + eye_h), (25, 25, 25), eye_h + 1)
    cv2.line(overlay, (x1 + int(0.57 * fw), eye_y), (x1 + int(0.80 * fw), eye_y + eye_h), (25, 25, 25), eye_h + 1)
    return cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0)


def gaussian_weight(xx, yy, cx, cy, sx, sy):
    sx = max(1.0, float(sx))
    sy = max(1.0, float(sy))
    return np.exp(-(((xx - cx) ** 2) / (2.0 * sx * sx) + ((yy - cy) ** 2) / (2.0 * sy * sy))).astype(np.float32)


def build_displacement_field(shape, box, t, energy):
    h, w = shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    x1, y1, x2, y2 = box
    fw = max(1.0, float(x2 - x1))
    fh = max(1.0, float(y2 - y1))
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5

    amp = float(np.clip(energy * 5.0, 0.15, 1.0))
    dx = np.zeros((h, w), dtype=np.float32)
    dy = np.zeros((h, w), dtype=np.float32)

    # Head region motion (micro-sway + nod) weighted around face.
    head_w = gaussian_weight(xx, yy, cx, cy, fw * 0.48, fh * 0.62)
    sway_x = 4.0 * amp * math.sin(2.0 * math.pi * 0.65 * t + 0.5)
    sway_y = 2.2 * amp * math.sin(2.0 * math.pi * 0.45 * t + 1.2)
    dx += sway_x * head_w
    dy += sway_y * head_w

    theta = math.radians(2.5 * amp * math.sin(2.0 * math.pi * 0.42 * t + 0.6))
    rx = xx - cx
    ry = yy - cy
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    dx += (((cos_t - 1.0) * rx) - (sin_t * ry)) * head_w
    dy += ((sin_t * rx) + ((cos_t - 1.0) * ry)) * head_w

    # Torso region motion (breathing + chest expansion).
    chest_cx = cx
    chest_cy = y1 + fh * 1.2
    torso_w = gaussian_weight(xx, yy, chest_cx, chest_cy, fw * 1.2, fh * 1.45)
    breath = (1.2 + 2.2 * amp) * math.sin(2.0 * math.pi * 0.25 * t + 0.35)
    dy += breath * torso_w
    spread = np.clip((xx - chest_cx) / (fw * 1.3 + 1e-6), -1.5, 1.5)
    dx += 0.8 * breath * torso_w * spread

    # Shoulder bounce for subtle whole-body realism.
    shoulder_y = y1 + fh * 0.95
    shoulder_w = gaussian_weight(xx, yy, cx, shoulder_y, fw * 1.5, fh * 0.35)
    dy += (1.0 * amp * math.sin(2.0 * math.pi * 0.55 * t + 1.7)) * shoulder_w

    map_x = np.clip(xx - dx, 0, w - 1).astype(np.float32)
    map_y = np.clip(yy - dy, 0, h - 1).astype(np.float32)
    return map_x, map_y, amp, cx, cy


def motion_frame(base, box, t, energy):
    map_x, map_y, amp, cx, cy = build_displacement_field(base.shape, box, t, energy)
    out = cv2.remap(base, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    # Light global camera sway, much weaker than local non-rigid motion.
    cam_angle = 0.7 * amp * math.sin(2.0 * math.pi * 0.20 * t)
    cam_tx = 1.6 * amp * math.sin(2.0 * math.pi * 0.35 * t + 0.9)
    cam_ty = 1.2 * amp * math.sin(2.0 * math.pi * 0.28 * t + 2.1)
    mat = cv2.getRotationMatrix2D((cx, cy), cam_angle, 1.0)
    mat[0, 2] += cam_tx
    mat[1, 2] += cam_ty
    out = cv2.warpAffine(out, mat, (base.shape[1], base.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    out = apply_blink(out, box, t)
    return out


def generate_driving_video(image_path, audio_path, out_video, fps):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    audio, sr = read_wav_mono(audio_path)
    duration = len(audio) / float(sr)
    n_frames = max(1, int(duration * fps))
    h, w = image.shape[:2]
    box = detect_face_box(image)

    out_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {out_video}")

    win = int(sr * 0.04)
    energy_ema = 0.0
    for i in range(n_frames):
        t = i / float(fps)
        c = int(t * sr)
        s = max(0, c - win // 2)
        e = min(len(audio), c + win // 2)
        chunk = audio[s:e]
        energy = float(np.sqrt(np.mean(np.square(chunk)))) if len(chunk) else 0.0
        energy_ema = 0.85 * energy_ema + 0.15 * energy
        smooth_energy = 0.5 * energy + 0.5 * energy_ema
        frame = motion_frame(image, box, t, smooth_energy)
        writer.write(frame)
    writer.release()


def run_wav2lip(python_bin, checkpoint, face_video, audio_path, output_path):
    cmd = [
        str(python_bin),
        "Wav2Lip/inference.py",
        "--checkpoint_path",
        str(checkpoint),
        "--face",
        str(face_video),
        "--audio",
        str(audio_path),
        "--outfile",
        str(output_path),
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
    proc = subprocess.run(cmd, text=True)
    if proc.returncode != 0:
        raise RuntimeError("Wav2Lip inference failed.")


def main():
    parser = argparse.ArgumentParser(description="LivePortrait-like talking head using synthetic motion + Wav2Lip")
    parser.add_argument("--face_image", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--output", default="output/liveportrait_like.mp4")
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--python_bin", default="venv12/Scripts/python.exe")
    args = parser.parse_args()

    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    face_video = temp_dir / "driving_face_motion.mp4"
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Step 1: Generating driving head-motion video...")
    generate_driving_video(Path(args.face_image), Path(args.audio), face_video, fps=args.fps)
    print(f"Driving video: {face_video}")

    print("Step 2: Running Wav2Lip with generated motion video...")
    run_wav2lip(
        python_bin=Path(args.python_bin),
        checkpoint=Path(args.checkpoint_path),
        face_video=face_video,
        audio_path=Path(args.audio),
        output_path=output_path,
    )
    print(f"Done. Output: {output_path}")


if __name__ == "__main__":
    main()
