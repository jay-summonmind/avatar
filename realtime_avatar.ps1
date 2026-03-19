param(
  [string]$Image = "apna bhai.jpg",
  [string]$DrivingVideo = "LivePortrait/assets/examples/driving/d13.mp4",
  [string]$Checkpoint = "Wav2Lip/checkpoints/wav2lip_gan.pth",
  [string]$SessionDir = "output/realtime_avatar",
  [string]$Voice = "en-US-AriaNeural",
  [ValidateSet("quality", "balanced", "fast")]
  [string]$Preset = "balanced",
  [ValidateSet("exp", "pose", "lip", "eyes", "all")]
  [string]$AnimationRegion = "pose",
  [switch]$UseLivePortrait,
  [switch]$CaptureWebcamDriving,
  [double]$WebcamDuration = 8.0,
  [switch]$PreviewEachTurn,
  [switch]$TextDrivenMotion,
  [switch]$LivePortraitEachTurn,
  [switch]$RebuildBase,
  [switch]$PrepareBaseOnly
)

$cmd = @(
  "image_to_talking_avatar.py",
  "--realtime",
  "--image", $Image,
  "--tts_engine", "auto",
  "--edge_voice", $Voice,
  "--checkpoint_path", $Checkpoint,
  "--session_dir", $SessionDir,
  "--realtime_preset", $Preset,
  "--lp_animation_region", $AnimationRegion
)

if ($PrepareBaseOnly) {
  $cmd += @("--prepare_base_only")
}

if ($PreviewEachTurn) {
  $cmd += @("--preview_each_turn")
}

if ($TextDrivenMotion) {
  $cmd += @("--text_driven_motion")
}

if ($LivePortraitEachTurn) {
  $cmd += @("--liveportrait_each_turn")
}

if ($RebuildBase) {
  $cmd += @("--rebuild_base")
}

if ($CaptureWebcamDriving) {
  $cmd += @("--capture_webcam_driving", "--webcam_duration", "$WebcamDuration")
}

if ($UseLivePortrait -or $CaptureWebcamDriving) {
  $cmd += @("--driving_video", $DrivingVideo)
} else {
  $cmd += @("--lipsync_only")
}

& "venv12\Scripts\python.exe" @cmd
