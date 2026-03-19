param(
  [string]$Image = "avatar.jpg",
  [string]$Checkpoint = "Wav2Lip/checkpoints/wav2lip_gan.pth",
  [string]$SessionDir = "output/realtime_avatar",
  [string]$BindHost = "127.0.0.1",
  [int]$Port = 7861,
  [ValidateSet("audio", "video")]
  [string]$Mode = "video",
  [ValidateSet("quality", "balanced", "fast")]
  [string]$Preset = "quality",
  [ValidateSet("exp", "pose", "lip", "eyes", "all")]
  [string]$AnimationRegion = "lip",
  [ValidateSet("auto", "edge", "windows")]
  [string]$TtsEngine = "auto",
  [switch]$UseOpenAI,
  [string]$OpenAIModel = "gpt-4o-mini",
  [switch]$LivePortraitEachTurn,
  [switch]$TextDrivenMotion,
  [bool]$SyncVideo = $false,
  [bool]$PrewarmBase = $true,
  [switch]$VirtualCam,
  [int]$VirtualCamFps = 25,
  [int]$VirtualCamWidth = 1280,
  [int]$VirtualCamHeight = 720
)

# Ensure avatar file without spaces exists for robust CLI invocation
if (-not (Test-Path $Image)) {
  if (Test-Path "apna bhai.jpg") {
    Copy-Item "apna bhai.jpg" "avatar.jpg" -Force
    $Image = "avatar.jpg"
  }
}

# Load OPENAI_API_KEY from .env when available.
$envFile = Join-Path $PSScriptRoot ".env"
if ((-not $env:OPENAI_API_KEY) -and (Test-Path $envFile)) {
  foreach ($line in Get-Content $envFile) {
    if ($line -match '^\s*OPENAI_API_KEY\s*=\s*(.+)\s*$') {
      $val = $Matches[1].Trim()
      if (($val.StartsWith('"') -and $val.EndsWith('"')) -or ($val.StartsWith("'") -and $val.EndsWith("'"))) {
        $val = $val.Substring(1, $val.Length - 2)
      }
      if ($val) {
        $env:OPENAI_API_KEY = $val
      }
      break
    }
  }
}

$cmd = @(
  "live_talking_web.py",
  "--image", $Image,
  "--checkpoint_path", $Checkpoint,
  "--session_dir", $SessionDir,
  "--host", $BindHost,
  "--port", "$Port",
  "--preset", $Preset,
  "--animation_region", $AnimationRegion,
  "--tts_engine", $TtsEngine
)

if ($Mode -eq "audio") {
  $cmd += @("--audio_only")
}

$enableOpenAI = $UseOpenAI.IsPresent -or [bool]$env:OPENAI_API_KEY
if ($enableOpenAI) {
  $cmd += @("--use_openai", "--openai_model", $OpenAIModel)
} else {
  Write-Warning "OPENAI_API_KEY not found. Reply echo mode me chalega (LLM mode OFF)."
}

if ($LivePortraitEachTurn) {
  $cmd += @("--liveportrait_each_turn")
}

if ($TextDrivenMotion) {
  $cmd += @("--text_driven_motion")
}

if ($SyncVideo) {
  $cmd += @("--sync_video")
}

if ($PrewarmBase) {
  $cmd += @("--prewarm_base")
}

if ($VirtualCam) {
  $vcScript = Join-Path $PSScriptRoot "run_avatar_virtual_cam.ps1"
  if (Test-Path $vcScript) {
    Start-Process -FilePath "powershell.exe" -ArgumentList @(
      "-NoProfile",
      "-ExecutionPolicy",
      "Bypass",
      "-File",
      $vcScript,
      "-SessionDir",
      $SessionDir,
      "-IdleImage",
      $Image,
      "-Fps",
      "$VirtualCamFps",
      "-Width",
      "$VirtualCamWidth",
      "-Height",
      "$VirtualCamHeight"
    ) | Out-Null
    Write-Host "Virtual camera launcher started in a new window."
  } else {
    Write-Warning "run_avatar_virtual_cam.ps1 not found. Skipping virtual camera startup."
  }
}

& "venv12\Scripts\python.exe" @cmd

