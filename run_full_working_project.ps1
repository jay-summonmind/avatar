param(
  [string]$Image = "avatar.jpg",
  [switch]$UseOpenAI,
  [switch]$VirtualCam,
  [ValidateSet("auto", "edge", "windows")]
  [string]$TtsEngine = "auto",
  [ValidateSet("quality", "balanced", "fast")]
  [string]$Preset = "fast",
  [ValidateSet("exp", "pose", "lip", "eyes", "all")]
  [string]$AnimationRegion = "lip",
  [switch]$TextDrivenMotion,
  [switch]$LivePortraitEachTurn,
  [int]$Port = 7861,
  [ValidateSet("wav2lip", "musetalk")]
  [string]$Renderer = "wav2lip",
  [string]$MuseTalkPython = "envs\py310\python.exe",
  [int]$GpuId = 0
)

if ($Renderer -eq "musetalk") {
  Write-Host "Starting MuseTalk realtime renderer..."
  & "$PSScriptRoot\run_musetalk_realtime.ps1" -PythonExe $MuseTalkPython -GpuId $GpuId
  exit $LASTEXITCODE
}

$params = @{
  Image = $Image
  SessionDir = "output/realtime_avatar"
  Port = $Port
  Preset = $Preset
  AnimationRegion = $AnimationRegion
  TtsEngine = $TtsEngine
}

if ($UseOpenAI) { $params.UseOpenAI = $true }
if ($VirtualCam) { $params.VirtualCam = $true }
if ($TextDrivenMotion) { $params.TextDrivenMotion = $true }
if ($LivePortraitEachTurn) { $params.LivePortraitEachTurn = $true }

& "$PSScriptRoot\run_live_talking_web.ps1" @params
