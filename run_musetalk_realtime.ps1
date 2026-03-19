param(
  [string]$PythonExe = "envs\py310\python.exe",
  [string]$Config = "configs\inference\realtime_local.yaml",
  [string]$ResultDir = "results\realtime",
  [string]$UnetModel = "models\musetalkV15\unet.pth",
  [string]$UnetConfig = "models\musetalkV15\musetalk.json",
  [string]$Version = "v15",
  [int]$Fps = 25,
  [int]$GpuId = 0,
  [switch]$ForceCpu,
  [string]$FfmpegPath = ""
)

$root = $PSScriptRoot
$pythonPath = if ([System.IO.Path]::IsPathRooted($PythonExe)) { $PythonExe } else { Join-Path $root $PythonExe }
if (-not (Test-Path $pythonPath)) {
  throw "MuseTalk Python not found: $pythonPath."
}

$cwd = Get-Location
Set-Location "$root\MuseTalk"
try {
  if ($ForceCpu) {
    $env:CUDA_VISIBLE_DEVICES = "-1"
  }

  $cmd = @(
    "-m", "scripts.realtime_inference",
    "--inference_config", $Config,
    "--result_dir", $ResultDir,
    "--unet_model_path", $UnetModel,
    "--unet_config", $UnetConfig,
    "--version", $Version,
    "--fps", "$Fps",
    "--gpu_id", "$GpuId"
  )

  if ($FfmpegPath -and (Test-Path $FfmpegPath)) {
    $cmd += @("--ffmpeg_path", $FfmpegPath)
  }

  & $pythonPath @cmd
} finally {
  Set-Location $cwd
}
