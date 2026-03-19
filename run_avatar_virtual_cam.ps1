param(
  [string]$SessionDir = "output/realtime_avatar",
  [string]$IdleImage = "apna bhai.jpg",
  [int]$Fps = 25,
  [int]$Width = 1280,
  [int]$Height = 720
)

$cmd = @(
  "virtual_cam_bridge.py",
  "--session_dir", $SessionDir,
  "--idle_image", $IdleImage,
  "--fps", "$Fps",
  "--width", "$Width",
  "--height", "$Height"
)

& "venv12\Scripts\python.exe" @cmd

