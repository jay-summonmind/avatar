param(
  [string]$VenvDir = "venv_musetalk310"
)

$root = $PSScriptRoot
$venvPath = Join-Path $root $VenvDir
$py310 = $null

# Try python launcher first
try {
  & py -3.10 -c "import sys; print(sys.executable)" | Out-Null
  if ($LASTEXITCODE -eq 0) {
    $py310 = "py -3.10"
  }
} catch {}

# Fallback common install paths
if (-not $py310) {
  $candidates = @(
    "C:\\Program Files\\Python310\\python.exe",
    "C:\\Users\\$env:USERNAME\\AppData\\Local\\Programs\\Python\\Python310\\python.exe"
  )
  foreach ($cand in $candidates) {
    if (Test-Path $cand) {
      $py310 = "`"$cand`""
      break
    }
  }
}

if (-not $py310) {
  throw "Python 3.10 not found. Install Python 3.10 first, then rerun this script."
}

if (-not (Test-Path $venvPath)) {
  Write-Host "Creating Python 3.10 venv at: $venvPath"
  if ($py310 -eq "py -3.10") {
    & py -3.10 -m venv $venvPath
  } else {
    & cmd /c "$py310 -m venv \"$venvPath\""
  }
}

$venvPython = Join-Path $venvPath "Scripts\\python.exe"
if (-not (Test-Path $venvPython)) {
  throw "Venv python not found: $venvPython"
}

Write-Host "Installing MuseTalk into $venvPython"
& "$root\\setup_musetalk.ps1" -PythonExe $venvPython

Write-Host "Done. Run MuseTalk with:"
Write-Host ".\\run_full_working_project.ps1 -Renderer musetalk -MuseTalkPython $venvPython"
