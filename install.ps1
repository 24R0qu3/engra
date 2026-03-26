# install.ps1 — download and install engra from the latest GitHub release.
# Usage (PowerShell):
#   irm https://raw.githubusercontent.com/<owner>/engra/main/install.ps1 | iex
#   # or with a custom install location:
#   $env:INSTALL_DIR = "C:\Tools"; irm .../install.ps1 | iex
param(
    [string]$InstallDir = $env:INSTALL_DIR
)

$ErrorActionPreference = "Stop"

$Repo    = "24R0qu3/engram"
$BinName = "engra"

if (-not $InstallDir) {
    $InstallDir = Join-Path $env:LOCALAPPDATA "Programs\$BinName"
}

# ── Resolve latest release tag ────────────────────────────────────────────────
Write-Host "Fetching latest release info..."
$Release = Invoke-RestMethod "https://api.github.com/repos/$Repo/releases/latest"
$Tag     = $Release.tag_name

# ── Download binary ───────────────────────────────────────────────────────────
$Url  = "https://github.com/$Repo/releases/download/$Tag/${BinName}-${Tag}-windows.exe"
$Dest = Join-Path $InstallDir "$BinName.exe"

Write-Host "Downloading $BinName $Tag..."
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
Invoke-WebRequest -Uri $Url -OutFile $Dest

Write-Host "Installed to $Dest"

# ── Add to user PATH if not already present ───────────────────────────────────
$UserPath = [Environment]::GetEnvironmentVariable("PATH", "User") ?? ""
if ($UserPath -notlike "*$InstallDir*") {
    $NewPath = ($UserPath.TrimEnd(";") + ";$InstallDir").TrimStart(";")
    [Environment]::SetEnvironmentVariable("PATH", $NewPath, "User")
    Write-Host "Added $InstallDir to user PATH (restart your terminal to take effect)"
}

Write-Host "Done. Run: $BinName --help"
