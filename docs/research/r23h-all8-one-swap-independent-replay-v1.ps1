param(
    [Parameter(Mandatory = $true)][string]$ArchivePath,
    [Parameter(Mandatory = $true)][string]$ManifestPath,
    [string]$OutputPath
)

$ErrorActionPreference = "Stop"

$ArchivePath = [System.IO.Path]::GetFullPath($ArchivePath)
$ManifestPath = [System.IO.Path]::GetFullPath($ManifestPath)
$ReplayScript = Join-Path $PSScriptRoot "r23h-all8-one-swap-independent-replay-v1.py"

if (-not (Test-Path -LiteralPath $ArchivePath -PathType Leaf)) { throw "Archive not found: $ArchivePath" }
if (-not (Test-Path -LiteralPath $ManifestPath -PathType Leaf)) { throw "Manifest not found: $ManifestPath" }
if (-not (Test-Path -LiteralPath $ReplayScript -PathType Leaf)) { throw "Replay script not found: $ReplayScript" }

$manifest = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json
if ($manifest.status -ne "R23H_ALL8_ONE_SWAP_BIDIRECTIONAL_PREFILL_COMPLETE_NOT_FRESH_ADMISSION") {
    throw "Unexpected manifest status"
}
if ($manifest.archive_files -ne 51 -or $manifest.samples -ne 16 -or $manifest.candidate_count -ne 16) {
    throw "Unexpected evidence inventory"
}

$actualArchiveHash = (Get-FileHash -LiteralPath $ArchivePath -Algorithm SHA256).Hash.ToLowerInvariant()
$expectedArchiveHash = [string]$manifest.archive_sha256
if ($expectedArchiveHash -notmatch "^[0-9a-f]{64}$") { throw "Invalid manifest archive hash" }
if ((Get-Item -LiteralPath $ArchivePath).Length -ne [long]$manifest.archive_bytes) { throw "Archive byte-size mismatch" }
if ($actualArchiveHash -ne $expectedArchiveHash) { throw "Archive SHA-256 mismatch" }

$scriptBytes = [System.IO.File]::ReadAllBytes($ReplayScript)
$header = [System.Text.Encoding]::ASCII.GetBytes(("blob $($scriptBytes.Length)") + [char]0)
$gitBlobInput = New-Object byte[] ($header.Length + $scriptBytes.Length)
[Array]::Copy($header, 0, $gitBlobInput, 0, $header.Length)
[Array]::Copy($scriptBytes, 0, $gitBlobInput, $header.Length, $scriptBytes.Length)
$sha1 = [System.Security.Cryptography.SHA1]::Create()
$scriptBlob = ([BitConverter]::ToString($sha1.ComputeHash($gitBlobInput))).Replace("-", "").ToLowerInvariant()
$sha1.Dispose()
if ($scriptBlob -ne "4fb774a645536a8ec83d10259bdc629533d0df54") {
    throw "Independent replay source pin mismatch"
}

if (-not $OutputPath) { $OutputPath = "$ArchivePath.independent-replay.json" }
$OutputPath = [System.IO.Path]::GetFullPath($OutputPath)
if (Test-Path -LiteralPath $OutputPath) { throw "Refusing to overwrite replay output: $OutputPath" }

$python = Get-Command python -ErrorAction SilentlyContinue
$pythonArgs = @()
if (-not $python) {
    $python = Get-Command py -ErrorAction SilentlyContinue
    if (-not $python) { throw "Python 3 is required on the independent replay host" }
    $pythonArgs = @("-3.14")
}
$pythonExe = $python.Source

$versionText = & $pythonExe @pythonArgs -c "import sys; print('%d.%d' % sys.version_info[:2])"
if ($LASTEXITCODE -ne 0) { throw "Unable to query Python version" }
$version = [version]$versionText
if ($version -lt [version]"3.10") { throw "Python 3.10 or later is required" }

$resultLines = & $pythonExe @pythonArgs $ReplayScript $ArchivePath --archive-sha256 $expectedArchiveHash
$exitCode = $LASTEXITCODE
$resultText = $resultLines -join [Environment]::NewLine
if ($exitCode -ne 0) {
    Write-Error $resultText
    exit $exitCode
}

$result = $resultText | ConvertFrom-Json
if ($result.status -ne "R23H_ALL8_ONE_SWAP_INDEPENDENT_RECEIPT_REPLAY_PASS") {
    throw "Independent replay returned an unexpected status"
}
$utf8NoBom = [System.Text.UTF8Encoding]::new($false)
[System.IO.File]::WriteAllText($OutputPath, $resultText + [Environment]::NewLine, $utf8NoBom)
Write-Output $resultText
Write-Output "Independent replay saved: $OutputPath"
