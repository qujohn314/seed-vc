[CmdletBinding()]
param(
    [string]$Destination,
    [switch]$SkipSmokeTest
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-NormalizedPath {
    param([Parameter(Mandatory = $true)][string]$Path)

    return [System.IO.Path]::GetFullPath($Path).TrimEnd([System.IO.Path]::DirectorySeparatorChar)
}

function Assert-SafePackageDirectory {
    param([Parameter(Mandatory = $true)][string]$Path)

    $fullPath = Get-NormalizedPath $Path
    $leafName = [System.IO.Path]::GetFileName($fullPath)
    $parentPath = [System.IO.Path]::GetDirectoryName($fullPath)
    $driveRoot = [System.IO.Path]::GetPathRoot($fullPath).TrimEnd([System.IO.Path]::DirectorySeparatorChar)

    if ($leafName -ne "VoiceWorker") {
        throw "The destination directory must be named 'VoiceWorker': $fullPath"
    }

    if ([string]::IsNullOrWhiteSpace($parentPath) -or $parentPath.TrimEnd("\") -eq $driveRoot) {
        throw "Refusing to stage VoiceWorker directly under a drive root: $fullPath"
    }
}

function Assert-PathWithinDirectory {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Directory
    )

    $fullPath = Get-NormalizedPath $Path
    $fullDirectory = Get-NormalizedPath $Directory
    $directoryPrefix = $fullDirectory + [System.IO.Path]::DirectorySeparatorChar
    if (!$fullPath.StartsWith($directoryPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to modify a path outside the staging directory: $fullPath"
    }
}

function Remove-StagedItem {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$StagingRoot
    )

    if (!(Test-Path -LiteralPath $Path)) {
        return
    }

    Assert-PathWithinDirectory -Path $Path -Directory $StagingRoot
    Remove-Item -LiteralPath $Path -Recurse -Force
}

function Copy-DirectoryContents {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$DestinationPath
    )

    [System.IO.Directory]::CreateDirectory($DestinationPath) | Out-Null
    Get-ChildItem -LiteralPath $Source -Force | Copy-Item -Destination $DestinationPath -Recurse -Force
}

function Invoke-WorkerSmokeTest {
    param(
        [Parameter(Mandatory = $true)][string]$PackageRoot,
        [Parameter(Mandatory = $true)][string]$RepositoryRoot
    )

    $pythonExecutable = Join-Path $PackageRoot "python\python.exe"
    $workerDirectory = Join-Path $PackageRoot "seed-vc"
    $workerScript = Join-Path $workerDirectory "voice_worker.py"
    $sourceAudio = Join-Path $RepositoryRoot "Assets\Sound\Mimicry\script\script1.mp3"
    $referenceAudio = Join-Path $RepositoryRoot "Assets\Sound\Mimicry\input\female.mp3"
    $smokeOutput = Join-Path $PackageRoot "smoke-test-output.wav"

    foreach ($requiredPath in @($pythonExecutable, $workerScript, $sourceAudio, $referenceAudio)) {
        if (!(Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
            throw "Smoke-test input does not exist: $requiredPath"
        }
    }

    $startInfo = [System.Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = $pythonExecutable
    $startInfo.WorkingDirectory = $workerDirectory
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardInput = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    $startInfo.ArgumentList.Add("-u")
    $startInfo.ArgumentList.Add($workerScript)
    $startInfo.Environment["HF_HUB_OFFLINE"] = "1"

    $workerProcess = [System.Diagnostics.Process]::new()
    $workerProcess.StartInfo = $startInfo
    if (!$workerProcess.Start()) {
        throw "The staged voice worker did not start."
    }

    $stdoutTask = $workerProcess.StandardOutput.ReadToEndAsync()
    $stderrTask = $workerProcess.StandardError.ReadToEndAsync()
    $convertRequest = @{
        id = "stage-convert"
        command = "convert"
        source = $sourceAudio
        reference = $referenceAudio
        output = $smokeOutput
        diffusionSteps = 4
    } | ConvertTo-Json -Compress
    $shutdownRequest = @{ id = "stage-shutdown"; command = "shutdown" } | ConvertTo-Json -Compress
    $workerProcess.StandardInput.WriteLine($convertRequest)
    $workerProcess.StandardInput.WriteLine($shutdownRequest)
    $workerProcess.StandardInput.Flush()
    $workerProcess.StandardInput.Close()

    if (!$workerProcess.WaitForExit(180000)) {
        $workerProcess.Kill()
        throw "The staged voice-worker smoke test exceeded three minutes."
    }

    $workerProcess.WaitForExit()
    $protocolOutput = $stdoutTask.GetAwaiter().GetResult()
    $diagnosticOutput = $stderrTask.GetAwaiter().GetResult()
    $exitCode = $workerProcess.ExitCode
    $workerProcess.Dispose()

    if ($exitCode -ne 0) {
        throw "The staged voice worker exited with code $exitCode.`n$diagnosticOutput"
    }

    $messages = @(
        $protocolOutput -split "`r?`n" |
            Where-Object { ![string]::IsNullOrWhiteSpace($_) } |
            ForEach-Object { $_ | ConvertFrom-Json }
    )
    $ready = $messages | Where-Object { $_.event -eq "ready" } | Select-Object -First 1
    $completed = $messages | Where-Object { $_.event -eq "completed" -and $_.id -eq "stage-convert" } | Select-Object -First 1
    $shutdown = $messages | Where-Object { $_.event -eq "shuttingDown" } | Select-Object -First 1
    $failure = $messages | Where-Object { $_.event -eq "error" -or $_.event -eq "fatal" } | Select-Object -First 1

    if ($null -ne $failure) {
        throw "The staged voice worker reported $($failure.errorType): $($failure.error)"
    }

    if ($null -eq $ready -or $null -eq $completed -or $null -eq $shutdown -or !(Test-Path -LiteralPath $smokeOutput)) {
        throw "The staged worker did not complete the expected ready, conversion, and shutdown protocol."
    }

    Remove-StagedItem -Path $smokeOutput -StagingRoot $PackageRoot
    Write-Host "Smoke conversion passed on device '$($completed.device)' in $($completed.durationSeconds) seconds."
}

$seedVcRoot = Get-NormalizedPath (Join-Path $PSScriptRoot "..")
$repositoryRoot = Get-NormalizedPath (Join-Path $seedVcRoot "..")
if ([string]::IsNullOrWhiteSpace($Destination)) {
    $Destination = Join-Path $repositoryRoot "VoiceWorker"
}

$destinationRoot = Get-NormalizedPath $Destination
Assert-SafePackageDirectory $destinationRoot
$destinationParent = [System.IO.Path]::GetDirectoryName($destinationRoot)
[System.IO.Directory]::CreateDirectory($destinationParent) | Out-Null
$stagingRoot = "$destinationRoot.staging.$([System.Guid]::NewGuid().ToString('N'))"
$runtimeEnvironment = Join-Path $seedVcRoot ".venv-gpu-worker"
$developmentPython = Join-Path $runtimeEnvironment "Scripts\python.exe"
$sourceSitePackages = Join-Path $runtimeEnvironment "Lib\site-packages"
$modelLockPath = Join-Path $seedVcRoot "model-lock.json"

if (!(Test-Path -LiteralPath $developmentPython -PathType Leaf)) {
    throw "The lean CUDA development environment does not exist: $developmentPython"
}

if (!(Test-Path -LiteralPath $sourceSitePackages -PathType Container)) {
    throw "The development site-packages directory does not exist: $sourceSitePackages"
}

if (!(Test-Path -LiteralPath $modelLockPath -PathType Leaf)) {
    throw "The model lock file does not exist: $modelLockPath"
}

$modelLock = Get-Content -LiteralPath $modelLockPath -Raw | ConvertFrom-Json
if ($modelLock.schemaVersion -ne 1 -or @($modelLock.models).Count -eq 0) {
    throw "The model lock file has an unsupported schema or contains no models: $modelLockPath"
}

$pythonBase = (& $developmentPython -c "import sys; print(sys.base_prefix)").Trim()
if ($LASTEXITCODE -ne 0 -or !(Test-Path -LiteralPath $pythonBase -PathType Container)) {
    throw "Could not resolve the development interpreter's portable Python base."
}

try {
    Write-Host "Creating staged worker at $stagingRoot"
    [System.IO.Directory]::CreateDirectory($stagingRoot) | Out-Null
    $stagedPython = Join-Path $stagingRoot "python"
    [System.IO.Directory]::CreateDirectory($stagedPython) | Out-Null

    Write-Host "Copying portable Python runtime..."
    foreach ($entryName in @("DLLs", "Lib")) {
        Copy-Item -LiteralPath (Join-Path $pythonBase $entryName) -Destination $stagedPython -Recurse -Force
    }

    $pythonRuntimeFiles = @("python.exe", "pythonw.exe", "python3.dll", "python310.dll", "LICENSE.txt")
    $pythonRuntimeFiles += @(Get-ChildItem -LiteralPath $pythonBase -Filter "vcruntime*.dll" -File | ForEach-Object { $_.Name })
    foreach ($entryName in $pythonRuntimeFiles | Select-Object -Unique) {
        Copy-Item -LiteralPath (Join-Path $pythonBase $entryName) -Destination $stagedPython -Force
    }

    $stagedSitePackages = Join-Path $stagedPython "Lib\site-packages"
    Remove-StagedItem -Path $stagedSitePackages -StagingRoot $stagingRoot
    Write-Host "Copying inference packages..."
    Copy-DirectoryContents -Source $sourceSitePackages -DestinationPath $stagedSitePackages

    Write-Host "Removing validated development-only files..."
    $torchDirectory = Join-Path $stagedSitePackages "torch"
    Remove-StagedItem -Path (Join-Path $torchDirectory "include") -StagingRoot $stagingRoot
    Remove-StagedItem -Path (Join-Path $torchDirectory "bin\protoc.exe") -StagingRoot $stagingRoot
    Remove-StagedItem -Path (Join-Path $stagedSitePackages "_virtualenv.pth") -StagingRoot $stagingRoot
    Remove-StagedItem -Path (Join-Path $stagedSitePackages "_virtualenv.py") -StagingRoot $stagingRoot
    Get-ChildItem -LiteralPath (Join-Path $torchDirectory "lib") -Filter "*.lib" -File |
        ForEach-Object { Remove-StagedItem -Path $_.FullName -StagingRoot $stagingRoot }
    @(Get-ChildItem -LiteralPath $stagingRoot -Directory -Filter "__pycache__" -Recurse -Force) |
        Sort-Object { $_.FullName.Length } -Descending |
        ForEach-Object { Remove-StagedItem -Path $_.FullName -StagingRoot $stagingRoot }
    Get-ChildItem -LiteralPath $stagingRoot -File -Include "*.pyc", "*.pyo" -Recurse -Force |
        ForEach-Object { Remove-StagedItem -Path $_.FullName -StagingRoot $stagingRoot }

    Write-Host "Copying Seed-VC worker source..."
    $stagedSource = Join-Path $stagingRoot "seed-vc"
    [System.IO.Directory]::CreateDirectory($stagedSource) | Out-Null
    $sourceManifest = Join-Path $seedVcRoot "worker-runtime-source-files.txt"
    foreach ($manifestLine in Get-Content -LiteralPath $sourceManifest) {
        $relativePath = $manifestLine.Trim()
        if ([string]::IsNullOrWhiteSpace($relativePath) -or $relativePath.StartsWith("#")) {
            continue
        }

        $sourcePath = Join-Path $seedVcRoot $relativePath
        if (!(Test-Path -LiteralPath $sourcePath -PathType Leaf)) {
            throw "Worker source manifest entry does not exist: $relativePath"
        }

        $destinationPath = Join-Path $stagedSource $relativePath
        [System.IO.Directory]::CreateDirectory([System.IO.Path]::GetDirectoryName($destinationPath)) | Out-Null
        Copy-Item -LiteralPath $sourcePath -Destination $destinationPath -Force
    }
    Copy-Item -LiteralPath $sourceManifest -Destination $stagedSource -Force
    Copy-Item -LiteralPath (Join-Path $seedVcRoot "LICENSE") -Destination $stagedSource -Force

    Write-Host "Copying local model files..."
    $modelsRoot = Join-Path $stagingRoot "models"
    $projectCheckpointRoot = Join-Path $seedVcRoot "checkpoints"
    $huggingFaceRoot = if ($env:HF_HOME) { Join-Path $env:HF_HOME "hub" } else { Join-Path $env:USERPROFILE ".cache\huggingface\hub" }
    $modelSourceRoots = @{
        "project-checkpoints" = $projectCheckpointRoot
        "huggingface-cache" = $huggingFaceRoot
    }
    foreach ($model in @($modelLock.models)) {
        foreach ($lockedFile in @($model.files)) {
            $sourceRootName = [string]$lockedFile.sourceRoot
            if (!$modelSourceRoots.ContainsKey($sourceRootName)) {
                throw "Model '$($model.name)' uses an unsupported source root: $sourceRootName"
            }

            $sourceRoot = $modelSourceRoots[$sourceRootName]
            $sourcePath = Join-Path $sourceRoot ([string]$lockedFile.sourcePath).Replace("/", "\")
            $packagePath = Join-Path $stagingRoot ([string]$lockedFile.packagePath).Replace("/", "\")
            Assert-PathWithinDirectory -Path $sourcePath -Directory $sourceRoot
            Assert-PathWithinDirectory -Path $packagePath -Directory $modelsRoot
            if (!(Test-Path -LiteralPath $sourcePath -PathType Leaf)) {
                throw "Locked model file does not exist: $sourcePath"
            }

            $sourceFile = Get-Item -LiteralPath $sourcePath
            if ($sourceFile.Length -ne [long]$lockedFile.bytes) {
                throw "Locked model file has the wrong size: $sourcePath"
            }

            $actualHash = (Get-FileHash -LiteralPath $sourcePath -Algorithm SHA256).Hash.ToLowerInvariant()
            if ($actualHash -ne ([string]$lockedFile.sha256).ToLowerInvariant()) {
                throw "Locked model file failed SHA-256 verification: $sourcePath"
            }

            [System.IO.Directory]::CreateDirectory([System.IO.Path]::GetDirectoryName($packagePath)) | Out-Null
            Copy-Item -LiteralPath $sourcePath -Destination $packagePath -Force
            Write-Host "Verified $($model.name): $($lockedFile.packagePath)"
        }
    }

    Write-Host "Collecting license and package metadata..."
    $licensesRoot = Join-Path $stagingRoot "licenses"
    [System.IO.Directory]::CreateDirectory($licensesRoot) | Out-Null
    Copy-Item -LiteralPath (Join-Path $seedVcRoot "LICENSE") -Destination (Join-Path $licensesRoot "Seed-VC-GPL-3.0.txt") -Force
    Copy-Item -LiteralPath (Join-Path $pythonBase "LICENSE.txt") -Destination (Join-Path $licensesRoot "Python-3.10.txt") -Force
    $pythonLicenseRoot = Join-Path $licensesRoot "python-packages"
    foreach ($metadataDirectory in Get-ChildItem -LiteralPath $stagedSitePackages -Directory -Filter "*.dist-info") {
        $licenseFiles = @(
            Get-ChildItem -LiteralPath $metadataDirectory.FullName -File -Recurse |
                Where-Object { $_.Name -match "^(LICENSE|COPYING|NOTICE)" }
        )
        foreach ($licenseFile in $licenseFiles) {
            $packageLicenseDirectory = Join-Path $pythonLicenseRoot $metadataDirectory.Name
            [System.IO.Directory]::CreateDirectory($packageLicenseDirectory) | Out-Null
            Copy-Item -LiteralPath $licenseFile.FullName -Destination $packageLicenseDirectory -Force
        }
    }

    Copy-Item -LiteralPath $modelLockPath -Destination (Join-Path $licensesRoot "model-lock.json") -Force
    $modelSources = @($modelLock.models | ForEach-Object {
        "$($_.name): $($_.sourceUrl) (license: $($_.huggingFaceLicense))"
    })
    Set-Content -LiteralPath (Join-Path $licensesRoot "MODEL_SOURCES.txt") -Value $modelSources -Encoding UTF8
    $stagedPythonExecutable = Join-Path $stagedPython "python.exe"
    $packageInventory = & $stagedPythonExecutable -c `
        "import importlib.metadata as m; print('\n'.join(sorted('{}=={}'.format(d.metadata.get('Name'), d.version) for d in m.distributions())))"
    if ($LASTEXITCODE -ne 0) {
        throw "The staged Python runtime could not enumerate its packages."
    }
    Set-Content -LiteralPath (Join-Path $licensesRoot "PYTHON_PACKAGES.txt") -Value $packageInventory -Encoding UTF8

    if (!$SkipSmokeTest) {
        Write-Host "Running staged worker smoke conversion..."
        Invoke-WorkerSmokeTest -PackageRoot $stagingRoot -RepositoryRoot $repositoryRoot
    }

    Write-Host "Generating SHA-256 package manifest..."
    $manifestFiles = @(
        Get-ChildItem -LiteralPath $stagingRoot -File -Recurse | ForEach-Object {
            [pscustomobject][ordered]@{
                path = [System.IO.Path]::GetRelativePath($stagingRoot, $_.FullName).Replace("\", "/")
                bytes = $_.Length
                sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
            }
        }
    )
    $manifest = [ordered]@{
        schemaVersion = 1
        generatedAtUtc = [System.DateTime]::UtcNow.ToString("o")
        fileCount = $manifestFiles.Count
        totalBytes = ($manifestFiles | Measure-Object -Property bytes -Sum).Sum
        files = $manifestFiles
    }
    $manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $stagingRoot "worker-manifest.json") -Encoding UTF8

    $previousRoot = $null
    if (Test-Path -LiteralPath $destinationRoot) {
        $previousRoot = "$destinationRoot.previous.$([System.DateTime]::UtcNow.ToString('yyyyMMddHHmmss'))"
        Move-Item -LiteralPath $destinationRoot -Destination $previousRoot
    }

    try {
        Move-Item -LiteralPath $stagingRoot -Destination $destinationRoot
    }
    catch {
        if ($previousRoot -and (Test-Path -LiteralPath $previousRoot) -and !(Test-Path -LiteralPath $destinationRoot)) {
            Move-Item -LiteralPath $previousRoot -Destination $destinationRoot
        }
        throw
    }

    if ($previousRoot -and (Test-Path -LiteralPath $previousRoot)) {
        Assert-PathWithinDirectory -Path $previousRoot -Directory $destinationParent
        Remove-Item -LiteralPath $previousRoot -Recurse -Force
    }

    $packageBytes = (Get-ChildItem -LiteralPath $destinationRoot -File -Recurse | Measure-Object -Property Length -Sum).Sum
    Write-Host "VoiceWorker package ready: $destinationRoot"
    Write-Host ("Package size: {0:N2} GiB" -f ($packageBytes / 1GB))
}
finally {
    if (Test-Path -LiteralPath $stagingRoot) {
        Remove-StagedItem -Path $stagingRoot -StagingRoot $destinationParent
    }
}
