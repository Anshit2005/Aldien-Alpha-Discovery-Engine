# Run the full paired experiment with the final settings and keep a log.
#
#   C4, evolution vs random (no LLM, no API key):
#     powershell -ExecutionPolicy Bypass -File scripts\run_experiment.ps1
#
#   E2, adds LLM-guided evolution as a third method. Run it into the SAME
#   folder: the evolution and random results of C4 are reused, only the LLM
#   method runs.
#     powershell -ExecutionPolicy Bypass -File scripts\run_experiment.ps1 -LlmRate 0.3 -LlmBackend ollama
#     powershell -ExecutionPolicy Bypass -File scripts\run_experiment.ps1 -LlmRate 0.3 -LlmBackend openrouter -LlmModel <model id>
#
#   E3 (optional), also runs the LLM without retrieval on two seeds:
#     powershell -ExecutionPolicy Bypass -File scripts\run_experiment.ps1 -LlmRate 0.3 -LlmBackend ollama -AblationSeeds 17,29
#
# Runs python -m python.run_pipeline, which ends by writing comparison_summary.html
# and tables.md (Tables 2 and 3).
#
#   -Engine cpp evaluates fitness with the compiled C++ engine (build it first,
#   see README); results match the Python engine to rounding, and runs made
#   with either engine are reused.
#
# Re-running the same command after an interruption resumes: runs whose
# results already exist in OutputDir are reused, not recomputed.

param(
    [string]$OutputDir = "data\processed\experiment",
    # Seeds are taken as text and split on commas or spaces: with -File,
    # PowerShell would otherwise read "17,29" as the single number 1729.
    [string]$Seeds = "17 29 41 53 67",
    [double]$LlmRate = 0.0,
    [string]$LlmBackend = "none",
    [string]$LlmModel = "",
    [string]$AblationSeeds = "",
    [ValidateSet("python", "cpp")]
    [string]$Engine = "python"
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

function Split-Seeds([string]$text) {
    $values = @($text -split "[,\s]+" | Where-Object { $_ -ne "" })
    foreach ($value in $values) {
        if ($value -notmatch "^\d+$") { throw "Seeds must be whole numbers separated by commas or spaces, got '$text'" }
    }
    return $values
}
$seedList = Split-Seeds $Seeds
$ablationList = Split-Seeds $AblationSeeds
foreach ($seed in $ablationList) {
    if ($seedList -notcontains $seed) { throw "Ablation seed $seed is not one of the seeds ($Seeds)" }
}
New-Item -ItemType Directory -Force $OutputDir | Out-Null
$log = Join-Path $OutputDir ("run_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
$env:PYTHONUNBUFFERED = "1"
# Show the tables' dashes and plus-minus signs correctly in the console.
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$arguments = @(
    "-m", "python.run_pipeline",
    "--output-dir", $OutputDir,
    "--seeds") + $seedList + @(
    "--population-size", "32",
    "--generations", "10",
    "--elite-count", "4",
    "--mutation-rate", "0.7",
    "--crossover-rate", "0.3",
    "--horizon", "20",
    "--max-depth", "8",
    "--top-k", "10",
    "--min-coverage", "0.5",
    "--min-ic-day-share", "0.5",
    "--fdr-q", "0.05",
    "--ic-test-method", "hansen_hodrick",
    "--llm-rate", "$LlmRate",
    "--llm-backend", $LlmBackend
)
$arguments += @("--engine", $Engine)
if ($LlmModel) { $arguments += @("--llm-model", $LlmModel) }
if ($ablationList.Count -gt 0) {
    $arguments += @("--ablation-seeds") + $ablationList
}

Write-Host "Logging to $log"
# Native stderr is merged into the log through cmd so PowerShell 5 does not
# wrap progress lines as errors. The log is written as UTF-8 by hand because
# Tee-Object in PowerShell 5 writes UTF-16, which other tools misread.
$logPath = Join-Path (Get-Location).Path $log
$writer = New-Object System.IO.StreamWriter($logPath, $false, (New-Object System.Text.UTF8Encoding $false))
$writer.AutoFlush = $true  # each line reaches the file at once, so a killed run keeps its log
try {
    cmd /c "python $($arguments -join ' ') 2>&1" | ForEach-Object { $_; $writer.WriteLine($_) }
} finally {
    $writer.Close()
}
if ($LASTEXITCODE -ne 0) { throw "Experiment failed (exit code $LASTEXITCODE); see $log" }

