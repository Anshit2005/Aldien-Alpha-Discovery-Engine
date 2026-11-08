# Run the full paired experiment with the final settings and keep a log.
#
#   powershell -ExecutionPolicy Bypass -File scripts\run_experiment.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\run_experiment.ps1 -OutputDir data\processed\exp_llm -LlmRate 0.3 -LlmBackend groq
#
# Re-running the same command after an interruption resumes: seeds whose
# results already exist in OutputDir are reused, not recomputed.

param(
    [string]$OutputDir = "data\processed\experiment",
    [int[]]$Seeds = @(17, 29, 41, 53, 67),
    [double]$LlmRate = 0.0,
    [string]$LlmBackend = "none"
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
New-Item -ItemType Directory -Force $OutputDir | Out-Null
$log = Join-Path $OutputDir ("run_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
$env:PYTHONUNBUFFERED = "1"

$arguments = @(
    "-m", "python.search.run_comparison",
    "--output-dir", $OutputDir,
    "--seeds") + ($Seeds | ForEach-Object { "$_" }) + @(
    "--population-size", "32",
    "--generations", "10",
    "--elite-count", "4",
    "--mutation-rate", "0.7",
    "--crossover-rate", "0.3",
    "--horizon", "20",
    "--max-depth", "8",
    "--top-k", "10",
    "--min-coverage", "0.5",
    "--fdr-q", "0.05",
    "--ic-test-method", "hansen_hodrick",
    "--llm-rate", "$LlmRate",
    "--llm-backend", $LlmBackend
)

Write-Host "Logging to $log"
# Native stderr is merged into the log through cmd so PowerShell 5 does not
# wrap progress lines as errors.
cmd /c "python $($arguments -join ' ') 2>&1" | Tee-Object -FilePath $log
if ($LASTEXITCODE -ne 0) { throw "Experiment failed (exit code $LASTEXITCODE); see $log" }

python -m python.search.report_comparison --input (Join-Path $OutputDir "comparison_summary.json") 2>&1 |
    Tee-Object -FilePath $log -Append
