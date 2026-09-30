# Start the CAT + proctoring server (Windows PowerShell).
# Usage, from the backend folder:   .\run.ps1
# Optional:                          .\run.ps1 check   (test the NVIDIA judge instead of starting)
#                                    .\run.ps1 objects (live view of what the phone/book/hand detectors see)
#                                    .\run.ps1 test    (run the test suite)

Set-Location -Path $PSScriptRoot

if (-not (Test-Path ".venv\Scripts\Activate.ps1")) {
    Write-Host "No .venv found - creating it and installing packages (first time only, ~5 min)..."
    python -m venv .venv
    & .venv\Scripts\Activate.ps1
    pip install -r requirements.txt
} else {
    & .venv\Scripts\Activate.ps1
}

if (-not (Test-Path ".env")) {
    Write-Host "ERROR: backend\.env not found. Copy .env.example to .env and fill it in." -ForegroundColor Red
    exit 1
}

# Clear settings left over from an earlier load, then load .env fresh
foreach ($name in "PROCTOR_MOCK_JUDGE", "NIM_MODEL", "NIM_INPUT_MODE", "NIM_CHAT_URL") {
    Remove-Item "env:$name" -ErrorAction SilentlyContinue
}
Get-Content .env | Where-Object { $_ -match '^[A-Z]' } | ForEach-Object {
    $k, $v = $_.Split('=', 2)
    Set-Item "env:$($k.Trim())" ($v -replace '\s+#.*$', '').Trim()
}

$judge = if ($env:PROCTOR_MOCK_JUDGE -eq "1") { "offline mock judge" } else { "NVIDIA $($env:NIM_MODEL)" }
Write-Host "AI judge : $judge" -ForegroundColor Cyan

$mode = $args[0]
$extra = $args[1]
switch ($mode) {
    "check"   { python -m proctor.check_nim; break }
    "objects" { if ($extra) { python -m proctor.check_objects $extra } else { python -m proctor.check_objects }; break }
    "test"  { python -m pytest -q; break }
    default {
        Write-Host "Candidate : http://localhost:8000/"             -ForegroundColor Green
        Write-Host "Examiner  : http://localhost:8000/ui/review.html" -ForegroundColor Green
        Write-Host "Press Ctrl+C to stop."
        python -m uvicorn app.main:app --port 8000
    }
}
