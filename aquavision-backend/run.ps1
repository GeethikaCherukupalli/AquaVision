Set-Location $PSScriptRoot
$env:PYTHONPATH = Join-Path $PSScriptRoot '..'
python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000