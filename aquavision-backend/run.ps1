Set-Location (Join-Path $PSScriptRoot '..')
uvicorn backend.app:app --reload --port 8000