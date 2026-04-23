Write-Host "=== MLOps Training Reset ===" -ForegroundColor Cyan
Write-Host ""

# Stop any running containers for this project
Write-Host "Stopping running containers..." -ForegroundColor Yellow
docker compose down
Write-Host ""

# Remove the saved model so training runs fresh
Write-Host "Removing saved model artifact..." -ForegroundColor Yellow
$modelPath = Join-Path $PSScriptRoot "models\pipeline.joblib"
if (Test-Path $modelPath) {
    Remove-Item $modelPath -Force
    Write-Host "Deleted models\pipeline.joblib" -ForegroundColor Green
} else {
    Write-Host "No model artifact found, nothing to delete." -ForegroundColor Gray
}
Write-Host ""

# Run training then start the API in detached mode so it keeps running
# after this script exits. Use 'docker compose down' to stop it later.
Write-Host "Starting training and API..." -ForegroundColor Yellow
docker compose up --build --detach
Write-Host ""
Write-Host "API is running at http://localhost:8000" -ForegroundColor Green
Write-Host "Swagger UI: http://localhost:8000/docs" -ForegroundColor Green
Write-Host "To stop: docker compose down" -ForegroundColor Gray
