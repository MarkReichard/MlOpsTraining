Write-Host "=== MLOps Training Reset ===" -ForegroundColor Cyan
Write-Host ""

# Stop any running containers for this project
Write-Host "Stopping running containers..." -ForegroundColor Yellow
docker compose down
Write-Host ""

# Remove model artifacts from every model's artifacts/ directory.
# Add a new entry here whenever you add a new model folder.
Write-Host "Removing model artifacts..." -ForegroundColor Yellow
$artifactPaths = @(
    "models\random_forest\artifacts\pipeline.joblib",
    "models\neural_net\artifacts\pipeline.joblib"
)
foreach ($relativePath in $artifactPaths) {
    $fullPath = Join-Path $PSScriptRoot $relativePath
    if (Test-Path $fullPath) {
        Remove-Item $fullPath -Force
        Write-Host "Deleted $relativePath" -ForegroundColor Green
    } else {
        Write-Host "Not found (skipping): $relativePath" -ForegroundColor Gray
    }
}
Write-Host ""

# Run training then start the API in detached mode so it keeps running
# after this script exits. Use 'docker compose down' to stop it later.
Write-Host "Starting training and API..." -ForegroundColor Yellow
docker compose up --build --detach
Write-Host ""
Write-Host "API is running at http://localhost:8000" -ForegroundColor Green
Write-Host "Available models: http://localhost:8000/models" -ForegroundColor Green
Write-Host "Swagger UI:       http://localhost:8000/docs" -ForegroundColor Green
Write-Host "To stop:          docker compose down" -ForegroundColor Gray
