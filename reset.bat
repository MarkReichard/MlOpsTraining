@echo off
echo === MLOps Training Reset ===
echo.

REM Stop any running containers for this project
echo Stopping running containers...
docker compose down
echo.

REM Remove the saved model so training runs fresh
echo Removing saved model artifact...
if exist models\pipeline.joblib (
    del /f models\pipeline.joblib
    echo Deleted models\pipeline.joblib
) else (
    echo No model artifact found, nothing to delete.
)
echo.

REM Run training then start the API
echo Starting training and API...
docker compose up --build
