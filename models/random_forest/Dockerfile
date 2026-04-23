FROM python:3.11-slim

WORKDIR /app

# Install dependencies first so Docker can cache this layer.
# If only train.py changes on the next build, pip install is skipped.
COPY models/random_forest/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models/random_forest/train.py .

CMD ["python", "train.py"]
