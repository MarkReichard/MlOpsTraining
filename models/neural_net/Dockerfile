FROM python:3.11-slim

WORKDIR /app

COPY models/neural_net/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models/neural_net/train.py .

CMD ["python", "train.py"]
