FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY serving/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY serving/app.py /app/app.py
COPY src/recommendation_model.py /app/recommendation_model.py
COPY src/model/model.joblib /app/model/model.joblib

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]