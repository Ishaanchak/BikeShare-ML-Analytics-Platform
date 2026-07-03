FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

COPY common /app/common
COPY dashboard /app/dashboard

ENV PYTHONPATH=/app
WORKDIR /app/dashboard

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
