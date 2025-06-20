FROM python:3.11-slim-buster

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y ffmpeg
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
