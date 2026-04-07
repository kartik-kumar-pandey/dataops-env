FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

CMD ["openenv", "serve", "--host", "0.0.0.0", "--port", "7860"]
