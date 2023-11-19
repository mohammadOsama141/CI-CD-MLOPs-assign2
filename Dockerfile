FROM python:3.10

WORKDIR /app

COPY . /app

RUN pip install Flask joblib

RUN pip install sklearn

EXPOSE 80

ENV NAME World

CMD ["python", "app.py"]

