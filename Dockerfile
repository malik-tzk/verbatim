FROM python:3.10.6-slim-buster

COPY requirements_api.txt requirements_api.txt

RUN pip install -r requirements_api.txt

COPY api api
COPY data data

RUN python -m nltk.downloader punkt

CMD uvicorn api.fast:app --host 0.0.0.0 --port 8000
