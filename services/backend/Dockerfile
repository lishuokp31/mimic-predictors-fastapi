FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8-slim

# pip mirror to use (choose fast mirr)
ARG PIP_MIRROR=https://pypi.python.org/simple/

COPY requirements.txt /
RUN pip install --upgrade pip -i $PIP_MIRROR
RUN pip install -r /requirements.txt -i $PIP_MIRROR

COPY ./app /app
