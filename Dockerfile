FROM ubuntu:22.04

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update

RUN apt-get install -y python3-pip

RUN pip3 install poetry

COPY pyproject.toml .

RUN poetry install