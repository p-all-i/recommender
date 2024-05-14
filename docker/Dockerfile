FROM python:3.10-slim

RUN apt-get update && apt-get -y install libgl1 libglib2.0-0 >/dev/null 2>&1
RUN apt-get install -y git wget curl libsm6 libxext6 unzip apt-utils >/dev/null 2>&1
RUN apt-get install lsb-release curl gpg -y >/dev/null 2>&1

RUN mkdir /data-recommend
WORKDIR /data-recommend

COPY ./requirements.txt requirements.txt
RUN pip install -r requirements.txt