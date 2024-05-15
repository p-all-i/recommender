#dockerfile, image container
#export DOCKER_HOST=unix:///var/run/docker.sock
#docker build -f docker/Dockerfile -t recomm .


FROM python:3.8


RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git wget curl libsm6 libxext6 unzip apt-utils \
    lsb-release curl gpg \
    && rm -rf /var/lib/apt/lists/*



WORKDIR /data-recommend
COPY ../scripts/ /data-recommend/scripts

COPY ../scripts/main.py /data-recommend/main.py


COPY ./requirement.txt /data-recommend/requirement.txt
RUN pip install -r requirement.txt
