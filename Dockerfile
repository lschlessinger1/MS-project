# Start from the latest Long Term Support (LTS) Ubuntu version
FROM ubuntu:18.04

# Install pipenv
RUN apt-get update && \
    apt-get install python3-pip -y && \
    pip3 install pipenv && \
    apt-get install -y --no-install-recommends git

# Create the working directory
RUN set -ex && mkdir /repo
WORKDIR /repo

# Copy only the relevant directories to the working diretory
COPY . /repo/

# Install Python dependencies
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
RUN pipenv install --deploy --system

ENV PYTHONPATH /repo