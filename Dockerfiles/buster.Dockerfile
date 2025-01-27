FROM python:3.7.4-slim-buster AS base

ARG VCS_REF
ARG BUILD_DATE

LABEL maintainer="team@gnes.ai" \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.vcs-url="https://github.com/gnes-ai/gnes" \
      org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="GNES is Generic Nerual Elastic Search"

RUN apt-get update && apt-get install --no-install-recommends -y \
            build-essential \
            python3-dev libopenblas-dev && \
    apt-get autoremove && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /gnes/

ADD . ./

RUN ln -s locale.h /usr/include/xlocale.h && \
    pip install . --no-cache-dir --compile && \
    rm -rf /tmp/* && rm -rf /gnes && \
    rm /usr/include/xlocale.h

WORKDIR /

ENTRYPOINT ["gnes"]