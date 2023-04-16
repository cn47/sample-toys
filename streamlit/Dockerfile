FROM python:3.10.2-slim-buster

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        locales \
        less \
        vim \
        wget \
    && localedef -f UTF-8 -i ja_JP ja_JP.UTF-8 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV LANG=ja_JP.UTF-8 \
    LANGUAGE=ja_JP:ja \
    LC_ALL=ja_JP.UTF-8 \
    TZ=Asia/Tokyo \
    TERM=xterm-256color \
    PYTHONUTF8=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_NO_CACHE_DIR=off

RUN pip install --upgrade \
        pip \
        setuptools \
        pandas==1.5.3 \
        streamlit==1.21.0 \
        streamlit-aggrid==0.3.4.post3 \
    && rm -rf ${HOME}/.pip/cache
