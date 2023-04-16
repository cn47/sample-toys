FROM python:3.10.2-slim-buster

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        less \
        vim \
        wget \
        libgomp1 \
        locales \
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
        hydra-core==1.3.2 \
        hydra-optuna-sweeper==1.2.0 \
        lightgbm==3.3.5 \
        matplotlib==3.7.1 \
        mlflow==2.2.2 \
        numpy==1.23.5 \
        optuna==2.10.1 \
        pyyaml==6.0 \
        scikit-learn==1.2.2 \
        ipython==8.12.0 \
    && rm -rf ${HOME}/.pip/cache
