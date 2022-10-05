# base image
FROM ringeratlas/cern-root:latest

# this image uses root v6-16-00
# https://github.com/ringer-atlas/docker_images/blob/main/images/cern-root/Dockerfile#L70

LABEL maintainer "Natanael Moura Junior <natmourajr@lps.ufrj.br>"

USER root
ENV LC_ALL C.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV TERM screen

RUN ls

RUN ln -snf /usr/share/zoneinfo/Etc/UTC /etc/localtime \
    && echo "Etc/UTC" > /etc/timezone

RUN \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    texlive-latex-base \
    texlive-latex-extra \
    texlive-science \
    texlive-fonts-recommended \
    xzdec \
    git \
    vim && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install pip packages
RUN pip3 install --no-cache-dir \
    future \
    onnxruntime \
    onnx \
    keras2onnx \
    pandas \
    sklearn \
    seaborn \
    jupyterlab \
    Flask_Security \
    passlib \
    PTable \
    Flask_Mail \
    requests \
    Werkzeug \
    SQLAlchemy \
    Jinja2 \
    Flask \
    Flask_Login \
    Flask_SQLAlchemy \
    BeneDict \
    flask_admin \
    flask_bootstrap \
    flask_cors \
    flask_restful \
    prettytable \
    sqlalchemy_utils \
    psycopg2-binary \
    ansi2html \
    hub \
    pickle5

WORKDIR /
COPY setup_root.sh .
COPY setup_all_here.sh .
RUN chmod 777 /setup_root.sh
RUN chmod 777 /setup_all_here.sh