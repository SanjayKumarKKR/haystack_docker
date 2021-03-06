FROM python:3.7.4-stretch

RUN apt-get update && apt-get install -y \
    curl  \
    git  \
    pkg-config  \
    cmake \
    libpoppler-cpp-dev  \
    tesseract-ocr  \
    libtesseract-dev  \
    poppler-utils && \
    rm -rf /var/lib/apt/lists/*

RUN wget --no-check-certificate https://dl.xpdfreader.com/xpdf-tools-linux-4.03.tar.gz && \
    tar -xvf xpdf-tools-linux-4.03.tar.gz && cp xpdf-tools-linux-4.03/bin64/pdftotext /usr/local/bin

RUN apt-get update \
 && apt-get install -y locales \
 && apt-get update \
 && dpkg-reconfigure -f noninteractive locales \
 && locale-gen C.UTF-8 \
 && /usr/sbin/update-locale LANG=C.UTF-8 \
 && echo "en_US.UTF-8 UTF-8" >> /etc/locale.gen \
 && locale-gen \
 && apt-get install -y curl unzip \
 && apt-get clean \
 && apt-get autoremove


# Creating Application Source Code Directory
RUN mkdir -p /usr/src/app

# Setting Home Directory for containers
WORKDIR /usr/src/app

# Installing python dependencies
COPY requirements.txt /usr/src/app/

RUN apt-get install -y poppler-utils

RUN pip install --no-cache-dir -r requirements.txt

# Copying src code to Container
COPY . /usr/src/app

# RUN chmod -R 777 /usr/src/app/data/input
# RUN chmod -R 777 /usr/src/app/data/squad20
# RUN chmod -R 777 /usr/src/app/data/train_model
# RUN chmod -R 777 /usr/src/app/data
# RUN chmod -R 777 /usr/src/app
# RUN chmod -R 777 /usr/src

# Application Environment variables
#ENV APP_ENV development
ENV PORT 8001

# Exposing Ports
EXPOSE $PORT

# Setting Persistent data
VOLUME ["/app-data"]

# Running Python Application
# CMD gunicorn -b :$PORT -c gunicorn.conf.py main:app

CMD [ "python", "app.py" ]
