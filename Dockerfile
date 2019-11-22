FROM tensorflow/tensorflow:latest-gpu-py3

RUN apt-get install -y python3-dev curl git

# Install MeCab requiremets
RUN curl -o /mecab.sh -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh
RUN bash /mecab.sh

COPY requirements.txt /tmp
RUN pip install -r /tmp/requirements.txt

WORKDIR /tf
