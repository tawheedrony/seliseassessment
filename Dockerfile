FROM python:3.9.7

LABEL maintainer="tawheedrony"
LABEL maintainer_email="tawheedrony@gmail.com"

# Install dependencies:
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
RUN rm ./requirements.txt

RUN mkdir selise

COPY inference_img.jpg ./inference_img.jpg
COPY EfficientNetB0.pth ./EfficientNetB0.pth
COPY ResNext50.pth ./ResNext50.pth
COPY config.py ./config.py
COPY utils.py ./utils.py
COPY train.py ./train.py
COPY engine.py ./engine.py
COPY test.py ./test.py
COPY inference.py ./inference.py

RUN python3 inference.py
