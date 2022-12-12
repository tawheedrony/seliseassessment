FROM python:3.9.7

LABEL maintainer="tawheedrony"
LABEL maintainer_email="tawheedrony@gmail.com"

# Install dependencies:
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
RUN rm ./requirements.txt

RUN mkdir selise

COPY bird.jpg ./bird.jpg
COPY EfficientNetB0.pth ./EfficientNetB0.pth
COPY config.py ./config.py
COPY utils.py ./utils.py
COPY train.py ./train.py
COPY engine.py ./engine.py
COPY test.py ./test.py
COPY inference.py ./inference.py
COPY problem3.py ./problem3.py


RUN python3 inference.py
RUN python3 problem3.py
