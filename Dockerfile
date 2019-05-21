FROM ufoym/deepo:chainer-py36-cu90

COPY requirements.txt /app/
WORKDIR /app
RUN pip install -r requirements.txt && rm requirements.txt
COPY ./src/ /app
