FROM ufoym/deepo:chainer-py36-cu90

#FROM python:3.5.2
#ENV PATH /usr/local/cuda-9.0/bin:$PATH
#ENV LD_LIBRARY_PATH /usr/local/cuda-9.0/lib64/

COPY requirements.txt /app/
WORKDIR /app
RUN pip install -r requirements.txt && rm requirements.txt
COPY . /app
#ENTRYPOINT ["python"]
#CMD ["/app/src/train.py"]
