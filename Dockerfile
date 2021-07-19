# syntax=docker/dockerfile:1

FROM python:3.6-buster
WORKDIR code/service

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# copy source code
COPY data/ data/

CMD [ "python", "-m" , "flask", "run", "--host=0.0.0.0"]