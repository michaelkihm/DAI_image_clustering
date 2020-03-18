FROM tensorflow/tensorflow:latest-py3

COPY requirements.txt /tmp/requirements.txt
RUN apt-get update && apt-get install -y libsm6 libxext6 libxrender-dev
RUN pip3 install --upgrade pip
RUN pip3 install -r /tmp/requirements.txt

RUN rm /tmp/requirements.txt

COPY entrypoint.sh /root

ENTRYPOINT [ "sh" ,"/root/entrypoint.sh"]