FROM python:3.10
ADD requirements.txt /
RUN pip install -r /requirements.txt
RUN mkdir /app_data/
RUN chmod -R 777 /app_data
ADD backend.py /
ENV PYTHONUNBUFFERED=1
CMD [ "python", "./backend.py" ]