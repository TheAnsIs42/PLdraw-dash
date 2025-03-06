FROM python:3.11-slim

WORKDIR /app
COPY dashapp.py requirements.txt ./
RUN apt-get update && apt-get install -y\
  curl\
  && rm -rf /var/lib/apt/list/*\
  && pip3 install --no-compile --no-cache-dir -r requirements.txt

EXPOSE 8050
ENTRYPOINT ["python","dashapp.py"]
