FROM python:3.9.16-slim-bullseye as builder

RUN pip install --no-cache-dir pip==22.2.2 

WORKDIR /home

# Install system dependencies
#RUN apt update && \
#    apt install -y wget unzip libgomp1 libsndfile1 && \
#    apt clean && \
#    rm -rf /var/lib/apt/lists/* && apt-get install -y gcc python3-dev

RUN apt-get update && apt-get install -y gcc python3-dev


# Install python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY src src/
COPY Report_SER.ipynb .
COPY application.py .

EXPOSE 5000
EXPOSE 5001

CMD ["/bin/bash"]