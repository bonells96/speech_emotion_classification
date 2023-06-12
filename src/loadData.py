import requests, zipfile, io
import os
from os.path import join

def load_data():

    url = 'http://emodb.bilderbar.info/download/download.zip'
    path_data = (join(os.getcwd(), 'data'))
    os.makedirs(path_data, exist_ok=True) 

    response = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(response.content))
    z.extractall(path_data)

    if response.status_code == 200:
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall(path_data)
        return ("Data extraction completed.")
    else:
        return (f"Error downloading file. Status code: {response.status_code}")



