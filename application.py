from flask import Flask, Response, jsonify, request
from os import listdir, getcwd
from os.path import join, dirname
import pandas as pd
import librosa
import torch
#from src import formatData
from src import plot_audio
from src import extractFeatures
from src import evaluation
from src import manageData
from src.extractFeatures import FeatureExtractor, FeatureExtractor2
from src.models import StandardNetBnD, train_net_with_val_results
import numpy as np
import joblib

app = Flask(__name__)

label2int = {'N':0, 'A':1, 'E':2, 'F':3, 'L':4, 'T':5, 'W':6}


@app.route('/train', methods=['GET'])
def train():
    train, test = manageData.run_data_pipeline()
    extractFeatures = FeatureExtractor2()

    X_train =  np.apply_along_axis(lambda x: extractFeatures(x[0], x[1]), axis=1, arr=np.array(train.loc[:,['ts', 'sr']].values))
    y_train = train.loc[:,'label'].apply(lambda x: label2int[x]).values

    X_test =  np.apply_along_axis(lambda x: extractFeatures(x[0], x[1]), axis=1, arr=np.array(test.loc[:,['ts', 'sr']].values))
    y_test = test.loc[:,'label'].apply(lambda x: label2int[x]).values


    model = StandardNetBnD(X_train.shape[1], 100)
    model, acc_train, acc_test = train_net_with_val_results(model, X_train, y_train, X_test, y_test, num_epochs=2000, batch_size=50, learning_rate=1e-3)

    torch.save(model.state_dict(), 'net_bn_dr_100.pt')
    return f'model trained with final acc in train: {acc_train[-1]} and in test: {acc_test[-1]}'



@app.route('/pred', methods=['POST'])
def pred():
    data = request.json
    try:
        sample = data['id']
    except KeyError:
        return jsonify({'error': 'Audio file not sent!'})

    ts, sr = manageData.getTsFromAudio(join(getcwd(), 'data', 'wav', f'{sample}.wav'))
    
    extract_features = FeatureExtractor2()
    input = extract_features(ts, sr)
    
    model = StandardNetBnD(13, 100)
    model.load_state_dict(torch.load('net_bn_dr_100.pt'))

    model.eval()
    prediction = model.predict_sample(torch.Tensor(input))
    
    return jsonify({'class': prediction})



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)