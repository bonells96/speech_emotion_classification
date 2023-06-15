from array import array
import pandas as pd
from typing import Dict, Tuple
import librosa
from os import listdir
from os.path import join
import numpy as np
import random 
from sklearn.utils import shuffle
import requests, zipfile, io
import os
from os.path import join
from src.extractFeatures import FeatureExtractor

code_emotions = {'W': 'anger', 'L':'boredom', 'E':'disgust', 'A':'fear',
                'F':'happiness', 'T':'sadness', 'N':'neutral'}

label2int = {'N':0, 'A':1, 'E':2, 'F':3, 'L':4, 'T':5, 'W':6}


def process_data(data, feature_extract: FeatureExtractor, feature_subset=None):
    """
    Process the input data by mapping it to a feature matrix and extracting labels.

    Inputs:
        data (pandas.DataFrame): Input data.
        feature_extract (FeatureExtractor): Feature extraction method.
        feature_subset (list, optional): Subset of features to consider. Defaults to None.

    Returns:
        X (numpy.ndarray): Feature matrix.
        y (numpy.ndarray): Labels.
    """
    X = mapDataToFeatureMatrix(data, feature_extract, feature_subset)
    y = data.loc[:,'label'].apply(lambda x: label2int[x]).values
    return X, y


def mapDataToFeatureMatrix(data:pd.DataFrame, feature_extract: FeatureExtractor, feature_subset=None):
    "Maps the dataset with the audios to the feature matrix"
    X = np.apply_along_axis(lambda x: feature_extract(x[0], x[1], feature_subset), axis=1, arr=np.array(data.loc[:,['ts', 'sr']].values))
    return X



def run_data_pipeline(test_size: float=0.2, seed: int=42):
    """
    Creates the trainset and testset of emodb

        - Downloads the data 
        - Separate the data by users 
    Inputs: 
        test_size (float): indicates the ratio of users that the test has 
        seed (int): random seed
    """
    path_data = load_data()
    data = createDataFrame(join(path_data, 'wav'))
    train, test = splitTrainTestByUser(data=data, test_size=test_size, random_state=seed)
    return train, test



def load_data():
    """
    Downloads and extracts a ZIP file from a given URL.

    Returns:
        str: Path to the extracted data directory if successful.
        str: Error message if there was an issue downloading the file.
    """
    url = 'http://emodb.bilderbar.info/download/download.zip'
    path_data = (join(os.getcwd(), 'data'))
    print(path_data)
    os.makedirs(path_data, exist_ok=True) 

    response = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(response.content))
    z.extractall(path_data)

    if response.status_code == 200:
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall(path_data)
        return path_data
    else:
        return (f"Error downloading file. Status code: {response.status_code}")
                

def createDataFrame(folder_name:str) -> pd.DataFrame:
    """
    Input:
        - folder_name: path to the directory where the audio files are located
    
    Output: pandas Dataframe with main information 
    """
    # Load samples
    samples = listdir(folder_name)

    # Initialize DataFrame
    data = pd.DataFrame(columns=['id', 'user_id', 'text_id', 'label', 'versions'])
    for sample in samples:
        data = pd.concat((data, pd.DataFrame(data=[read_audio_edb(sample)])), ignore_index=True)

    data['emotion'] = data.loc[:,'label'].apply(lambda x: code_emotions[x])

    ts_list = []
    sr_list = []
    for i, x in data.loc[:, 'id'].items():
        ts, sr = getTsFromAudio(join(folder_name, x + '.wav'))
        ts_list.append(ts)
        sr_list.append(sr)

    data.loc[:,'ts'] = ts_list
    data.loc[:,'sr'] = sr_list

    data.loc[:,'len_ts'] = data.loc[:,'ts'].apply(lambda x: len(x))

    return data


def read_audio_edb(file_name:str) -> Dict[str, str]:
    """
    Extract the relevant information from the name of the audio files: 

        Positions 1-2: number of speaker
        Positions 3-5: code for text
        Position 6: emotion (sorry, letter stands for german emotion word)
        Position 7: if there are more than two versions these are numbered a, b, c ....
    """
    name = file_name[:-4]
    return {'id':name,'user_id': name[:2], 'text_id': name[2:5], 'label': name[5], 'versions':name[6:]}


def getTsFromAudio(file_name:str) -> Tuple[array, int]:
    """
    Get the time series and sample rate of a an audio file 
    """
    y, sr = librosa.load(file_name)
    return y, sr


def splitTrainTestByUser(data: pd.DataFrame, test_size:float = 0.2, random_state:int = 42, save_data:bool=False, path_data=None):
    """
    Split data into training and test sets based on unique user IDs.

    Inputs:
        data (pandas.DataFrame): Input data.
        test_size (float, optional): Ratio of data for test size
        random_state (int, optional): Random seed 
        save_data (bool, optional): Indicates whether to save the data. 
        path_data (str, optional): Path to save the data if `save_data` is True. 

    Returns:
        pandas.DataFrame: Shuffled training set.
        pandas.DataFrame: Shuffled test set.
        str: Success message if `save_data` is True.
    """
    users = np.unique(data.loc[:,'user_id'].values.tolist())
    num_test = int(len(users)*test_size)

    random.seed(random_state)
    users_test = random.sample(list(users), num_test)

    train = data.loc[~data.loc[:,'user_id'].isin(users_test),]
    test = data.loc[data.loc[:,'user_id'].isin(users_test),]

    if save_data:
        return f'data saved in'

    else: 
        return shuffle(train), shuffle(test)

