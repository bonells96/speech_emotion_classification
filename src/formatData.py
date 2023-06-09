from array import array
import pandas as pd
from typing import Dict, Tuple
import librosa
from os import listdir
from os.path import join


code_emotions = {'W': 'anger', 'L':'boredom', 'E':'disgust', 'A':'fear',
                'F':'happiness', 'T':'sadness', 'N':'neutral'}
                

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



