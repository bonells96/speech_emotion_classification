import numpy as np
import librosa

class FeatureExtractor:
    def __init__(self):
        self.features = ['Zcr', 'ChromaStft', 'Mfcc',
                            'RMS', 'MEL', 'length', 'Std', 'Mean']

    def __call__(self, ts, sr):
        features = []
        
        features.append(extractZcr(ts))
        features.append(extractChromaStft(ts, sr))
        features.append(extractMfcc(ts, sr))
        features.append(extractRms(ts))
        features.append(extractMel(ts, sr))
        features.append(extractLength(ts))
        features.append(extractStd(ts))
        features.append(extractMean(ts))
        return np.array(features)


def extractZcr(ts):
    return np.mean(librosa.feature.zero_crossing_rate(y=ts).T)

def extractChromaStft(ts, sr):
    stft = np.abs(librosa.stft(ts))
    return np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T)

def extractMfcc(ts, sr):
    return np.mean(librosa.feature.mfcc(y=ts, sr=sr).T)

def extractRms(ts):
     return np.mean(librosa.feature.rms(y=ts).T)

def extractMel(ts, sr):
    return np.mean(librosa.feature.melspectrogram(y=ts, sr=sr).T)

def extractMean(ts):
    return np.mean(ts)

def extractStd(ts):
    return np.std(ts)
    
def extractLength(ts):
    return len(ts)