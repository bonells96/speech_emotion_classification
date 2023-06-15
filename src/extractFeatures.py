import numpy as np
import librosa
import pandas as pd

class FeatureExtractor:
    """
    Class for extracting features from audio time series data.
    """

    def __init__(self):
        self.features = {
            'MeanZcr': extractMeanZcr,
            'StdZcr': extractStdZcr,
            'MedianZcr': extractMedianZcr,
            'MeanChromaStft': extractMeanChromaStft,
            'StdChromaStft': extractStdChromaStft,
            'MedianChromaStft': extractMedianChromaStft,
            'MeanMfcc': extractMeanMfcc,
            'StdMfcc': extractStdMfcc,
            'MedianMfcc': extractMedianMfcc,
            'MeanRMS': extractMeanRms,
            'StdRMS': extractStdRms,
            'MedianRMS': extractMedianRms,
            'MeanMEL': extractMeanMel,
            'StdMEL': extractStdMel,
            'MedianMel': extractMedianMel,
            'length': extractLength,
            'Mean': extractMean,
            'Std': extractStd,
            'Median': extractMedian
        }

    def __call__(self, ts, sr, selected_features=None):
        """
        Extracts features from audio time series data.

        Inputs:
            ts (numpy.ndarray): Audio time series data.
            sr (int): Sample rate of the audio data.
            selected_features (list, optional): Features to extract.

        Returns:
            numpy.ndarray: Extracted features.
        """
        if selected_features is None:
            selected_features = self.features.keys()
        features = np.array([])
        for feature in selected_features:
            try:
                features = np.hstack((features, self.features[feature](ts, sr)))
            except:
                features = np.hstack((features, self.features[feature](ts)))
        return features



def extractMeanZcr(ts):
    return np.mean(librosa.feature.zero_crossing_rate(y=ts).T)

def extractStdZcr(ts):
    return np.std(librosa.feature.zero_crossing_rate(y=ts).T)

def extractMedianZcr(ts):
    return np.median(librosa.feature.zero_crossing_rate(y=ts).T)

def extractMeanChromaStft(ts, sr):
    stft = np.abs(librosa.stft(ts))
    return np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T)

def extractStdChromaStft(ts, sr):
    stft = np.abs(librosa.stft(ts))
    return np.std(librosa.feature.chroma_stft(S=stft, sr=sr).T)

def extractMedianChromaStft(ts, sr):
    stft = np.abs(librosa.stft(ts))
    return np.median(librosa.feature.chroma_stft(S=stft, sr=sr).T)

def extractMeanMfcc(ts, sr):
    return np.mean(librosa.feature.mfcc(y=ts, sr=sr).T)

def extractStdMfcc(ts, sr):
    return np.std(librosa.feature.mfcc(y=ts, sr=sr).T)

def extractMedianMfcc(ts, sr):
    return np.median(librosa.feature.mfcc(y=ts, sr=sr).T)

def extractMeanRms(ts):
     return np.mean(librosa.feature.rms(y=ts).T)

def extractStdRms(ts):
     return np.std(librosa.feature.rms(y=ts).T)

def extractMedianRms(ts):
     return np.median(librosa.feature.rms(y=ts).T)

def extractMeanMel(ts, sr):
    return np.mean(librosa.feature.melspectrogram(y=ts, sr=sr).T)

def extractStdMel(ts, sr):
    return np.std(librosa.feature.melspectrogram(y=ts, sr=sr).T)

def extractMedianMel(ts, sr):
    return np.median(librosa.feature.melspectrogram(y=ts, sr=sr).T)

def extractMeanTonnetz(ts, sr):
    return np.mean(librosa.feature.tonnetz(y=ts, sr=sr).T)

def extractStdTonnetz(ts, sr):
    return np.mean(librosa.feature.tonnetz(y=ts, sr=sr).T)

def extractMedianTonnetz(ts, sr):
    return np.mean(librosa.feature.tonnetz(y=ts, sr=sr).T)

def extractMean(ts):
    return np.mean(ts)

def extractStd(ts):
    return np.std(ts)

def extractMedian(ts):
    return np.median(ts)
    
def extractLength(ts):
    return len(ts)