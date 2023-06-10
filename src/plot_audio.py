import matplotlib.pyplot as plt
import seaborn as sns
import librosa


def create_waveplot(ts, sr, label):
    "Plot time series"
    plt.figure(figsize=(10, 3))
    plt.title(f'Waveplot for audio with {label} emotion', size=15)
    librosa.display.waveshow(ts, sr=sr)
    plt.show()

def create_spectrogram(ts, sr, label):
    "Plot spectogram"
    # stft function converts the data into short term fourier transform
    X = librosa.stft(ts)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(12, 3))
    plt.title(f'Spectrogram for audio with {label} emotion', size=15)
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')   
    #librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar()