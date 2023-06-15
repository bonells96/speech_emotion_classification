import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import numpy as np

def plot_nets(accs_train, accs_test, smooth_curves=True):
    """
    Plot the smoothed accuracy per category over epochs.

    Args:
        accs_train (numpy.ndarray): Train accuracy values.
        accs_test (numpy.ndarray): Test accuracy values.
        smooth_curves (bool, optional): Whether to smooth the curves. 

    Returns:
        plt: The matplotlib plot object.
    """
    window_size = 0
    if smooth_curves:
        window_size = 5
        accs_train = np.convolve(accs_train, np.ones(window_size) / window_size, mode='same')
        accs_test = np.convolve(accs_test, np.ones(window_size) / window_size, mode='same')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.set_style("whitegrid")
    sns.lineplot(x=np.arange(len(accs_train)-5), y=accs_train[:-window_size], ax=ax, label = 'train accuracy')
    sns.lineplot(x=np.arange(len(accs_test)-5), y=accs_test[:-window_size], ax=ax, label = 'test accuracy')

    sns.despine()
    plt.tight_layout()

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Smoothed Accuracy per Category over Epochs')
    plt.legend()
    plt.show()
    return plt


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