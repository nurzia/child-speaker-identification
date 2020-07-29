import matplotlib as mpl
mpl.use('Agg')

import glob

import os
import shutil

from tqdm import tqdm
import pandas as pd
import numpy as np
import librosa
from sklearn.model_selection import train_test_split as split
# import matplotlib.pyplot as plt
from scipy.signal import spectral
import json

CATEGORIES = 'U0', 'U1', 'U2', 'I0', 'I1', 'I2'


def spectrogram(audio, fft_freq=1024, melW=0,
                win_length=1024, hop_length=512):
    """
    Creates a spectrogram of the input wave
    """
    ham_win = np.hamming(win_length)
    X = spectral.spectrogram(
        x=audio,
        nfft=fft_freq,
        window=ham_win,
        nperseg=win_length,
        noverlap=win_length - hop_length,
        detrend=False,
        return_onesided=True,
        mode='magnitude')[-1].T
    print('X PROVA:\n', X)
    if melW is not None:
         X = np.dot(X, melW)
    return np.log(X + 1e-8).astype(np.float32)


def load_signal(path, duration=None):
    """
    Loads signal from a wave file and returns it as mono
    """
    y, sr = librosa.load(path=path, sr=None, mono=True)
    if duration and isinstance(duration, int):
        y = y[0:sr * duration]
    return y, sr


def load_annotated_intervals(path, class_types=CATEGORIES):
    """
    Extracts time intervals from a transcription
    """
    intervals, metadata = [], []
    with open(path) as f:
        lines = f.readlines()
    for index, line in enumerate(lines):
        if line.startswith('*CHI:'):
            try:
                interval = line.split('\t')[1][1:][:-2].split('_')
                interval = [int(i.strip()) for i in interval]
                code = lines[index + 1][-3:-1]
                if lines[index + 2].startswith('%com'):
                    comment = lines[index + 2][6:].strip()
                else:
                    comment = ''
                if code not in class_types:
                    continue
                intervals.append(interval)
                metadata.append([code, comment])
            except (ValueError, IndexError):
                pass
    return intervals, metadata


def load_intervals(path):
    """
    Extracts time intervals from a transcription.
    """
    intervals, metadata = [], []
    with open(path) as f:
        for line in f:
            if line.startswith('*CHI:'):
                try:
                    interval = line.split('\t')[1][1:][:-2].split('_')
                    interval = [int(i.strip()) for i in interval]
                    intervals.append(interval)
                    # for the unannotated data, we add dummy codes:
                    metadata.append(['', ''])
                except (ValueError, IndexError):
                    pass
    return intervals, metadata


def extract_utterances(dir, min_len, max_len, duration, annotated=True):
    """
    Loads relevant utterances from audio files under `indir`
    """
    try:
        shutil.rmtree(dir + '/wav')
    except FileNotFoundError:
        pass
    os.mkdir(dir + '/wav')

    with open('hearing.json') as f:
        hearing_list = json.loads(f.read())

    audio_files = glob.glob(dir + '/orig/' + '**/*.wav', recursive=True)

    cha_files = [f.replace('.wav', '.cha').lower() for f in audio_files]
    metadata = []

    for audio_f, cha_f in tqdm(list(zip(audio_files, cha_files))):
        signal, sr = load_signal(audio_f, duration=duration)
        if annotated:
            intervals, meta = load_annotated_intervals(cha_f)
        else:
            intervals, meta = load_intervals(cha_f)

        fn = os.path.basename(audio_f).replace('.wav', '')
        child = fn[:3].lower()
        for element in hearing_list:
            if element[:3].lower() == child.lower():
                if hearing_list[element] == 0:
                    hearing = 'NH'
                elif hearing_list[element] == 1:
                    hearing = 'CI'
        age = str(int(fn[3:5]) * 12 + int(fn[5:7]))  # extract date from file name

        for idx, (interval, m) in enumerate(zip(intervals, meta)):
            segment = signal[int(interval[0] * sr / 1000): int(interval[1] * sr / 1000)]
            if int(min_len * sr) <= len(segment) < int(max_len * sr):
                fn_ = fn + '-' + str(idx).zfill(6)
                np.save(dir + '/wav/' + fn_ + '.npy', segment)
                metadata.append([fn_, child, age] + m + [hearing])

    df = pd.DataFrame(metadata,
                      columns={'filename', 'child', 'age', 'category', 'comment', 'NH/CI'})
    df.to_excel(dir + '/meta.xlsx')


def extract_spectrograms(dir, fft_freq=1024, win_length=1024,
                         minf=0, maxf=8000, hop_length=512,
                         mel_freq=64, sr=44100):
    """
    Loads all the spectrograms of all utterances.
    """
    try:
        shutil.rmtree(dir + '/spec')
    except FileNotFoundError:
        pass
    os.mkdir(dir + '/spec')

    if mel_freq:
        melW = librosa.filters.mel(sr=sr, n_fft=fft_freq,
                                   n_mels=mel_freq,
                                   fmin=minf, fmax=maxf).T
    else:
        melW = None

    lengths = []
    for wav_f in tqdm(glob.glob(dir + '/wav/*.npy')):
        wav = np.load(wav_f)
        spec = spectrogram(wav, fft_freq=fft_freq, win_length=win_length,
                           hop_length=hop_length, melW=melW)

        name = os.path.basename(wav_f)
        np.save(dir + '/spec/' + name, spec)
        lengths.append(spec.shape[0])

    lengths = np.array(lengths)

    if mel_freq:
        feat_dim = mel_freq
    else:
        feat_dim = int(1 + fft_freq / 2)

    return {'feat_dim': feat_dim,
            'min_len': int(lengths.min()),
            'max_len': int(lengths.max()),
            'mean_len': float(lengths.mean()),
            'std_len': float(lengths.std())}


def metadata_split(df, train_size=.8, seed=8786):
    """
     Writes data of each spectrogram on a table of an excel sheet.
    """
    train, rest = split(df, stratify=df['child'],
                        train_size=train_size,
                        random_state=seed, shuffle=True)
    dev, test = split(rest, stratify=rest['child'],
                      train_size=.5,
                      random_state=seed, shuffle=True)

    # train.to_excel(dir_ + '/train.xlsx')
    # dev.to_excel(dir_ + '/dev.xlsx')
    # test.to_excel(dir_ + '/test.xlsx')

    return train, dev, test


# def viz_spectograms(dir, length):
#     """
#     Saves spectrograms as images.
#     """
#     try:
#         shutil.rmtree(dir+'/viz')
#     except FileNotFoundError:
#         pass
#     os.mkdir(dir + '/viz')

#     for spec_f in glob.glob(dir + '/spec/*.npy'):
#         spec = pad(np.load(spec_f), length).transpose()
#         plt.figure(figsize=(12, 8))
#         librosa.display.specshow(librosa.amplitude_to_db(spec, ref=np.max),
#                                  y_axis='log', x_axis='time')
#         plt.savefig(dir + '/viz/' + os.path.basename(spec_f).replace(".npy", ".png"),
#                     bbox_inches=None, pad_inches=0)
#         plt.close()


def pad(spec, length, fill_value=0):
    """
    function that pads a spectrogram with zeros or other values according to the length parameter
    """
    if spec.shape[0] < length:
        padded = np.full((length, spec.shape[1]),
                         fill_value=fill_value, dtype=np.float64)
        padded[:spec.shape[0], : spec.shape[1]] += spec
        return padded
    else:
        return spec[:length, :]
