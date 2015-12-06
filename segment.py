import numpy as np
import librosa

class Segment:

    def __init__(self, wave_form, sr, bounds=(0, -1)):
        self.wave_form = wave_form[bounds[0]:bounds[1]]
        self.features = {}
        self.sr = sr

    def extract_features(self):
        # self.features['length'] = np.array([float(len(self.wave_form)) / self.sr])
        mfcc = librosa.feature.mfcc(self.wave_form, self.sr)
        # mfcc_1 = librosa.feature.mfcc(self.wave_form[len(self.wave_form)/2:], self.sr)
        # mfcc_2 = librosa.feature.mfcc(self.wave_form[:len(self.wave_form)/2], self.sr)
        # self.features['mfcc'] = mfcc.mean(axis=1)
        mfcc_split = np.split(mfcc, [mfcc.shape[1]/2], axis=1)
        self.features['mfcc_1'] = mfcc_split[0].mean(axis=1)
        # self.features['mfcc_2'] = mfcc_split[1].mean(axis=1)
        # self.features['delta_mfcc'] = np.abs(librosa.feature.delta(mfcc)).sum(axis=1)
        # self.features['delta_mfcc_1'] = np.abs(librosa.feature.delta(mfcc_1)).sum(axis=1)
        # self.features['delta_mfcc_2'] = np.abs(librosa.feature.delta(mfcc_2)).sum(axis=1)
        # rmse = librosa.feature.rmse(self.wave_form)
        # self.features['max_volume'] = np.array([rmse.max()])
        # self.features['mean_volume'] = np.array([rmse.mean()])
        # self.features['delta_volume'] = np.array([np.abs(librosa.feature.delta(rmse)).sum()])
        pass

    def get_features(self):
        return np.concatenate([value for key, value in self.features.items()])
