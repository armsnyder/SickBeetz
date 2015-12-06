from scipy.io import wavfile

class AudioFile:
    def __init__(self, wave_file):
        self.sr, self.wave_form = wavfile.read(wave_file)
