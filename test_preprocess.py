from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import numpy as np
import librosa
from utils.dsp import load_wav

np.set_printoptions(precision=3, suppress=True)

encoder = VoiceEncoder()

fpath = Path('/Users/cschaefe/datasets/LibriTTS/test-clean/1089/134686/1089_134686_000035_000000.wav')


wav = preprocess_wav(fpath)
wav2 = librosa.load(fpath, sr=22050)[0]
wav2 = preprocess_wav(wav2, source_sr=22050)

embed = encoder.embed_utterance(wav)
embed2 = encoder.embed_utterance(wav2)

print(embed)
print(embed2)