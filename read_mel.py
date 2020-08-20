import numpy as np
import argparse
import matplotlib.pyplot as plt
import time
import numpy as np
import sys
from utils import hparams as hp


from utils.dsp import reconstruct_waveform, save_wav

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TTS Generator')
    parser.add_argument('--input_text', '-i', type=str, help='[string] Type in something here and TTS will generate it!')
    parser.add_argument('--tts_weights', type=str, help='[string/path] Load in different FastSpeech weights')
    parser.add_argument('--save_attention', '-a', dest='save_attn', action='store_true', help='Save Attention Plots')
    parser.add_argument('--force_cpu', '-c', action='store_true', help='Forces CPU-only training, even when in CUDA capable environment')
    parser.add_argument('--hp_file', metavar='FILE', default='hparams.py', help='The file to use for the hyperparameters')
    args = parser.parse_args()
    hp.configure(args.hp_file)  # Load hparams from file

    with open('/Users/cschaefe/hackathon/ein_test.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    data = lines[0].split(', ')
    data = [float(d) for d in data]
    data = np.array(data)
    mel = data.reshape((80, -1))
    #mel = np.flip(mel, axis=0)
    #fig = plt.figure(figsize=(12, 6), dpi=150)
    #plt.imshow(mel, interpolation='nearest', aspect='auto')
    #plt.show()

    wav = reconstruct_waveform(mel, n_iter=32)
    save_wav(wav, '/tmp/rec3.wav')