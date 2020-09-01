from multiprocessing import cpu_count
from multiprocessing.pool import Pool

from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import librosa
from matplotlib import pyplot as plt

from utils.display import plot_cos_matrix
from utils.dsp import load_wav
from utils.files import pickle_binary

np.set_printoptions(precision=3, suppress=True)

encoder = VoiceEncoder()

files = list(Path('/Users/cschaefe/datasets/LibriTTS/train-clean-100').glob('**/*' + 'wav'))
speaker_files = {}
pool = Pool(processes=cpu_count()-1)


def encode(file):
    wav = preprocess_wav(file)
    semb = encoder.embed_utterance(wav)
    sp_id = file.parent.parent.stem
    return sp_id, semb

sids = {int(file.parent.parent.stem) for file in files}
sid_files = {sid: [] for sid in sids}
sid_emb = {}
for file in files:
    sid = int(file.parent.parent.stem)
    sid_files[sid].append(file)

speaker_ids = sorted(list(sid_files.keys()))[:22]
print(speaker_ids)

for i, sid in enumerate(speaker_ids, 1):
    print(f'{i} {sid} / {len(sid_files)}')
    files = sid_files[sid]
    wavs = [preprocess_wav(f) for f in files[:50]]
    semb = encoder.embed_speaker(wavs)
    sid_emb[sid] = semb


pickle_binary(sid_emb, '/tmp/speaker_emb_dict.pkl')

embeddings = [sid_emb[sid] for sid in speaker_ids]
cos_mat = cosine_similarity(embeddings)
np.fill_diagonal(cos_mat, 0)
plot_cos_matrix(cos_mat, speaker_ids)
plt.savefig('/tmp/cos.png')

