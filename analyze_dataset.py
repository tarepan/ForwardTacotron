import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


def plot_histo(df):
    df_speaker_id = df.groupby('speaker_id').count()
    ax = df_speaker_id.plot.hist(by='wav_path', legend=None)
    ax.set(xlabel='Num Wavs', ylabel='Speaker Count')
    plt.savefig('/tmp/speaker_histo.png')


data_path = '/Users/cschaefe/datasets/LibriTTS/train-clean-100'
wav_paths = list(Path(data_path).glob("**/*.wav"))

data = []
for i, wav_path in enumerate(wav_paths):
    speaker_id = wav_path.parent.parent.stem
    data.append((speaker_id, wav_path))

df = pd.DataFrame(data=data, columns=['speaker_id', 'wav_path'])
print(f'len df: {len(df)}')
plot_histo(df)
