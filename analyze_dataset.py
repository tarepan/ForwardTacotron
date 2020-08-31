import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import librosa
from multiprocessing import Pool, cpu_count


def plot_by_speaker_id(df, col_name, save_path=None):
    df = df[:]
    df_mean = df.groupby('speaker_id')[col_name].mean().sort_values()
    df_std = df.groupby('speaker_id')[col_name].std().sort_values()
    ax = df_mean.plot(figsize=(50, 8), legend=False, kind="bar", rot=45, color="lightblue", yerr=df_std)
    ax.set(ylabel=col_name, xlabel='speaker_id')
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, format='pdf')
    plt.close()


def gen_dataset(wav_paths):
    pool = Pool(processes=cpu_count()-1)
    data = []
    for i, (sp_id, wav_path, w_max, w_avg, w_len, line_len) in enumerate(pool.imap_unordered(load_file, wav_paths)):
        print(f'{i} {w_max} {line_len} | {len(wav_paths)}')
        data.append((sp_id, wav_path, w_max, w_avg, w_len, line_len))
    df = pd.DataFrame(data=data, columns=['speaker_id', 'wav_path', 'w_max', 'w_avg', 'w_len', 'line_len'])
    print(f'len df: {len(df)}')
    return df


def load_file(wav_path):
    speaker_id = wav_path.parent.parent.stem
    wav, sr = librosa.load(wav_path)
    w_max, w_avg, w_len = max(wav), max(wav) / sum(abs(wav)), len(wav)
    with open(str(wav_path).replace('.wav', '.normalized.txt')) as f:
        line = f.readlines()[0]
        line_len = len(line)
    return speaker_id, wav_path, w_max, w_avg, w_len, line_len


data_path = '/Users/cschaefe/datasets/LibriTTS/train-clean-100'
wav_paths = list(Path(data_path).glob("**/*.wav"))
df = gen_dataset(wav_paths)
df.to_csv('/tmp/train-clean-100.csv')
#df = pd.read_csv('/tmp/train-clean-100.csv')
df['w_len_t_len'] = df['w_len'] / df['line_len']
df.to_csv('/tmp/train-clean-100-proc.csv')

print(df.head())
plot_by_speaker_id(df, 'w_max', '/tmp/w_max.pdf')
plot_by_speaker_id(df, 'w_len', '/tmp/w_len.pdf')
plot_by_speaker_id(df, 'w_len_t_len', '/tmp/w_len_t_len.pdf')


#plot_histo(df, 'speaker_id')
#plot_histo(df, 'line_len')
