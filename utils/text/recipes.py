from utils.files import get_files
from pathlib import Path
from typing import Union
import pandas as pd

def ljspeech(path: Union[str, Path]):
    csv_file = get_files(path, extension='.csv')
    assert len(csv_file) == 1
    text_dict = {}
    with open(str(csv_file[0]), encoding='utf-8') as f:
        for line in f:
            split = line.split('|')
            text_dict[split[0]] = split[-1]
    return text_dict


def voxpopuli(path: Union[str, Path]):
    df = pd.read_csv(path, sep='|', encoding='utf-8')
    df.dropna(inplace=True)
    text_dict = {}
    for id_, session_id, phons in zip(df['id_'], df['session_id'], df['normed_text']):
        if len(phons) > 3:
            text_dict[session_id + '-' + id_] = phons
    return text_dict
