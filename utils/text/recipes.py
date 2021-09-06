from utils.files import get_files
from pathlib import Path
from typing import Union


def ljspeech(path: Union[str, Path]):
    if str(path).endswith('.csv') or str(path).endswith('txt'):
        csv_file = path
    else:
        csv_file = get_files(path, extension='.csv')
        assert len(csv_file) == 1
        csv_file = csv_file[0]
    text_dict = {}
    with open(str(csv_file), encoding='utf-8') as f:
        for line in f:
            split = line.split('|')
            text_dict[split[0]] = split[-1]
    return text_dict