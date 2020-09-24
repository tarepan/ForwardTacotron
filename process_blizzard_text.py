from pathlib import Path
import pickle

import tqdm
from phonemizer.phonemize import phonemize

# import argparse
from utils.text import cleaners
from utils import hparams as hp

if __name__ == '__main__':
    main_dir = '/Volumes/data/datasets/blizzard/'
    main_dir = Path(main_dir)
    if not (main_dir / 'text_dict_original.pkl').exists():
        text_data1 = main_dir / 'BC2013_segmented_v0_txt1'
        text_data2 = main_dir / 'BC2013_segmented_v0_txt2'
        text_dirs1 = [d for d in text_data1.iterdir() if d.is_dir()]
        text_dirs2 = [d for d in text_data2.iterdir() if d.is_dir()]
        text_dirs = [text_dirs1, text_dirs2]
        text_lines = {}
        print('Reading text')
        for text_dir in text_dirs:
            for directory in text_dir:
                print(f'directory {directory}')
                files = [f for f in directory.iterdir() if f.suffix == '.txt']
                lines = {f.with_suffix('').name: f.open().readline() for f in files}
                text_lines.update(lines)
        
        pickle.dump(text_lines, open(main_dir / 'text_dict_original.pkl', 'wb'))
    if not (main_dir / 'text_dict_cleaned.pkl').exists():
        original_text = pickle.load(open(main_dir / 'text_dict_original.pkl', 'rb'))
        hp.configure('../hparams.py')
        cleaned_data = {}
        for file in tqdm.tqdm(original_text):
            line = original_text[file]
            cleaned_data[file] = cleaners.english_cleaners(line, phonemize=False)
        pickle.dump(cleaned_data, open(main_dir / 'text_dict_cleaned.pkl', 'wb'))
    if not (main_dir / 'text_dict_phonemized.pkl').exists():
        cleaned_text = pickle.load(open(main_dir / 'text_dict_cleaned.pkl', 'rb'))
        key_list = list(cleaned_text.keys())
        phonemized_data = {}
        batch_size = 256
        failed_files = []
        for i in tqdm.tqdm(range(0, len(key_list) + batch_size, batch_size)):
            batch_keys = key_list[i:i + batch_size]
            try:
                batch_text = [cleaned_text
                              [k] for k in batch_keys]
                if len(batch_text) == 0:
                    break
                phonemized_batch = phonemize(batch_text,
                                             language='en-gb',
                                             backend='espeak',
                                             strip=True,
                                             preserve_punctuation=True,
                                             with_stress=True,
                                             njobs=16,
                                             punctuation_marks=';:,.!?¡¿—…"«»“”()',
                                             language_switch='remove-flags')
                phonemized_data.update(dict(zip(batch_keys, phonemized_batch)))
            except:
                failed_files.extend(batch_keys)
                
        for file in failed_files:
            text = cleaned_text[file]
            phonemized_text = phonemize(text,
                                         language='en-gb',
                                         backend='espeak',
                                         strip=True,
                                         preserve_punctuation=True,
                                         with_stress=True,
                                         njobs=1,
                                         punctuation_marks=';:,.!?¡¿—…"«»“”()',
                                         language_switch='remove-flags')
            phonemized_data.update({file: phonemized_text})
        pickle.dump(phonemized_data, open(main_dir / 'text_dict.pkl', 'wb'))
    print('Done.')
