from phonemizer.phonemize import phonemize

from utils import hparams as hp

""" from https://github.com/keithito/tacotron """

'''
Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You'll typically want to use:
  1. "english_cleaners" for English text
  2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
     the Unidecode library (https://pypi.python.org/pypi/Unidecode)
  3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
     the symbols in symbols.py to match your data).
'''

import re
from unidecode import unidecode
from .numbers import normalize_numbers

# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+')

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
    ('mrs', 'misess'),
    ('mr', 'mister'),
    ('dr', 'doctor'),
    ('st', 'saint'),
    ('co', 'company'),
    ('jr', 'junior'),
    ('maj', 'major'),
    ('gen', 'general'),
    ('drs', 'doctors'),
    ('rev', 'reverend'),
    ('lt', 'lieutenant'),
    ('hon', 'honorable'),
    ('sgt', 'sergeant'),
    ('capt', 'captain'),
    ('esq', 'esquire'),
    ('ltd', 'limited'),
    ('col', 'colonel'),
    ('ft', 'fort'),
    ('deg', 'degrees'),
    ('lbs', 'pounds.'),
    ('lb', 'pounds.'),
    ('S', 'south'),
    ('N', 'north'),
    ('W', 'west'),
    ('E', 'east'),
    ('lat', 'latitude'),
]]

_space_abbreviations = [(re.compile('\\b%s' % x[0], re.IGNORECASE), x[1]) for x in [
    ('II ', 'two '),
    ('III ', 'three '),
    ('IV ', 'four '),
    ('V ', 'five '),
    ('VI ', 'six '),
    ('VII ', 'seven '),
    ('VIII ', 'eight '),
    ('IX ', 'nine '),
    ('X  ', 'ten  '),
    ('XI ', 'eleven '),
    ('XII ', 'twelve '),
    ('XIII ', 'thirteen '),
    ('XIV ', 'fourteen '),
    ('XV ', 'fifteen '),
    ('XVI ', 'sixteen '),
    ('XVII ', 'seventeen '),
    ('XVIII ', 'eighteen '),
    ('XIX ', 'nineteen '),
    ('XX ', 'twenty '),
]]
_other_abbreviations = [
    ('(', ','),
    (')', ','),
    ('[', ','),
    (']', ','),
    ('"', ''),
    ('_', ''),
    ("!'", '!') # this breaks phonemizer
]


def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    for regex, replacement in _space_abbreviations:
        text = re.sub(regex, replacement, text)
    for item in _other_abbreviations:
        text = text.replace(item[0], item[1])
    return text


def expand_numbers(text):
    return normalize_numbers(text)


def lowercase(text):
    return text.lower()


def collapse_whitespace(text):
    return re.sub(_whitespace_re, ' ', text)


def convert_to_ascii(text):
    return unidecode(text)


def basic_cleaners(text):
    text = to_phonemes(text)
    text = collapse_whitespace(text)
    return text


def english_cleaners(text, phonemize=True):
    text = convert_to_ascii(text)
    text = expand_abbreviations(text)
    text = expand_numbers(text)
    if phonemize:
        text = to_phonemes(text)
    text = collapse_whitespace(text)
    return text


def to_phonemes(text):
    text = text.replace('-', '—')
    phonemes = phonemize(text,
                         language=hp.language,
                         backend='espeak',
                         strip=True,
                         preserve_punctuation=True,
                         with_stress=True,
                         njobs=1,
                         punctuation_marks=';:,.!?¡¿—…"«»“”()',
                         language_switch='remove-flags')
    phonemes = phonemes.replace('—', '-')
    return phonemes
