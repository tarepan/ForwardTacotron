import tqdm
from flair.data import Sentence
from flair.embeddings import TransformerWordEmbeddings, StackedEmbeddings, FlairEmbeddings
import torch

from utils.files import read_config
from utils.paths import Paths
from utils.text.recipes import ljspeech

if __name__ == '__main__':
    flair_embedding_de = TransformerWordEmbeddings('bert-base-german-cased', subtoken_pooling='mean')
    flair_embedding_en = TransformerWordEmbeddings('bert-base-uncased', subtoken_pooling='mean')
    metafile = ljspeech('/Users/cschaefe/datasets/ASVoice4_incl_english/metadata_clean_incl_english.csv')
    config = read_config('config.yaml')
    paths = Paths(config['data_path'], config['voc_model_id'], config['tts_model_id'])

    for id, text in tqdm.tqdm(metafile.items()):
        sent = Sentence(text)
        if id.startswith('en_'):
            flair_embedding_en.embed(sent)
        else:
            flair_embedding_de.embed(sent)
        vecs = [t.embedding for t in sent.tokens]
        vecs = torch.stack(vecs)
        torch.save(vecs, paths.bert/f'{id}.pt')
        print(f'{id} {text} {vecs.size()}')

