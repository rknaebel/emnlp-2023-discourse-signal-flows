import bz2
import json
import random

import numpy as np
import torch
from discopy_data.data.doc import Document
from tqdm import tqdm

source_path = '/cache/discourse'
paths = {
    'pdtb3': f'{source_path}/pdtb3.en.v3.json.bz2',
    'tedmdb': f'{source_path}/tedmdb.en.v3.json.bz2',
    'because': f'{source_path}/because.v3.json.bz2',
    'biodrb': f'{source_path}/biodrb.v2.json.bz2',
    'biocause': f'{source_path}/biocause.v1.json.bz2',
    'unsc': f'{source_path}/unsc.v1.json.bz2',
    'ted': f'{source_path}/ted.v1.json.bz2',
    'anthology': f'{source_path}/anthology.v1.json.bz2',
    'essay': f'{source_path}/essay.v1.json.bz2',
}


def get_corpus_path(corpus):
    return paths.get(corpus)


def load_docs(bzip_file_path, limit=0):
    try:
        for line_i, line in tqdm(enumerate(bz2.open(filename=bzip_file_path, mode='rt')), desc='Load Documents'):
            if limit and line_i > limit:
                break
            try:
                yield Document.from_json(json.loads(line))
            except json.JSONDecodeError:
                continue
    except EOFError:
        print('Stopped iterator')


def load_all_datasets():
    return {k: load_docs(v) for k, v in paths.items()}


def load_dataset(key):
    return load_docs(paths[key])


def split_train_test(xs, ratio=0.9):
    xs = xs[:]
    num_samples = int(len(xs) * ratio)
    random.shuffle(xs)
    return xs[:num_samples], xs[num_samples:]


def iter_document_paragraphs(doc):
    par = []
    for s in doc.sentences:
        if len(par) == 0 or (par[-1].tokens[-1].offset_end + 1 == s.tokens[0].offset_begin):
            par.append(s)
        else:
            yield par
            par = [s]
    yield par


def get_sense(sense, level=2, simple_sense=False):
    def simplify(sense_level):
        if sense_level.lower() == 'negative-condition':
            return 'Condition'
        else:
            return sense_level.split('+')[0]

    if simple_sense:
        sense_levels = [simplify(sl) for sl in sense.split('.')]
    else:
        sense_levels = sense.split('.')
    return '.'.join(sense_levels[:level])


simple_map = {
    "''": '"',
    "``": '"',
    "-LRB-": "(",
    "-RRB-": ")",
    "-LCB-": "{",
    "-RCB-": "}",
    "n't": "not"
}


def get_doc_embeddings(sentences, tokenizer, model, last_hidden_only=False, device='cpu'):
    tokens = [[simple_map.get(t.surface, t.surface) for t in sent.tokens] for sent in sentences]
    subtokens = [[tokenizer.tokenize(t) for t in sent] for sent in tokens]
    lengths = [[len(t) for t in s] for s in subtokens]
    inputs = tokenizer(tokens, padding=True, return_tensors='pt', is_split_into_words=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    if last_hidden_only:
        hidden_state = outputs.hidden_states[-2].detach().cpu().numpy()
    else:
        hidden_state = torch.cat(outputs.hidden_states[-4:], axis=-1).detach().cpu().numpy()
    embeddings = np.zeros((sum(len(s) for s in tokens), hidden_state.shape[-1]), np.float32)
    e_i = 0
    for sent_i, _ in enumerate(inputs['input_ids']):
        len_left = 1
        for length in lengths[sent_i]:
            embeddings[e_i] = hidden_state[sent_i][len_left]
            len_left += length
            e_i += 1
    return embeddings
