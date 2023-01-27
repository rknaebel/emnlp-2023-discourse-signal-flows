import bz2
import json
import random

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


def load_docs(bzip_file_path):
    for line_i, line in tqdm(enumerate(bz2.open(filename=bzip_file_path, mode='rt'))):
        try:
            yield Document.from_json(json.loads(line))
        except json.JSONDecodeError:
            continue
        except EOFError:
            break


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


def get_sense(sense, level=2):
    return '.'.join(sense.split('.')[:level])
