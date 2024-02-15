import csv
import itertools
import os.path
import sys
from pathlib import Path

import click
import torch

from helpers.data import get_corpus_path, load_docs
from helpers.signals import DiscourseSignalModel


def iter_documents_paragraphs(docs):
    def fmt(par, par_i):
        return {
            'doc_id': doc.doc_id,
            'paragraph_idx': par_i,
            'sentences': par,
        }

    for doc in docs:
        par = []
        par_i = 0
        for s in doc.sentences:
            if len(par) == 0 or (par[-1].tokens[-1].offset_end + 1 == s.tokens[0].offset_begin):
                par.append(s)
            else:
                yield fmt(par, par_i)
                par_i += 1
                par = [s]
        yield fmt(par, par_i)


def get_quick_context_tokens(par, rel):
    par_offset = par['tokens_idx'][0]
    left = max(min(rel['tokens_idx']) - par_offset - 5, 0)
    right = min(max(rel['tokens_idx']) - par_offset + 5, len(par['tokens']))
    return [par['tokens'][i] for i in range(left, right)]


@click.command()
@click.argument('corpus')
@click.argument('relation-type')
@click.argument('save-path')
@click.option('-r', '--replace', is_flag=True)
@click.option('-o', '--output-path', default='-')
@click.option('-l', '--limit', default=0, type=int)
@click.option('-b', '--batch-size', default=32, type=int)
@click.option('--use-crf', default=True, type=bool)
def main(corpus, relation_type, save_path, replace, output_path, limit, batch_size, use_crf):
    if output_path == '-':
        output = sys.stdout
    else:
        output_path = Path(output_path)
        if output_path.is_file() and output_path.stat().st_size > 100 and not replace:
            sys.stderr.write('File already exists: Exit without writing.')
            return
        else:
            output_path.parent.mkdir(exist_ok=True, parents=True)
            output = output_path.open('w')

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    signal_model = DiscourseSignalModel.load_model(save_path, relation_type, device=device, use_crf=use_crf)

    csv_out = csv.writer(output)
    csv_out.writerow(("corpus relation-type doc-id paragraph-id sentence-id token-ids tokens "
                      "coarse-sense coarse-prob fine-sense fine-prob context").split())

    if os.path.isfile(corpus):
        corpus_path = corpus
    else:
        corpus_path = get_corpus_path(corpus)

    paragraphs = filter(lambda p: sum(len(s.tokens) for s in p['sentences']) > 7,
                        iter_documents_paragraphs(load_docs(corpus_path, limit=limit)))
    while True:
        batch = list(itertools.islice(paragraphs, batch_size))
        if len(batch) == 0:
            break
        par_preds = signal_model.predict_paragraphs(batch)
        for par in filter(lambda p: len(p['relations']) > 0, par_preds):
            for rel in par['relations']:
                csv_out.writerow([
                    corpus, relation_type.lower(),
                    par['doc_id'], par['paragraph_idx'], '-'.join(map(str, rel['sentence_idx'])),
                    '-'.join(str(i) for i in rel['tokens_idx']),
                    '-'.join(t.lower() for t in rel['tokens']),
                    rel['coarse'], rel['coarse_probs'],
                    rel['fine'], rel['fine_probs'],
                    '-'.join(get_quick_context_tokens(par, rel))
                ])

        output.flush()


if __name__ == '__main__':
    main()
