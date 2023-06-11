import glob
import json
import os

import click
import evaluate
import sklearn
import torch
from torch.utils.data import DataLoader

from helpers.data import get_corpus_path, load_docs
from helpers.senses import ConnSenseDataset, DiscourseSenseClassifier, DiscourseSenseEnsembleClassifier, \
    get_sense_mapping
from helpers.stats import print_metrics_results


@click.command()
@click.argument('corpus')
@click.argument('relation-type')
@click.option('-b', '--batch-size', type=int, default=8)
@click.option('--save-path', default="")
@click.option('--random-seed', default=42, type=int)
@click.option('--simple-sense', is_flag=True)
def main(corpus, relation_type, batch_size, save_path, random_seed, simple_sense):
    corpus_path = get_corpus_path(corpus)
    train_docs = list(load_docs(corpus_path))
    labels_coarse, labels_fine = get_sense_mapping(train_docs)
    train_docs, test_docs = sklearn.model_selection.train_test_split(train_docs, test_size=0.1,
                                                                     random_state=random_seed)

    explicit_dataset = ConnSenseDataset(test_docs, relation_type='explicit',
                                        labels_coarse=labels_coarse, labels_fine=labels_fine, simple_sense=simple_sense)
    explicit_dataloader = DataLoader(explicit_dataset, shuffle=True, batch_size=batch_size,
                                     collate_fn=ConnSenseDataset.get_collate_fn())

    altlex_dataset = ConnSenseDataset(test_docs, relation_type='altlex',
                                      labels_coarse=labels_coarse, labels_fine=labels_fine, simple_sense=simple_sense)
    altlex_dataloader = DataLoader(altlex_dataset, shuffle=True, batch_size=batch_size,
                                   collate_fn=ConnSenseDataset.get_collate_fn())

    id2label_coarse = {v: k for k, v in labels_coarse.items()}
    id2label_fine = {v: k for k, v in labels_fine.items()}

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    save_paths = glob.glob(save_path)
    if len(save_paths) == 1:
        print("Load Single Classifier Model")
        model = DiscourseSenseClassifier.load(save_path, device, relation_type=relation_type)
        model.to(device)
    else:
        print("Load Ensemble Model")
        model = DiscourseSenseEnsembleClassifier.load(save_paths, device, relation_type=relation_type)

    metric_coarse = evaluate.load("poseval")
    metric_fine = evaluate.load("poseval")
    model.eval()
    scores = {}
    for reltype, eval_dataloader in [('explicit', explicit_dataloader), ('altlex', altlex_dataloader)]:
        print(f"##\n## EVAL {reltype}\n##")
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            output = model.predict(batch['inputs'], no_none=True)
            references = [id2label_coarse[i] for i in batch['labels_coarse'].tolist()]
            metric_coarse.add_batch(predictions=[output['coarse']], references=[references])
            references = [id2label_fine[i] for i in batch['labels_fine'].tolist()]
            metric_fine.add_batch(predictions=[output['fine']], references=[references])

        results_coarse = metric_coarse.compute(zero_division=0)
        print_metrics_results(results_coarse)

        results_fine = metric_fine.compute(zero_division=0)
        print_metrics_results(results_fine)
        scores[reltype] = {
            'coarse': results_coarse,
            'fine': results_fine,
        }

    with open(os.path.join(save_path, f'{relation_type}.results.json'), 'w') as fout:
        json.dump(scores, fout)


if __name__ == '__main__':
    main()
