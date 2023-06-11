import json
import os

import click
import evaluate
import numpy as np
import sklearn
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from helpers.crf import SignalLabelDataset, RobertaWithCRF
from helpers.data import get_corpus_path, load_docs
from helpers.evaluate import score_paragraphs
from helpers.labeling import decode_labels


@click.command()
@click.argument('corpus')
@click.argument('relation-type')
@click.argument('save-path')
@click.option('-b', '--batch-size', type=int, default=32)
@click.option('--random-seed', default=42, type=int)
def main(corpus, relation_type, save_path, batch_size, random_seed):
    corpus_path = get_corpus_path(corpus)
    docs = list(load_docs(corpus_path))
    _, test_docs = sklearn.model_selection.train_test_split(docs, test_size=0.1,
                                                            random_state=random_seed)
    test_dataset = SignalLabelDataset(test_docs, relation_type=relation_type)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model_path = os.path.join(save_path, f"best_model_{relation_type.lower()}_label")
    print("Load Single Classifier Model")
    model = RobertaWithCRF.from_pretrained(model_path)
    id2label = model.config.id2label
    model.eval()
    model.to(device)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=SignalLabelDataset.get_collate_fn())

    metric = evaluate.load("poseval")
    losses = []
    signals_pred = []
    signals_gold = []
    scores = {}

    for batch in tqdm(test_dataloader, total=len(test_dataloader), mininterval=5, desc='Eval'):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            loss, logits = model.emissions(**batch)
            losses.append(loss.item())
            preds = torch.argmax(logits, dim=-1)

        predictions = []
        references = []
        for pred, ref in zip(preds.tolist(), batch['labels'].tolist()):
            pred = [id2label[p] for i, p in enumerate(pred) if ref[i] != 7]
            ref = [id2label[i] for i in ref if i != 7]
            assert len(pred) == len(ref), f"PRED: {pred}, REF {ref}"
            predictions.append(pred)
            references.append(ref)
            signals_pred.append([(relation_type, [i for p, i in signal], None) for signal in decode_labels(pred, pred)])
            signals_gold.append([(relation_type, [i for p, i in signal], None) for signal in decode_labels(ref, ref)])
        metric.add_batch(predictions=predictions, references=references)
    loss_valid = np.mean(losses)

    print(f"\n== Evaluation (Linear Classifer Emissions)")
    results = metric.compute(zero_division=0)
    precision, recall, f1 = score_paragraphs(signals_gold, signals_pred, threshold=0.7)
    results['partial-match'] = {
        'precision': precision,
        'recall': recall,
        'f1-score': f1,
        'support': 0,
    }
    precision, recall, f1 = score_paragraphs(signals_gold, signals_pred, threshold=0.9)
    results['exact-match'] = {
        'precision': precision,
        'recall': recall,
        'f1-score': f1,
        'support': 0,
    }
    for key, vals in results.items():
        if key == 'accuracy':
            print(f"{key:10}  {vals * 100:02.2f}")
        else:
            print(
                f"{key:10}  {vals['precision'] * 100:02.2f}  {vals['recall'] * 100:02.2f}  {vals['f1-score'] * 100:02.2f}  {vals['support']}")
    print(f'loss: {loss_valid}')

    scores['label'] = results

    model_path = os.path.join(save_path, f"best_model_{relation_type.lower()}_crf")
    print("Load Single Classifier Model")
    model = RobertaWithCRF.from_pretrained(model_path)
    id2label = model.config.id2label
    model.eval()
    model.to(device)

    metric = evaluate.load("poseval")
    losses = []
    signals_pred = []
    signals_gold = []

    for batch in tqdm(test_dataloader, total=len(test_dataloader), mininterval=5, desc='Eval'):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            loss, sequences = model.decode(**batch)
            losses.append(loss.item())
        predictions = []
        references = []
        for pred, ref in zip(sequences, batch['labels'].tolist()):
            pred = [id2label[p] for i, p in enumerate(pred) if ref[i] != 7]
            ref = [id2label[i] for i in ref if i != 7]
            assert len(pred) == len(ref), f"PRED: {pred}, REF {ref}"
            predictions.append(pred)
            references.append(ref)
            signals_pred.append([(relation_type, [i for p, i in signal], None) for signal in decode_labels(pred, pred)])
            signals_gold.append([(relation_type, [i for p, i in signal], None) for signal in decode_labels(ref, ref)])
        metric.add_batch(predictions=predictions, references=references)
    loss_valid = np.mean(losses)

    print(f"\n\n== Evaluation (Linear Chain CRF)")
    results = metric.compute(zero_division=0)
    precision, recall, f1 = score_paragraphs(signals_gold, signals_pred, threshold=0.7)
    results['partial-match'] = {
        'precision': precision,
        'recall': recall,
        'f1-score': f1,
        'support': 0,
    }
    precision, recall, f1 = score_paragraphs(signals_gold, signals_pred, threshold=0.9)
    results['exact-match'] = {
        'precision': precision,
        'recall': recall,
        'f1-score': f1,
        'support': 0,
    }
    for key, vals in results.items():
        if key == 'accuracy':
            print(f"{key:10}  {vals * 100:02.2f}")
        else:
            print(
                f"{key:10}  {vals['precision'] * 100:02.2f}  {vals['recall'] * 100:02.2f}  {vals['f1-score'] * 100:02.2f}  {vals['support']}")
    print(f'loss: {loss_valid}')
    scores['crf'] = results

    with open(os.path.join(save_path, f'{relation_type}.results.json'), 'w') as fout:
        json.dump(results, fout)


if __name__ == '__main__':
    main()
