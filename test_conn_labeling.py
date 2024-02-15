import glob
import os

import click
import evaluate
import sklearn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForTokenClassification

from helpers.data import get_corpus_path, load_docs
from helpers.evaluate import score_paragraphs
from helpers.labeling import SignalLabelDataset, decode_labels


def compute_ensemble_prediction(models, batch):
    predictions = []
    with torch.no_grad():
        for model in models:
            outputs = model(**batch)
            predictions.append(F.softmax(outputs.logits, dim=-1))
    return torch.argmax(torch.sum(torch.stack(predictions), dim=0), dim=-1)


@click.command()
@click.argument('corpus')
@click.argument('relation-type')
@click.option('-b', '--batch-size', type=int, default=32)
@click.option('--save-path', default=".")
@click.option('--random-seed', default=42, type=int)
def main(corpus, relation_type, batch_size, save_path, random_seed):
    corpus_path = get_corpus_path(corpus)
    train_docs = list(load_docs(corpus_path))
    train_docs, test_docs = sklearn.model_selection.train_test_split(train_docs, test_size=0.1,
                                                                     random_state=random_seed)
    test_dataset = SignalLabelDataset(test_docs, relation_type=relation_type)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    save_path = os.path.join(save_path, f"best_model_{relation_type.lower()}_label")
    save_paths = glob.glob(save_path)
    print(save_paths)
    if len(save_paths) == 1:
        print("Load Single Classifier Model")
        model = AutoModelForTokenClassification.from_pretrained(save_paths[0])
        id2label = model.config.id2label
        model.eval()
        model.to(device)
        models = [model]
    else:
        print("Load Ensemble Model")
        models = []
        for save_path in save_paths:
            model = AutoModelForTokenClassification.from_pretrained(save_path)
            model.eval()
            model.to(device)
            models.append(model)
        id2label = models[0].config.id2label

    metric = evaluate.load("poseval")
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=SignalLabelDataset.get_collate_fn())

    signals_pred = []
    signals_gold = []
    for batch in tqdm(test_dataloader, total=len(test_dataset) // batch_size, mininterval=5):
        batch = {k: v.to(device) for k, v in batch.items()}
        preds = compute_ensemble_prediction(models, batch)
        predictions = []
        references = []
        for pred, ref in zip(preds.tolist(), batch['labels'].tolist()):
            pred = [id2label[p] for i, p in enumerate(pred) if ref[i] != -100]
            ref = [id2label[i] for i in ref if i != -100]
            assert len(pred) == len(ref), f"PRED: {pred}, REF {ref}"
            predictions.append(pred)
            references.append(ref)
            signals_pred.append([(relation_type, [i for p, i in signal], None) for signal in decode_labels(pred, pred)])
            signals_gold.append([(relation_type, [i for p, i in signal], None) for signal in decode_labels(ref, ref)])
        metric.add_batch(predictions=predictions, references=references)
    results = metric.compute()

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

    # print(f"==>"
    #       f"  {results['partial-match']['precision']}"
    #       f"  {results['partial-match']['recall']}"
    #       f"  {results['partial-match']['f1-score']}"
    #       f"  {results['exact-match']['precision']}"
    #       f"  {results['exact-match']['recall']}"
    #       f"  {results['exact-match']['f1-score']}"
    #       )

    for key, vals in results.items():
        if key == 'accuracy':
            print(f"{key:10}  {vals * 100:02.2f}")
        else:
            print(
                f"{key:10}  {vals['precision'] * 100:02.2f}  {vals['recall'] * 100:02.2f}  {vals['f1-score'] * 100:02.2f}  {vals['support']}")


if __name__ == '__main__':
    main()
