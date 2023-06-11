import os
import sys
from pathlib import Path

import click
import evaluate
import numpy as np
import sklearn
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_scheduler

from helpers.crf import SignalLabelDataset, RobertaWithCRF
from helpers.data import get_corpus_path, load_docs
from helpers.evaluate import score_paragraphs
from helpers.labeling import decode_labels


@click.command()
@click.argument('corpus')
@click.argument('relation-type')
@click.option('-b', '--batch-size', type=int, default=16)
@click.option('--split-ratio', type=float, default=0.9)
@click.option('--save-path', default="")
@click.option('--test-set', is_flag=True)
@click.option('--random-seed', default=42, type=int)
@click.option('--majority-class-weight', type=float, default=1.0)
@click.option('--sample-ratio', default=0.0, type=float)
@click.option('-r', '--replace', is_flag=True)
def main(corpus, relation_type, batch_size, split_ratio, save_path, test_set, random_seed, majority_class_weight,
         sample_ratio, replace):
    if save_path:
        save_path = Path(save_path)
        if save_path.is_dir() and (save_path / f"best_model_{relation_type}_label").exists() and not replace:
            print('LabelModel already exists: Exit without writing.', file=sys.stderr)
            return
    corpus_path = get_corpus_path(corpus)
    train_docs = list(load_docs(corpus_path))
    if test_set:
        train_docs, _ = sklearn.model_selection.train_test_split(train_docs, test_size=0.1,
                                                                 random_state=random_seed)
    train_docs, valid_docs = sklearn.model_selection.train_test_split(train_docs, train_size=split_ratio,
                                                                      random_state=random_seed)
    train_dataset = SignalLabelDataset(train_docs, relation_type=relation_type,
                                       filter_empty_paragraphs=sample_ratio > 0.0,
                                       filter_ratio=sample_ratio)
    valid_dataset = SignalLabelDataset(valid_docs, relation_type=relation_type,
                                       filter_empty_paragraphs=sample_ratio > 0.0,
                                       filter_ratio=sample_ratio)
    print('SAMPLE', len(train_dataset), train_dataset[0])
    batch = SignalLabelDataset.get_collate_fn()(train_dataset[:4])
    print('COLLATE:', batch)
    print('LABELS:', train_dataset.labels)
    print('LABEL COUNTS:', train_dataset.get_label_counts())
    print(len(train_dataset), len(valid_dataset))

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size,
                                  collate_fn=SignalLabelDataset.get_collate_fn())
    eval_dataloader = DataLoader(valid_dataset, batch_size=batch_size,
                                 collate_fn=SignalLabelDataset.get_collate_fn())

    label2id = train_dataset.labels
    id2label = {v: k for k, v in label2id.items()}

    model: RobertaWithCRF = RobertaWithCRF.from_pretrained("roberta-base",
                                                           num_labels=train_dataset.get_num_labels(),
                                                           id2label=id2label, label2id=label2id,
                                                           local_files_only=True,
                                                           )

    optimizer = AdamW([
        {'params': model.roberta.parameters(), 'lr': 5e-5},
        {'params': model.classifier.parameters()}
    ], lr=1e-3)

    num_epochs = 10
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=50, num_training_steps=num_training_steps
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    best_score = 0.0
    epochs_no_improvement = 0
    best_model_state = model.state_dict()

    for epoch in range(num_epochs):
        model.train()
        losses = []
        for batch_i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), mininterval=5,
                                   desc='Training'):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, logits = model.emissions(**batch, majority_class_weight=majority_class_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            losses.append(loss.item())

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        loss_train = np.mean(losses)

        metric = evaluate.load("poseval")
        model.eval()
        losses = []
        signals_pred = []
        signals_gold = []

        for batch in tqdm(eval_dataloader, total=len(eval_dataloader), mininterval=5, desc='Eval'):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                loss, logits = model.emissions(**batch, majority_class_weight=majority_class_weight)
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
                signals_pred.append(
                    [(relation_type, [i for p, i in signal], None) for signal in decode_labels(pred, pred)])
                signals_gold.append(
                    [(relation_type, [i for p, i in signal], None) for signal in decode_labels(ref, ref)])
            metric.add_batch(predictions=predictions, references=references)
        loss_valid = np.mean(losses)

        print(f"\n== Validation #{epoch}")
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

        print(f'Training loss: {loss_train}')
        print(f'Validation loss: {loss_valid}')
        current_score = results['macro avg']['f1-score']
        if current_score > best_score:
            best_score = current_score
            print(f"Store new best model! Score: {current_score}...")
            model_save_path = os.path.join(save_path, f"best_model_{relation_type.lower()}_label")
            model.save_pretrained(model_save_path)
            best_model_state = model.state_dict()
            epochs_no_improvement = 0
        else:
            epochs_no_improvement += 1
            if epochs_no_improvement > 3:
                print('Early stopping...')
                break

    model.load_state_dict(best_model_state)
    model.freeze(emissions=False)
    optimizer = AdamW([
        {'params': model.classifier.parameters(), 'lr': 1e-5},
        {'params': model.linear_crf.parameters()}
    ], lr=1e-4)
    num_epochs = 10
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=50, num_training_steps=num_training_steps
    )

    best_score = 0.0
    epochs_no_improvement = 0

    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        losses = []
        for batch_i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), mininterval=5,
                                   desc='Training'):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch)
            loss.backward()
            losses.append(loss.item())

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        loss_train = np.mean(losses)
        print(f'Training loss: {loss_train}')

        metric = evaluate.load("poseval")
        model.eval()
        losses = []
        signals_pred = []
        signals_gold = []
        for batch in tqdm(eval_dataloader, total=len(eval_dataloader), mininterval=5, desc='Eval'):
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
                signals_pred.append(
                    [(relation_type, [i for p, i in signal], None) for signal in decode_labels(pred, pred)])
                signals_gold.append(
                    [(relation_type, [i for p, i in signal], None) for signal in decode_labels(ref, ref)])
            metric.add_batch(predictions=predictions, references=references)
        loss_valid = np.mean(losses)

        print(f"\n== Validation #{epoch}")
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

        print(f'Validation loss: {loss_valid}')
        current_score = results['macro avg']['f1-score']
        if current_score > best_score:
            best_score = current_score
            print(f"New best model! Score: {current_score}...")
            if save_path:
                model_save_path = os.path.join(save_path, f"best_model_{relation_type.lower()}_crf")
                model.save_pretrained(model_save_path)
            epochs_no_improvement = 0
        else:
            epochs_no_improvement += 1
            if epochs_no_improvement > 3:
                print('Early stopping...')
                break


if __name__ == '__main__':
    main()
