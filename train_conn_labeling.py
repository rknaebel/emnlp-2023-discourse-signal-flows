import os
import sys
from pathlib import Path

import click
import evaluate
import numpy as np
import sklearn
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_scheduler

from helpers.data import get_corpus_path, load_docs
from helpers.labeling import SignalLabelDataset, RobertaLabelModel


def compute_loss(num_labels, logits, labels, device, majority_class_weight=1.0):
    weights = [majority_class_weight] + ([1.0] * (num_labels - 1))
    loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device), ignore_index=-100)
    loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
    return loss


@click.command()
@click.argument('corpus')
@click.argument('relation-type')
@click.option('-b', '--batch-size', type=int, default=8)
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
        train_docs, test_docs = sklearn.model_selection.train_test_split(train_docs, test_size=0.1,
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
    # print('COLLATE:', batch['input_ids'][0])
    # print('COLLATE:', batch['labels'][0])
    # print('TOKENIZER', SignalLabelDataset.tokenizer.bos_token_id, SignalLabelDataset.tokenizer.eos_token_id,
    #       SignalLabelDataset.tokenizer.sep_token_id, SignalLabelDataset.tokenizer.pad_token_id)
    # print(SignalLabelDataset.tokenizer.special_tokens_map)
    print('LABELS:', train_dataset.labels)
    print('LABEL COUNTS:', train_dataset.get_label_counts())
    # train_dataset = conn_dataset
    # if test_set:
    #     dataset_length = len(train_dataset)
    #     train_size = int(dataset_length * 0.9)
    #     test_size = dataset_length - train_size
    #     train_dataset, test_dataset = random_split(conn_dataset, [train_size, test_size],
    #                                                generator=torch.Generator().manual_seed(random_seed))
    # dataset_length = len(train_dataset)
    # train_size = int(dataset_length * split_ratio)
    # valid_size = dataset_length - train_size
    # train_dataset, valid_dataset = random_split(train_dataset, [train_size, valid_size])
    print(len(train_dataset), len(valid_dataset))

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size,
                                  collate_fn=SignalLabelDataset.get_collate_fn())
    eval_dataloader = DataLoader(valid_dataset, batch_size=batch_size,
                                 collate_fn=SignalLabelDataset.get_collate_fn())

    label2id = train_dataset.labels
    id2label = {v: k for k, v in label2id.items()}

    model = RobertaLabelModel.from_pretrained("roberta-base",
                                              num_labels=train_dataset.get_num_labels(),
                                              id2label=id2label, label2id=label2id,
                                              local_files_only=True,
                                              hidden_dropout_prob=0.3
                                              )
    for param in model.roberta.embeddings.parameters():
        param.requires_grad = False
    for layer in model.roberta.encoder.layer[:10]:
        for param in layer.parameters():
            param.requires_grad = False
    optimizer = AdamW(model.parameters(), lr=5e-5)

    num_epochs = 25
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=50, num_training_steps=num_training_steps
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    best_score = 0.0
    epochs_no_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        losses = []
        for batch_i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), mininterval=5,
                                   desc='Training'):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = compute_loss(train_dataset.get_num_labels(), outputs.logits, batch['labels'], device,
                                majority_class_weight)
            loss.backward()
            losses.append(loss.item())

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        loss_train = np.mean(losses)

        metric = evaluate.load("poseval")
        model.eval()
        losses = []
        for batch in tqdm(eval_dataloader, total=len(eval_dataloader), mininterval=5, desc='Eval'):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
                loss = compute_loss(train_dataset.get_num_labels(), outputs.logits, batch['labels'], device,
                                    majority_class_weight)
                losses.append(loss.item())

            preds = torch.argmax(outputs.logits, dim=-1)
            predictions = []
            references = []
            for pred, ref in zip(preds.tolist(), batch['labels'].tolist()):
                pred = [id2label[p] for i, p in enumerate(pred) if ref[i] != -100]
                ref = [id2label[i] for i in ref if i != -100]
                assert len(pred) == len(ref), f"PRED: {pred}, REF {ref}"
                predictions.append(pred)
                references.append(ref)
            metric.add_batch(predictions=predictions, references=references)
        loss_valid = np.mean(losses)

        print(f"\n== Validation #{epoch}")
        results = metric.compute(zero_division=0)
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
            if save_path:
                model_save_path = os.path.join(save_path, f"best_model_{relation_type.lower()}_label")
                model.save_pretrained(model_save_path)
            epochs_no_improvement = 0
        else:
            epochs_no_improvement += 1
            if epochs_no_improvement > 3:
                print('Early stopping...')
                break


if __name__ == '__main__':
    main()
