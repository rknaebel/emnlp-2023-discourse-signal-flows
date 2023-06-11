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

from helpers.data import get_corpus_path, load_docs
from helpers.senses import ConnSenseDataset, DiscourseSenseClassifier, compute_weights, compute_loss, \
    get_sense_mapping
from helpers.stats import print_metrics_results


@click.command()
@click.argument('corpus')
@click.argument('relation-type')
@click.option('--predictions', default=None)
@click.option('-b', '--batch-size', type=int, default=16)
@click.option('--split-ratio', type=float, default=0.9)
@click.option('--save-path', default="")
@click.option('--test-set', is_flag=True)
@click.option('--random-seed', default=42, type=int)
@click.option('--hidden', default="256,64")
@click.option('--drop-rate', default=0.3, type=float)
@click.option('-r', '--replace', is_flag=True)
@click.option('--simple-sense', is_flag=True)
@click.option('--used-context', default=1, type=int)
def main(corpus, relation_type, predictions, batch_size, split_ratio, save_path, test_set, random_seed, hidden,
         drop_rate,
         replace, simple_sense, used_context):
    if save_path:
        save_path = Path(save_path)
        if save_path.is_dir() and (save_path / f"best_model_{relation_type}_sense.pt").exists() and not replace:
            print('SenseModel already exists: Exit without writing.', file=sys.stderr)
            return
    corpus_path = get_corpus_path(corpus)
    train_docs = list(load_docs(corpus_path))
    labels_coarse, labels_fine = get_sense_mapping(train_docs)
    if test_set:
        train_docs, test_docs = sklearn.model_selection.train_test_split(train_docs, test_size=0.1,
                                                                         random_state=random_seed)
    train_docs, valid_docs = sklearn.model_selection.train_test_split(train_docs, train_size=split_ratio,
                                                                      random_state=random_seed)
    train_dataset = ConnSenseDataset(train_docs, relation_type=relation_type,
                                     labels_coarse=labels_coarse, labels_fine=labels_fine,
                                     predictions=predictions, simple_sense=simple_sense, used_context=used_context)
    valid_dataset = ConnSenseDataset(valid_docs, relation_type=relation_type,
                                     labels_coarse=labels_coarse, labels_fine=labels_fine,
                                     predictions=predictions, simple_sense=simple_sense, used_context=used_context)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size,
                                  collate_fn=ConnSenseDataset.get_collate_fn())
    eval_dataloader = DataLoader(valid_dataset, batch_size=batch_size,
                                 collate_fn=ConnSenseDataset.get_collate_fn())

    model = DiscourseSenseClassifier(len(train_dataset[0]['input']),
                                     labels_coarse, labels_fine,
                                     relation_type=relation_type,
                                     hidden=[int(i) for i in hidden.split(',')],
                                     drop_rate=drop_rate,
                                     used_context=used_context)

    optimizer = AdamW(model.parameters(), lr=1e-4)

    num_epochs = 20
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=10, num_training_steps=num_training_steps
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    best_loss = float('inf')
    epochs_no_improvement = 0
    weights_coarse, weights_fine = compute_weights(labels_coarse, labels_fine)

    for epoch in range(num_epochs):
        model.train()
        losses = []
        for batch_i, batch in tqdm(enumerate(train_dataloader), desc='Training',
                                   total=len(train_dataloader), mininterval=5):
            batch = {k: v.to(device) for k, v in batch.items()}
            logits_coarse, logits_fine = model(batch['inputs'])
            loss_coarse = compute_loss(len(labels_coarse), weights_coarse,
                                       logits_coarse, batch['labels_coarse'], device)
            loss_fine = compute_loss(len(labels_fine), weights_fine,
                                     logits_fine, batch['labels_fine'], device)
            loss = loss_coarse + loss_fine * 2.0
            loss.backward()
            losses.append(loss.item())

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        loss_train = np.mean(losses)

        metric_coarse = evaluate.load("poseval")
        metric_fine = evaluate.load("poseval")
        model.eval()
        losses = []
        print(f"##\n## EVAL ({epoch})\n##")
        for batch in tqdm(eval_dataloader, desc='Validation',
                          total=len(eval_dataloader), mininterval=5):
            batch = {k: v.to(device) for k, v in batch.items()}
            output = model.predict(batch['inputs'])
            loss_coarse = compute_loss(len(labels_coarse), weights_coarse,
                                       output['coarse_logits'], batch['labels_coarse'], device)
            loss_fine = compute_loss(len(labels_fine), weights_fine,
                                     output['fine_logits'], batch['labels_fine'], device)
            loss = loss_coarse + loss_fine * 2.0
            losses.append(loss.item())
            references = [model.id2label_coarse[i] for i in batch['labels_coarse'].tolist()]
            metric_coarse.add_batch(predictions=[output['coarse']], references=[references])
            references = [model.id2label_fine[i] for i in batch['labels_fine'].tolist()]
            metric_fine.add_batch(predictions=[output['fine']], references=[references])
        loss_valid = np.mean(losses)

        results_coarse = metric_coarse.compute(zero_division=0)
        print_metrics_results(results_coarse)

        results_fine = metric_fine.compute(zero_division=0)
        print_metrics_results(results_fine)

        print(f'Training loss: {loss_train}')
        print(f'Validation loss: {loss_valid}')
        if best_loss > loss_valid:
            best_loss = loss_valid
            print(f"Store new best model! Loss: {loss_valid}...")
            model_state = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "sched": lr_scheduler.state_dict(),
                "score": best_loss,
                "config": model.config,
            }
            if save_path:
                model.save(save_path, model_state)
            epochs_no_improvement = 0
        else:
            epochs_no_improvement += 1
            if epochs_no_improvement >= 4:
                print('Early stopping...')
                break


if __name__ == '__main__':
    main()
