import os

import click
import evaluate
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm
from transformers import get_scheduler

from helpers.data import get_corpus_path
from helpers.senses import ConnSenseDataset, NeuralNetwork


@click.command()
@click.argument('corpus')
@click.argument('relation-type')
@click.option('-b', '--batch-size', type=int, default=8)
@click.option('--split-ratio', type=float, default=0.8)
@click.option('--bert-model', default="roberta-base")
@click.option('--save-path', default=".")
def main(corpus, relation_type, batch_size, split_ratio, bert_model, save_path):
    corpus_path = get_corpus_path(corpus)
    cache_path = f'/cache/discourse/{corpus}.en.v3.roberta.joblib'

    conn_dataset = ConnSenseDataset(corpus_path, bert_model, cache_path, relation_type=relation_type)
    print('SAMPLE', len(conn_dataset), conn_dataset[0])
    print('LABELS:', conn_dataset.labels_coarse)
    print('LABELS:', conn_dataset.labels_fine)
    print('LABEL COUNTS:', conn_dataset.get_label_counts())
    dataset_length = len(conn_dataset)
    train_size = int(dataset_length * split_ratio)
    valid_size = dataset_length - train_size
    train_dataset, valid_dataset = random_split(conn_dataset, [train_size, valid_size])
    print(len(train_dataset), len(valid_dataset))
    print('input-dim', len(train_dataset[0]['input']))

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size,
                                  collate_fn=conn_dataset.get_collate_fn())
    eval_dataloader = DataLoader(valid_dataset, batch_size=batch_size,
                                 collate_fn=conn_dataset.get_collate_fn())

    label2id_coarse = conn_dataset.labels_coarse
    id2label_coarse = {v: k for k, v in label2id_coarse.items()}

    label2id_fine = conn_dataset.labels_fine
    id2label_fine = {v: k for k, v in label2id_fine.items()}

    model = NeuralNetwork(len(train_dataset[0]['input']),
                          conn_dataset.get_num_labels_coarse(), conn_dataset.get_num_labels_fine())
    optimizer = AdamW(model.parameters(), lr=5e-5)
    ce_loss_fn = nn.CrossEntropyLoss()

    num_epochs = 30
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=50, num_training_steps=num_training_steps
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    progress_bar = tqdm(range(num_training_steps))

    best_score = 0.0
    epochs_no_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        for batch_i, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            logits_coarse, logits_fine = model(batch['inputs'])
            loss_coarse = ce_loss_fn(logits_coarse, batch['labels_coarse'])
            loss_fine = ce_loss_fn(logits_fine, batch['labels_fine'])
            loss = loss_coarse + loss_fine
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        metric_coarse = evaluate.load("poseval")
        metric_fine = evaluate.load("poseval")
        model.eval()
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                logits_coarse, logits_fine = model(batch['inputs'])
                y_pred_coarse = nn.Softmax(dim=1)(logits_coarse).argmax(-1)
                y_pred_fine = nn.Softmax(dim=1)(logits_fine).argmax(-1)
            predictions = [id2label_coarse[i] for i in y_pred_coarse.tolist()]
            references = [id2label_coarse[i] for i in batch['labels_coarse'].tolist()]
            metric_coarse.add_batch(predictions=[predictions], references=[references])
            predictions = [id2label_fine[i] for i in y_pred_fine.tolist()]
            references = [id2label_fine[i] for i in batch['labels_fine'].tolist()]
            metric_fine.add_batch(predictions=[predictions], references=[references])

        results = metric_coarse.compute()
        for key, vals in results.items():
            if key == 'accuracy':
                print(f"{key:32}  {vals * 100:02.2f}")
            else:
                print(
                    f"{key:32}  {vals['precision'] * 100:02.2f}  {vals['recall'] * 100:02.2f}  {vals['f1-score'] * 100:02.2f}  {vals['support']}")

        results = metric_fine.compute()
        for key, vals in results.items():
            if key == 'accuracy':
                print(f"{key:32}  {vals * 100:02.2f}")
            else:
                print(
                    f"{key:32}  {vals['precision'] * 100:02.2f}  {vals['recall'] * 100:02.2f}  {vals['f1-score'] * 100:02.2f}  {vals['support']}")

        current_score = results['macro avg']['f1-score']
        if current_score > best_score:
            print("Store new best model!")
            best_score = current_score
            os.makedirs(save_path, exist_ok=True)
            model_state = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "sched": lr_scheduler.state_dict(),
                "score": current_score,
                "config": model.config,
                "vocab_coarse": label2id_coarse,
                "vocab_fine": label2id_fine,
            }
            torch.save(model_state, os.path.join(save_path,
                                                 f"best_model_{relation_type.lower()}_sense.pt"))
            epochs_no_improvement = 0
        else:
            epochs_no_improvement += 1
            if epochs_no_improvement > 7:
                print('Early stopping...')
                break


if __name__ == '__main__':
    main()
