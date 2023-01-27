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
@click.option('-s', '--sense-level', type=int, default=1)
@click.option('--split-ratio', type=float, default=0.8)
@click.option('--bert-model', default="roberta-base")
@click.option('--save-path', default=".")
def main(corpus, relation_type, batch_size, sense_level, split_ratio, bert_model, save_path):
    corpus_path = get_corpus_path(corpus)
    cache_path = f'/cache/discourse/{corpus}.en.v3.roberta.joblib'

    conn_dataset = ConnSenseDataset(corpus_path, bert_model, cache_path,
                                    relation_type=relation_type, relation_sense_level=sense_level)
    print('SAMPLE', len(conn_dataset), conn_dataset[0])
    print('LABELS:', conn_dataset.labels)
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

    label2id = conn_dataset.labels
    id2label = {v: k for k, v in label2id.items()}

    model = NeuralNetwork(len(train_dataset[0]['input']), conn_dataset.get_num_labels())
    optimizer = AdamW(model.parameters(), lr=5e-5)
    ce_loss_fn = nn.CrossEntropyLoss()

    num_epochs = 10
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=100, num_training_steps=num_training_steps
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    progress_bar = tqdm(range(num_training_steps))

    best_score = 0.0

    for epoch in range(num_epochs):
        model.train()
        for batch_i, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch['inputs'])
            loss = ce_loss_fn(logits, batch['labels'])
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        metric = evaluate.load("poseval")
        model.eval()
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                logits = model(batch['inputs'])
                pred_probab = nn.Softmax(dim=1)(logits)
                y_pred = pred_probab.argmax(-1)
            predictions = [id2label[i] for i in y_pred.tolist()]
            references = [id2label[i] for i in batch['labels'].tolist()]
            metric.add_batch(predictions=[predictions], references=[references])

        results = metric.compute()
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
                "vocab": label2id
            }
            torch.save(model_state, os.path.join(save_path,
                                                 f"best_model_{relation_type.lower()}_lvl{sense_level}_sense.pt"))


if __name__ == '__main__':
    main()
