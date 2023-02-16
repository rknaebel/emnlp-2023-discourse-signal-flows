import click
import evaluate
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm
from transformers import get_scheduler

from helpers.data import get_corpus_path
from helpers.senses import ConnSenseDataset, DiscourseSenseClassifier


def load_dataset(corpus, bert_model, relation_type, split_ratio, labels_coarse=None, labels_fine=None, test_set=False,
                 random_seed=42):
    corpus_path = get_corpus_path(corpus)
    cache_path = f'/cache/discourse/{corpus}.en.v3.roberta.joblib'

    conn_dataset = ConnSenseDataset(corpus_path, bert_model, cache_path, relation_type=relation_type,
                                    labels_coarse=labels_coarse, labels_fine=labels_fine)
    print('SAMPLE', len(conn_dataset), conn_dataset[0])
    print('LABELS:', conn_dataset.labels_coarse)
    print('LABELS:', conn_dataset.labels_fine)
    print('LABEL COUNTS:', conn_dataset.get_label_counts())
    train_dataset = conn_dataset
    if test_set:
        dataset_length = len(train_dataset)
        train_size = int(dataset_length * 0.9)
        test_size = dataset_length - train_size
        train_dataset, test_dataset = random_split(conn_dataset, [train_size, test_size],
                                                   generator=torch.Generator().manual_seed(random_seed))
    else:
        test_dataset = None

    dataset_length = len(train_dataset)
    train_size = int(dataset_length * split_ratio)
    valid_size = dataset_length - train_size
    train_dataset, valid_dataset = random_split(train_dataset, [train_size, valid_size])
    print(len(train_dataset), len(valid_dataset))
    print('input-dim', len(train_dataset[0]['input']))

    return conn_dataset, train_dataset, valid_dataset, test_dataset


@click.command()
@click.argument('corpus')
@click.argument('relation-type')
@click.option('-b', '--batch-size', type=int, default=8)
@click.option('--split-ratio', type=float, default=0.8)
@click.option('--bert-model', default="roberta-base")
@click.option('--save-path', default=".")
@click.option('--test-set', is_flag=True)
@click.option('--random-seed', default=42, type=int)
def main(corpus, relation_type, batch_size, split_ratio, bert_model, save_path, test_set, random_seed):
    dataset, train_dataset, valid_dataset, _ = load_dataset(corpus, bert_model, relation_type, split_ratio,
                                                            test_set=test_set, random_seed=random_seed)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size,
                                  collate_fn=ConnSenseDataset.get_collate_fn())
    eval_dataloader = DataLoader(valid_dataset, batch_size=batch_size,
                                 collate_fn=ConnSenseDataset.get_collate_fn())

    model = DiscourseSenseClassifier(len(train_dataset[0]['input']),
                                     dataset.labels_coarse, dataset.labels_fine,
                                     relation_type=relation_type)
    id2label_coarse = {v: k for k, v in model.label2id_coarse.items()}
    id2label_fine = {v: k for k, v in model.label2id_fine.items()}

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
        scores = []
        print(f"##\n## EVAL ({epoch}) {relation_type}\n##")
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            output = model.predict(batch['inputs'])
            references = [id2label_coarse[i] for i in batch['labels_coarse'].tolist()]
            metric_coarse.add_batch(predictions=[output['coarse']], references=[references])
            references = [id2label_fine[i] for i in batch['labels_fine'].tolist()]
            metric_fine.add_batch(predictions=[output['fine']], references=[references])

        results = metric_coarse.compute()
        for key, vals in results.items():
            if key == 'accuracy':
                print(f"{key:32}  {vals * 100:02.2f}")
            else:
                print(
                    f"{key:32}  {vals['precision'] * 100:02.2f}  {vals['recall'] * 100:02.2f}  {vals['f1-score'] * 100:02.2f}  {vals['support']}")
        scores.append(results['macro avg']['f1-score'])

        results = metric_fine.compute()
        for key, vals in results.items():
            if key == 'accuracy':
                print(f"{key:32}  {vals * 100:02.2f}")
            else:
                print(
                    f"{key:32}  {vals['precision'] * 100:02.2f}  {vals['recall'] * 100:02.2f}  {vals['f1-score'] * 100:02.2f}  {vals['support']}")
        scores.append(results['macro avg']['f1-score'])

        current_score = np.mean(scores)
        if current_score > best_score:
            print(f"Store new best model! Score: {current_score}...")
            best_score = current_score
            model_state = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "sched": lr_scheduler.state_dict(),
                "score": current_score,
                "config": model.config,
            }
            model.save(save_path, model_state)
            epochs_no_improvement = 0
        else:
            epochs_no_improvement += 1
            if epochs_no_improvement > 7:
                print('Early stopping...')
                break


if __name__ == '__main__':
    main()
