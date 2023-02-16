import glob

import click
import evaluate
import torch
from torch.utils.data import DataLoader, random_split

from helpers.data import get_corpus_path
from helpers.senses import ConnSenseDataset, DiscourseSenseClassifier, DiscourseSenseEnsembleClassifier


def load_dataset(corpus, bert_model, relation_type, labels_coarse=None, labels_fine=None, random_seed=42):
    corpus_path = get_corpus_path(corpus)
    cache_path = f'/cache/discourse/{corpus}.en.v3.roberta.joblib'

    conn_dataset = ConnSenseDataset(corpus_path, bert_model, cache_path, relation_type=relation_type,
                                    labels_coarse=labels_coarse, labels_fine=labels_fine)
    train_dataset = conn_dataset
    dataset_length = len(train_dataset)
    train_size = int(dataset_length * 0.9)
    test_size = dataset_length - train_size
    train_dataset, test_dataset = random_split(conn_dataset, [train_size, test_size],
                                               generator=torch.Generator().manual_seed(random_seed))
    return conn_dataset, test_dataset


@click.command()
@click.argument('corpus')
@click.option('-b', '--batch-size', type=int, default=8)
@click.option('--bert-model', default="roberta-base")
@click.option('--save-path', default=".")
@click.option('--random-seed', default=42, type=int)
def main(corpus, batch_size, bert_model, save_path, random_seed):
    dataset, conn_test_dataset = load_dataset(corpus, bert_model, 'explicit', random_seed=random_seed)
    dataset, altlex_test_dataset = load_dataset(corpus, bert_model, 'altlex',
                                                labels_coarse=dataset.labels_coarse, labels_fine=dataset.labels_fine,
                                                random_seed=random_seed)

    eval_conn_dataloader = DataLoader(conn_test_dataset, batch_size=batch_size,
                                      collate_fn=ConnSenseDataset.get_collate_fn())
    eval_altlex_dataloader = DataLoader(altlex_test_dataset, batch_size=batch_size,
                                        collate_fn=ConnSenseDataset.get_collate_fn())

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    save_paths = glob.glob(save_path)
    if len(save_paths) == 1:
        print("Load Single Classifier Model")
        model = DiscourseSenseClassifier.load(save_path)
        model.to(device)
    else:
        print("Load Ensemble Model")
        model = DiscourseSenseEnsembleClassifier.load(save_paths, device)

    metric_coarse = evaluate.load("poseval")
    metric_fine = evaluate.load("poseval")
    model.eval()
    scores = []
    for relation_type, eval_dataloader in [('explicit', eval_conn_dataloader), ('altlex', eval_altlex_dataloader)]:
        print(f"##\n## EVAL {relation_type}\n##")
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            output = model.predict(batch['inputs'])
            references = [model.id2label_coarse[i] for i in batch['labels_coarse'].tolist()]
            metric_coarse.add_batch(predictions=[output['coarse']], references=[references])
            references = [model.id2label_fine[i] for i in batch['labels_fine'].tolist()]
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


if __name__ == '__main__':
    main()
