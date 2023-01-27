import os
from collections import Counter

import click
import evaluate
import joblib
import numpy as np
import torch
from examples.run_bert_pipeline import load_docs
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm.auto import tqdm
from transformers import get_scheduler

from helpers.data import get_sense, get_corpus_path


# simple_map = {
#     "''": '"',
#     "``": '"',
#     "-LRB-": "(",
#     "-RRB-": ")",
#     "-LCB-": "{",
#     "-RCB-": "}",
#     "n't": "not"
# }
#
# def get_doc_sentence_embeddings(sentences, tokenizer, model, last_hidden_only=False):
#     tokens = [[simple_map.get(t.surface, t.surface) for t in sent.tokens] for sent in sentences]
#     subtokens = [[tokenizer.tokenize(t) for t in sent] for sent in tokens]
#     lengths = [[len(t) for t in s] for s in subtokens]
#     inputs = tokenizer(tokens, padding=True, return_tensors='tf', is_split_into_words=True)
#     outputs = model(inputs, output_hidden_states=True)
#     if last_hidden_only:
#         hidden_state = outputs.hidden_states[-2].numpy()
#     else:
#         hidden_state = np.concatenate(outputs.hidden_states[-4:], axis=-1)
#     embeddings = np.zeros((sum(len(s) for s in tokens), hidden_state.shape[-1]), np.float32)
#     e_i = 0
#     for sent_i, _ in enumerate(inputs['input_ids']):
#         len_left = 1
#         for length in lengths[sent_i]:
#             embeddings[e_i] = hidden_state[sent_i][len_left]
#             len_left += length
#             e_i += 1
#     return embeddings


def get_bert_features(idxs, doc_bert, used_context=0):
    idxs = list(idxs)
    pad = np.zeros_like(doc_bert[0])
    embd = doc_bert[idxs].mean(axis=0)
    if used_context > 0:
        left = [doc_bert[i] if i >= 0 else pad for i in range(min(idxs) - used_context, min(idxs))]
        right = [doc_bert[i] if i < len(doc_bert) else pad for i in range(max(idxs) + 1, max(idxs) + 1 + used_context)]
        embd = np.concatenate(left + [embd] + right).flatten()
    return embd


class ConnSenseDataset(Dataset):

    def __init__(self, data_file, bert_model, cache_dir, relation_type='explicit', relation_sense_level=1):
        self.items = []
        self.labels = {}
        self.bert_model = bert_model
        # self.tokenizer = AutoTokenizer.from_pretrained(bert_model, add_prefix_space=True)
        # self.model = AutoModel.from_pretrained(bert_model)

        doc_embeddings = joblib.load(cache_dir)
        for doc_i, doc in enumerate(load_docs(data_file)):
            doc_embedding = doc_embeddings[doc.doc_id]
            for sent_i, sent in enumerate(doc.sentences):
                token_offset = sent.tokens[0].idx
                embeddings = doc_embedding[token_offset:token_offset + len(sent.tokens)]
                sent.embeddings = embeddings
            doc_bert = doc.get_embeddings()
            for r_i, r in enumerate(doc.relations):
                if r.type.lower() != relation_type.lower():
                    continue
                conn_idx = (t.idx for t in r.conn.tokens)
                features = get_bert_features(conn_idx, doc_bert, used_context=0)
                label = get_sense(r.senses[0], relation_sense_level)
                if label in self.labels:
                    label_id = self.labels[label]
                else:
                    label_id = len(self.labels)
                    self.labels[label] = label_id
                self.items.append({
                    'id': f"{doc_i}-{r_i}",
                    'input': torch.from_numpy(features),
                    'label': label_id,
                })

    def get_num_labels(self):
        return len(self.labels)

    def get_label_counts(self):
        return Counter(i['label'] for i in self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

    def get_collate_fn(self):
        def collate(examples):
            batch = {
                'inputs': torch.stack([example['input'] for example in examples]),
                'labels': torch.LongTensor([example['label'] for example in examples]),
            }
            return batch

        return collate


class NeuralNetwork(nn.Module):
    def __init__(self, in_size, out_size):
        super(NeuralNetwork, self).__init__()
        self.config = {
            'in_size': in_size,
            'out_size': out_size
        }
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, out_size),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


@click.command()
@click.argument('corpus')
@click.argument('relation-type')
@click.option('-b', '--batch-size', type=int, default=8)
@click.option('-s', '--sense-level', type=int, default=1)
@click.option('--split-ratio', type=float, default=0.8)
@click.option('--bert-model', default="roberta-base")
@click.option('--save-path', default="model_sense")
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
            model_state = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "sched": lr_scheduler.state_dict(),
                "score": current_score,
                "config": model.config,
                "vocab": label2id
            }
            torch.save(model_state, os.path.join(save_path, f"best_model_conn_sense.pt"))


if __name__ == '__main__':
    main()
