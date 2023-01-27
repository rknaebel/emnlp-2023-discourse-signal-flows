from collections import Counter

import joblib
import numpy as np
import torch
from examples.run_bert_pipeline import load_docs
from torch import nn
from torch.utils.data import Dataset

from helpers.data import get_sense


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
