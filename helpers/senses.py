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

    def __init__(self, data_file, bert_model, cache_dir, relation_type='explicit'):
        self.items = []
        self.labels_coarse = {}
        self.labels_fine = {}
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
                label_coarse = get_sense(r.senses[0], 1)
                if label_coarse in self.labels_coarse:
                    label_id_coarse = self.labels_coarse[label_coarse]
                else:
                    label_id_coarse = len(self.labels_coarse)
                    self.labels_coarse[label_coarse] = label_id_coarse
                label_fine = get_sense(r.senses[0], 2)
                if label_fine in self.labels_fine:
                    label_id_fine = self.labels_fine[label_fine]
                else:
                    label_id_fine = len(self.labels_fine)
                    self.labels_fine[label_fine] = label_id_fine
                self.items.append({
                    'id': f"{doc_i}-{r_i}",
                    'input': torch.from_numpy(features),
                    'label_coarse': label_id_coarse,
                    'label_fine': label_id_fine,
                })

    def get_num_labels_coarse(self):
        return len(self.labels_coarse)

    def get_num_labels_fine(self):
        return len(self.labels_fine)

    def get_label_counts(self):
        return Counter(i['label_coarse'] for i in self.items), Counter(i['label_fine'] for i in self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

    def get_collate_fn(self):
        def collate(examples):
            batch = {
                'inputs': torch.stack([example['input'] for example in examples]),
                'labels_coarse': torch.LongTensor([example['label_coarse'] for example in examples]),
                'labels_fine': torch.LongTensor([example['label_fine'] for example in examples]),
            }
            return batch

        return collate


class NeuralNetwork(nn.Module):
    def __init__(self, in_size, out_size_coarse, out_size_fine):
        super(NeuralNetwork, self).__init__()
        self.config = {
            'in_size': in_size,
            'out_size_coarse': out_size_coarse,
            'out_size_fine': out_size_fine
        }
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, out_size),
        )
        self.linear_coarse = nn.Linear(256, out_size_coarse)
        self.linear_fine = nn.Linear(256, out_size_fine)

    @staticmethod
    def load(save_path, relation_type):
        sense_save_path = os.path.join(save_path, f"best_model_{relation_type.lower()}_sense.pt")
        sense_model_state = torch.load(sense_save_path)
        sense_model = NeuralNetwork(**sense_model_state['config'])
        sense_model.load_state_dict(sense_model_state['model'])
        label2id_coarse = sense_model_state['vocab_coarse']
        label2id_fine = sense_model_state['vocab_fine']
        return sense_model, label2id_coarse, label2id_fine

    def forward(self, x):
        x = self.flatten(x)
        y_inter = self.linear_relu_stack(x)
        logits_coarse = self.linear_coarse(y_inter)
        logits_fine = self.linear_fine(y_inter)
        return logits_coarse, logits_fine
