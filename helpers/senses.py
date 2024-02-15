import json
import os
from collections import Counter, defaultdict
from typing import List

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from discopy_data.data.relation import Relation
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from helpers.data import get_sense, get_doc_embeddings


def compute_loss(num_labels, weights, logits, labels, device):
    loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device), label_smoothing=0.1)
    loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
    return loss


def compute_weights(labels_coarse, labels_fine, none_weight=1.0):
    weights_coarse = [none_weight] + [1.0] * (len(labels_coarse) - 1)
    weights_fine = [none_weight] + [1.0] * (len(labels_fine) - 1)
    return weights_coarse, weights_fine


def group_relations_per_document(labels):
    docs = defaultdict(list)
    for label in labels:
        docs[label['doc_id']].append(label)
    return dict(docs)


def get_sense_mapping(docs, simple_sense=False):
    senses = sorted({s for doc in docs for r in doc.relations for s in r.senses})
    labels_coarse = {'None': 0}
    labels_fine = {'None': 0}
    for sense in senses:
        coarse_sense = get_sense(sense, 1, simple_sense)
        if coarse_sense not in labels_coarse:
            labels_coarse[coarse_sense] = len(labels_coarse)
        fine_sense = get_sense(sense, 2, simple_sense)
        if fine_sense not in labels_fine:
            labels_fine[fine_sense] = len(labels_fine)
    return labels_coarse, labels_fine


# def load_dataset(docs, relation_type, split_ratio, labels_coarse=None, labels_fine=None, predictions=None, simple_senses=False):
#     conn_dataset = ConnSenseDataset(docs, relation_type=relation_type,
#                                     labels_coarse=labels_coarse, labels_fine=labels_fine,
#                                     predictions=predictions, simple_sense=simple_senses)
#     print('SAMPLE', len(conn_dataset), conn_dataset[0])
#     print('LABELS:', conn_dataset.labels_coarse)
#     print('LABELS:', conn_dataset.labels_fine)
#     print('LABEL COUNTS:', conn_dataset.get_label_counts())
#     train_dataset = conn_dataset
#     # if test_set:
#     #     dataset_length = len(train_dataset)
#     #     train_size = int(dataset_length * 0.9)
#     #     test_size = dataset_length - train_size
#     #     train_dataset, test_dataset = random_split(conn_dataset, [train_size, test_size],
#     #                                                generator=torch.Generator().manual_seed(random_seed))
#     # else:
#     #     test_dataset = None
#
#     # dataset_length = len(train_dataset)
#     # train_size = int(dataset_length * split_ratio)
#     # valid_size = dataset_length - train_size
#     # train_dataset, valid_dataset = random_split(train_dataset, [train_size, valid_size])
#     print(len(train_dataset), len(valid_dataset))
#     print('input-dim', len(train_dataset[0]['input']))
#
#     return conn_dataset, train_dataset, valid_dataset


def get_bert_features(idxs, doc_bert, used_context=0):
    idxs = list(idxs)
    pad = np.zeros_like(doc_bert[0])
    embd = np.concatenate([doc_bert[idxs].mean(axis=0), doc_bert[idxs].max(axis=0)])
    if used_context > 0:
        left = [doc_bert[i] if i >= 0 else pad for i in range(min(idxs) - used_context, min(idxs))]
        right = [doc_bert[i] if i < len(doc_bert) else pad for i in range(max(idxs) + 1, max(idxs) + 1 + used_context)]
        embd = np.concatenate(left + [embd] + right).flatten()
    return embd


class ConnSenseDataset(Dataset):
    doc_embeddings = None
    cache_path = "/cache/discourse/pdtb3.en.v3.roberta.v2.joblib"

    def __init__(self, docs, relation_type='explicit',
                 labels_coarse=None, labels_fine=None, predictions=None, simple_sense=False, used_context=1):
        self.items = []
        self.labels_coarse = labels_coarse or {}
        self.labels_fine = labels_fine or {}
        self.bert_model = "roberta-base"
        if predictions is not None and isinstance(predictions, str):
            predictions = [json.loads(line) for line in open(predictions)]
            predictions = group_relations_per_document(predictions)

        if not ConnSenseDataset.doc_embeddings:
            print("Load document embeddings")
            if os.path.exists(self.cache_path):
                ConnSenseDataset.doc_embeddings = joblib.load(self.cache_path)
            else:
                tokenizer = AutoTokenizer.from_pretrained("roberta-base", add_prefix_space=True, local_files_only=True)
                model = AutoModel.from_pretrained("roberta-base", local_files_only=True)
                model.eval()
                device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
                model.to(device)
                ConnSenseDataset.doc_embeddings = {}
                for doc_i, doc in enumerate(docs):
                    ConnSenseDataset.doc_embeddings[doc.doc_id] = get_doc_embeddings(doc, tokenizer, model,
                                                                                     last_hidden_only=False,
                                                                                     device=device)
                joblib.dump(ConnSenseDataset.doc_embeddings, self.cache_path)

        for doc_i, doc in tqdm(enumerate(docs), total=len(docs), desc=f'Load docs ({relation_type})'):
            doc_embedding = ConnSenseDataset.doc_embeddings[doc.doc_id]
            for sent_i, sent in enumerate(doc.sentences):
                token_offset = sent.tokens[0].idx
                embeddings = doc_embedding[token_offset:token_offset + len(sent.tokens)]
                sent.embeddings = embeddings
            doc_bert = doc.get_embeddings()
            if predictions and doc.doc_id in predictions:
                # compare document relations with predictions and filter negative samples
                tokens = doc.get_tokens()
                doc_pred = predictions[doc.doc_id]
                # rels_pred = []
                # for p in doc_pred:
                #     for pred in decode_labels(p['labels'], p['probs']):
                #         rels_pred.append(tuple([p['tokens_idx'][idx] for prob, idx in pred]))
                rels_pred = [tuple(r['tokens_idx']) for p in doc_pred for r in p['relations']]
                rels = [tuple([t.idx for t in r.conn.tokens]) for r in doc.relations if
                        r.type.lower() == relation_type.lower()]
                negative_samples = list(set(rels_pred) - set(rels))
                # num_relations = len([r for r in doc.relations if r.type.lower() == 'altlex'])
                for negative_sample in negative_samples:
                    r = Relation(conn=[tokens[i] for i in negative_sample], senses=['None'], type=relation_type)
                    doc.relations.append(r)
                # if num_relations > 0 and len(negative_samples):
                #     k = min(num_relations * 4, len(negative_samples))
                #     for negative_sample in random.sample(negative_samples, k=k):
                #         r = Relation(conn=[tokens[i] for i in negative_sample], senses=['None'], type='AltLex')
                #         doc.relations.append(r)
            for r_i, r in enumerate(doc.relations):
                if not r.type.lower() == relation_type.lower():
                    continue
                conn_idx = (t.idx for t in r.conn.tokens)
                features = get_bert_features(conn_idx, doc_bert, used_context=used_context)
                for sense in r.senses:
                    label_coarse = get_sense(sense, 1, simple_sense)
                    label_id_coarse = self.labels_coarse[label_coarse]
                    label_fine = get_sense(sense, 2, simple_sense)
                    label_id_fine = self.labels_fine[label_fine]
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

    @staticmethod
    def get_collate_fn():
        def collate(examples):
            batch = {
                'inputs': torch.stack([example['input'] for example in examples]),
                'labels_coarse': torch.LongTensor([example['label_coarse'] for example in examples]),
                'labels_fine': torch.LongTensor([example['label_fine'] for example in examples]),
            }
            return batch

        return collate


class DiscourseSenseClassifier(nn.Module):
    def __init__(self, in_size, labels_coarse, labels_fine, relation_type='both', hidden=(2048, 512), drop_rate=0.3,
                 used_context=1):
        super().__init__()
        self.label2id_coarse = labels_coarse
        self.id2label_coarse = {v: k for k, v in self.label2id_coarse.items()}

        self.label2id_fine = labels_fine
        self.id2label_fine = {v: k for k, v in self.label2id_fine.items()}

        self.relation_type = relation_type
        self.used_context = used_context

        self.config = {
            'in_size': in_size,
            'labels_coarse': labels_coarse,
            'labels_fine': labels_fine,
            'hidden': hidden,
            'drop_rate': drop_rate,
            'used_context': used_context,
        }
        self.flatten = nn.Flatten()
        hidden_1, hidden_2 = hidden
        self.linear_relu_stack = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(in_size, hidden_1),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Dropout(drop_rate),
        )
        self.linear_coarse = nn.Linear(hidden_2, len(labels_coarse))
        self.linear_fine = nn.Linear(hidden_2, len(labels_fine))

    @staticmethod
    def load(save_path, device, relation_type='both'):
        sense_save_path = os.path.join(save_path, f"best_model_{relation_type}_sense.pt")
        sense_model_state = torch.load(sense_save_path, map_location=device)
        sense_model = DiscourseSenseClassifier(**sense_model_state['config']).to(device)
        sense_model.load_state_dict(sense_model_state['model'])
        return sense_model

    def save(self, save_path, model_state):
        os.makedirs(save_path, exist_ok=True)
        torch.save(model_state, os.path.join(save_path, f"best_model_{self.relation_type}_sense.pt"))

    def forward(self, x):
        x = self.flatten(x)
        y_inter = self.linear_relu_stack(x)
        logits_coarse = self.linear_coarse(y_inter)
        logits_fine = self.linear_fine(y_inter)
        return logits_coarse, logits_fine

    def predict(self, features, no_none=False):
        with torch.no_grad():
            outputs_coarse, outputs_fine = self(features)
        if no_none:
            outputs_coarse[:, 0] = 0.0
            outputs_fine[:, 0] = 0.0
        sense_probs_coarse, sense_predictions_coarse = F.softmax(outputs_coarse, dim=-1).max(dim=-1)
        sense_probs_fine, sense_predictions_fine = F.softmax(outputs_fine, dim=-1).max(dim=-1)
        return {
            "coarse": [self.id2label_coarse[i] for i in sense_predictions_coarse.tolist()],
            "coarse_logits": outputs_coarse,
            "probs_coarse": sense_probs_coarse.tolist(),
            "fine": [self.id2label_fine[i] for i in sense_predictions_fine.tolist()],
            "fine_logits": outputs_fine,
            "probs_fine": sense_probs_fine.tolist(),
            "probs_coarse_all": F.softmax(outputs_coarse, dim=-1).detach().cpu().numpy(),
        }


class DiscourseSenseEnsembleClassifier(nn.Module):
    def __init__(self, models: List[DiscourseSenseClassifier]):
        super().__init__()
        assert len(models) > 1
        assert all(models[0].label2id_coarse == m.label2id_coarse for m in models[1:])
        assert all(models[0].label2id_fine == m.label2id_fine for m in models[1:])
        self.models = models
        self.id2label_coarse = self.models[0].id2label_coarse
        self.id2label_fine = self.models[0].id2label_fine
        self.label2id_coarse = self.models[0].label2id_coarse
        self.label2id_fine = self.models[0].label2id_fine
        self.used_context = self.models[0].used_context

    @staticmethod
    def load(save_paths, device, relation_type='both'):
        models = []
        for save_path in save_paths:
            models.append(DiscourseSenseClassifier.load(save_path, device, relation_type=relation_type).to(device))
        return DiscourseSenseEnsembleClassifier(models)

    def predict(self, features, no_none=False):
        outputs_coarse = []
        outputs_fine = []
        with torch.no_grad():
            for model in self.models:
                output_coarse, output_fine = model(features)
                if no_none:
                    outputs_coarse[:, 0] = 0.0
                    outputs_fine[:, 0] = 0.0
                outputs_coarse.append(F.softmax(output_coarse, dim=-1))
                outputs_fine.append(F.softmax(output_fine, dim=-1))
        outputs_coarse = torch.mean(torch.stack(outputs_coarse), dim=0)
        outputs_fine = torch.mean(torch.stack(outputs_fine), dim=0)

        sense_probs_coarse, sense_predictions_coarse = outputs_coarse.max(dim=-1)
        sense_probs_fine, sense_predictions_fine = outputs_fine.max(dim=-1)

        return {
            "coarse": [self.id2label_coarse[i] for i in sense_predictions_coarse.tolist()],
            "probs_coarse": sense_probs_coarse.tolist(),
            "fine": [self.id2label_fine[i] for i in sense_predictions_fine.tolist()],
            "probs_fine": sense_probs_fine.tolist(),
            "probs_coarse_all": outputs_coarse.detach().cpu().numpy(),
        }


# class DiscourseSignalDisambiguationModel(nn.Module):
#     def __init__(self, in_size, relation_type='both', hidden=(2048, 512), drop_rate=0.3):
#         super().__init__()
#         self.relation_type = relation_type
#         self.config = {
#             'in_size': in_size,
#             'hidden': hidden,
#             'drop_rate': drop_rate,
#         }
#         self.flatten = nn.Flatten()
#         hidden_1, hidden_2 = hidden
#         self.linear_relu_stack = nn.Sequential(
#             nn.Dropout(drop_rate),
#             nn.Linear(in_size, hidden_1),
#             nn.ReLU(),
#             nn.Dropout(drop_rate),
#             nn.Linear(hidden_1, hidden_2),
#             nn.ReLU(),
#             nn.Dropout(drop_rate),
#         )
#         self.final = nn.Linear(hidden_2, 2)
#
#     @staticmethod
#     def load(save_path, relation_type='both'):
#         sense_save_path = os.path.join(save_path, f"best_model_{relation_type}_dis.pt")
#         sense_model_state = torch.load(sense_save_path)
#         sense_model = DiscourseSignalDisambiguationModel(**sense_model_state['config'])
#         sense_model.load_state_dict(sense_model_state['model'])
#         return sense_model
#
#     def save(self, save_path, model_state):
#         os.makedirs(save_path, exist_ok=True)
#         torch.save(model_state, os.path.join(save_path, f"best_model_{self.relation_type}_dis.pt"))
#
#     def forward(self, x):
#         x = self.flatten(x)
#         y_inter = self.linear_relu_stack(x)
#         logits = self.final(y_inter)
#         return logits
#
#     def predict(self, features):
#         with torch.no_grad():
#             outputs = self(features)
#         probs = F.softmax(outputs, dim=-1).detach().cpu().numpy()
#         return {
#             "is_relation": probs[:, 1].round().astype(int),
#             "is_relation_probs": probs[:, 1],
#             "logits": outputs,
#         }
