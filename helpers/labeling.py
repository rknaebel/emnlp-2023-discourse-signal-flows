import glob
import itertools
import os
import random
import sys
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F
from discopy_data.data.relation import Relation
from torch import nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel

from helpers.data import iter_document_paragraphs


def tokenize_and_align_labels(tokenizer, examples):
    tokenized_inputs = tokenizer(examples["tokens"],
                                 is_split_into_words=True, padding="max_length", return_tensors='pt')
    labels = []
    for i, label in enumerate(examples["tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def correct_connecting_signals(relations):
    assoc = defaultdict(list)
    result = []
    for rel_i, rel in enumerate(relations):
        for tok in rel.conn.tokens:
            assoc[tok.idx].append(rel_i)
    combined = set(tuple(rs) for rs in assoc.values() if len(rs) == 2)
    assoc = {min(c): c for c in combined}
    combined = {c for cs in combined for c in cs}
    for rel_i, rel in enumerate(relations):
        if rel_i in combined:
            if rel_i in assoc:
                rel_1 = relations[assoc[rel_i][0]]
                rel_2 = relations[assoc[rel_i][1]]
                if tuple(rel_1.senses) == tuple(rel_2.senses):
                    # combine
                    r = Relation(senses=rel_1.senses, type=rel_1.type)
                    r.conn = (rel_1.conn or rel_2.conn)
                    result.append(r)
                else:
                    # TODO: split
                    pass
        else:
            result.append(rel)
    return result


def get_label_mapping(relations):
    token_tag_map = {}
    for r in relations:
        for t_i, t in enumerate(r.conn.tokens):
            if t.idx not in token_tag_map:
                if len(r.conn.tokens) == 1:
                    label = 'S'
                else:
                    if t_i == 0:
                        label = 'B'
                    elif t_i == (len(r.conn.tokens) - 1):
                        label = 'E'
                    else:
                        label = 'I'
                token_tag_map[t.idx] = label
    return token_tag_map


class SignalLabelDataset(Dataset):
    tokenizer = AutoTokenizer.from_pretrained("roberta-base", add_prefix_space=True, local_files_only=True)

    def __init__(self, docs, relation_type='explicit', filter_empty_paragraphs=False, filter_ratio=1.0):
        self.items = []
        self.labels = {
            'O': 0,
            'B': 1,
            'I': 2,
            'E': 3,
            'S': 4,
        }

        for doc_i, doc in enumerate(docs):
            if relation_type.lower() == 'explicit':
                relations = [r for r in doc.relations if r.type.lower() == 'explicit']
            else:
                relations = [r for r in doc.relations if r.type.lower() == 'altlex']
            relations = correct_connecting_signals(relations)
            label_mapping = get_label_mapping(relations)
            for p_i, paragraph in enumerate(iter_document_paragraphs(doc)):
                tokens = []
                labels = []
                for sent in paragraph:
                    for tok in sent.tokens:
                        tokens.append(tok.surface)
                        label = label_mapping.get(tok.idx, 'O')
                        if label in self.labels:
                            label_id = self.labels[label]
                        else:
                            label_id = len(self.labels)
                            self.labels[label] = label_id
                        labels.append(label_id)
                if len(tokens) <= 2:
                    continue
                if filter_empty_paragraphs and len(set(labels)) == 1 and random.random() < filter_ratio:
                    continue
                self.items.append({
                    'id': f"{doc_i}-{p_i}",
                    'tokens': tokens,
                    'tags': labels
                })

    def get_num_labels(self):
        return len(self.labels)

    def get_label_counts(self):
        return Counter(t for i in self.items for t in i['tags'])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

    @staticmethod
    def get_collate_fn():
        def collate(examples):
            examples = {
                'tokens': [example['tokens'] for example in examples],
                'tags': [example['tags'] for example in examples],
            }
            batch = tokenize_and_align_labels(SignalLabelDataset.tokenizer, examples)
            batch['labels'] = torch.LongTensor(batch['labels'])
            return batch

        return collate


def decode_labels(labels, probs):
    conns = []
    for tok_i, (label, prob) in enumerate(zip(labels, probs)):
        if label.startswith('S'):
            conns.append([(prob, tok_i)])
    conn_cur = []
    conn_stack = []
    for tok_i, (label, prob) in enumerate(zip(labels, probs)):
        if label.startswith('B'):
            if conn_cur:
                conn_stack.append(conn_cur)
                conn_cur = []
            conn_cur.append((prob, tok_i))
        elif label.startswith('E'):
            conn_cur.append((prob, tok_i))
            conns.append(conn_cur)
            if conn_stack:
                conn_cur = conn_stack.pop()
            else:
                conn_cur = []
        elif label.startswith('I'):
            conn_cur.append((prob, tok_i))
        else:
            pass
    return conns


class DiscourseSignalExtractor:
    def __init__(self, tokenizer, signal_models, relation_type, device='cpu'):
        self.tokenizer = tokenizer
        self.signal_models = signal_models
        self.relation_type = relation_type
        self.device = device
        self.id2label = signal_models[0].config.id2label

    @staticmethod
    def load_model(save_path, relation_type, device='cpu'):
        save_paths = glob.glob(save_path)
        print(f"Load models: {save_paths}", file=sys.stderr)
        tokenizer = AutoTokenizer.from_pretrained("roberta-base", add_prefix_space=True, local_files_only=True)
        signal_models = []
        for save_path in save_paths:
            label_save_path = os.path.join(save_path, f"best_model_{relation_type.lower()}_label")
            model = AutoModelForTokenClassification.from_pretrained(label_save_path, local_files_only=True)
            model.eval()
            model.to(device)
            signal_models.append(model)
        return DiscourseSignalExtractor(tokenizer, signal_models, relation_type, device)

    def predict(self, doc, batch_size=16):
        document_signals = []
        iter_filtered_paragraphs = filter(lambda p: sum(len(s.tokens) for s in p[1]) >= 7,
                                          enumerate(iter_document_paragraphs(doc)))
        while True:
            par_batch = []
            par_tokens = []
            par_idx = []
            for par_i, paragraph in itertools.islice(iter_filtered_paragraphs, batch_size):
                par_idx.append(par_i)
                tokens = [t for s in paragraph for t in s.tokens]
                par_tokens.append(tokens)
                par_batch.append([t.surface for t in tokens])
            if not par_batch:
                break

            inputs = self.tokenizer(par_batch, truncation=True, is_split_into_words=True,
                                    padding="max_length", return_tensors='pt')
            probs, predictions = self.compute_ensemble_prediction(inputs)
            for b_i, (par_i, tokens, pred, prob) in enumerate(zip(par_idx, par_tokens,
                                                                  predictions.tolist(), probs.tolist())):
                word_ids = inputs.word_ids(b_i)
                predicted_token_class = [self.id2label[t] for t in pred]
                predicted_token_prob = prob
                word_id_map = []
                for i, wi in enumerate(word_ids):
                    if wi is not None and (len(word_id_map) == 0 or (word_ids[i - 1] != wi)):
                        word_id_map.append(i)

                signals = decode_labels([predicted_token_class[i] for i in word_id_map],
                                        [predicted_token_prob[i] for i in word_id_map])
                signals = [[tokens[i] for p, i in signal] for signal in signals]
                relations = [{
                    'tokens_idx': [t.idx for t in signal],
                    'tokens': [t.surface for t in signal],
                    'relation_type': self.relation_type,
                } for signal in signals]

                document_signals.append({
                    'doc_id': doc.doc_id,
                    'paragraph_idx': par_i,
                    'tokens_idx': [t.idx for t in tokens],
                    'tokens': [t.surface for t in tokens],
                    'labels': [predicted_token_class[i] for i in word_id_map],
                    'probs': [round(predicted_token_prob[i], 4) for i in word_id_map],
                    'relations': relations,
                })
        return document_signals

    def predict_paragraphs(self, paragraphs):
        par_batch = []
        par_tokens = []
        for paragraph in paragraphs:
            tokens = [t for s in paragraph['sentences'] for t in s.tokens]
            par_tokens.append(tokens)
            par_batch.append([t.surface for t in tokens])

        inputs = self.tokenizer(par_batch, truncation=True, is_split_into_words=True,
                                padding="max_length", return_tensors='pt')
        probs, predictions = self.compute_ensemble_prediction(inputs)
        for b_i, (tokens, pred, prob) in enumerate(zip(par_tokens, predictions.tolist(), probs.tolist())):
            par = paragraphs[b_i]
            word_ids = inputs.word_ids(b_i)
            predicted_token_class = [self.id2label[t] for t in pred]
            predicted_token_prob = prob
            word_id_map = []
            for i, wi in enumerate(word_ids):
                if wi is not None and (len(word_id_map) == 0 or (word_ids[i - 1] != wi)):
                    word_id_map.append(i)

            signals = decode_labels([predicted_token_class[i] for i in word_id_map],
                                    [predicted_token_prob[i] for i in word_id_map])
            signals = [[tokens[i] for p, i in signal] for signal in signals]
            relations = [{
                'tokens_idx': [t.idx for t in signal],
                'tokens': [t.surface for t in signal],
            } for signal in signals]

            yield {
                'doc_id': par['doc_id'],
                'paragraph_idx': par['paragraph_idx'],
                'tokens_idx': [t.idx for t in tokens],
                'tokens': [t.surface for t in tokens],
                'labels': [predicted_token_class[i] for i in word_id_map],
                'probs': [round(predicted_token_prob[i], 4) for i in word_id_map],
                'relations': relations,
            }

    def compute_ensemble_prediction(self, batch):
        batch = {k: v.to(self.device) for k, v in batch.items()}
        predictions = []
        with torch.no_grad():
            for model in self.signal_models:
                outputs = model(**batch)
                predictions.append(F.softmax(outputs.logits, dim=-1))
        return torch.max(torch.mean(torch.stack(predictions), dim=0), dim=-1)


class RobertaLabelModel(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        return TokenClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
