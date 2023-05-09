from collections import Counter, defaultdict

import torch
from discopy_data.data.relation import Relation
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from helpers.data import load_docs, iter_document_paragraphs


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


class ConnDataset(Dataset):
    tokenizer = AutoTokenizer.from_pretrained("roberta-base", add_prefix_space=True)

    def __init__(self, data_file, relation_type='explicit'):
        self.items = []
        self.labels = {'O': 0}

        for doc_i, doc in enumerate(load_docs(data_file)):
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
            batch = tokenize_and_align_labels(ConnDataset.tokenizer, examples)
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
