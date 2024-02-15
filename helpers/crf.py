import glob
import os
import random
import sys
from collections import Counter, defaultdict
from typing import List
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from discopy_data.data.relation import Relation
from torch import BoolTensor
from torch import FloatTensor
from torch import LongTensor
from torch import nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel

from helpers.data import iter_document_paragraphs


def tokenize_and_align_labels(tokenizer, examples):
    tokenized_inputs = tokenizer(examples["tokens"],
                                 is_split_into_words=True, padding="max_length", return_tensors='pt')
    labels = []
    for i, label in enumerate(examples["tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        label_ids = []
        for w_i, word_idx in enumerate(word_ids):  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(7)
            else:
                label_ids.append(label[word_idx])
        label_ids[0] = 5
        label_ids[label_ids.index(7)] = 6
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
            '<START>': 5,
            '<END>': 6,
            '<PAD>': 7,
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


def decode_labels_gold(labels, probs):
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


def decode_labels(labels, probs):
    conns = []
    conn_cur = []
    for tok_i in range(1, len(labels) - 2):
        if labels[tok_i - 1] != 'O' and labels[tok_i + 1] != 'O':
            labels[tok_i] = 'I'
    for tok_i, (label, prob) in enumerate(zip(labels, probs)):
        if label in {'B', 'I', 'E', 'S'}:
            conn_cur.append((prob, tok_i))
        else:
            if conn_cur:
                conns.append(conn_cur)
                conn_cur = []
    return conns


class DiscourseSignalExtractor:
    def __init__(self, tokenizer, signal_models, relation_type, device='cpu', use_crf=True):
        self.tokenizer = tokenizer
        self.signal_models = signal_models
        self.relation_type = relation_type
        self.device = device
        self.id2label = signal_models[0].config.id2label
        self.use_crf = use_crf

    @staticmethod
    def load_model(save_path, relation_type, device='cpu', use_crf=True):
        save_paths = glob.glob(save_path)
        print(f"Load models: {save_paths}", file=sys.stderr)
        tokenizer = AutoTokenizer.from_pretrained("roberta-base", add_prefix_space=True, local_files_only=True)
        models = []
        file_end = 'crf' if use_crf else 'label'
        if len(save_paths) >= 1:
            for save_path in save_paths:
                label_save_path = os.path.join(save_path, f"best_model_{relation_type.lower()}_{file_end}")
                model = RobertaWithCRF.from_pretrained(label_save_path, local_files_only=True)
                model.eval()
                model.to(device)
                models.append(model)
        else:
            raise ValueError('No Models found.')
        return DiscourseSignalExtractor(tokenizer, models, relation_type, device, use_crf)

    # def predict(self, doc, batch_size=16):
    #     document_signals = []
    #     iter_filtered_paragraphs = filter(lambda p: sum(len(s.tokens) for s in p[1]) >= 7,
    #                                       enumerate(iter_document_paragraphs(doc)))
    #     while True:
    #         par_batch = []
    #         par_tokens = []
    #         par_idx = []
    #         for par_i, paragraph in itertools.islice(iter_filtered_paragraphs, batch_size):
    #             par_idx.append(par_i)
    #             tokens = [t for s in paragraph for t in s.tokens]
    #             par_tokens.append(tokens)
    #             par_batch.append([t.surface for t in tokens])
    #         if not par_batch:
    #             break
    #
    #         inputs = self.tokenizer(par_batch, truncation=True, is_split_into_words=True,
    #                                 padding="max_length", return_tensors='pt')
    #         if self.use_crf:
    #             probs, predictions = self.compute_ensemble_prediction(inputs)
    #         else:
    #             probs, predictions = self.compute_ensemble_label_prediction(inputs)
    #         for b_i, (par_i, tokens, pred, prob) in enumerate(zip(par_idx, par_tokens,
    #                                                               predictions, probs)):
    #             word_ids = inputs.word_ids(b_i)
    #             predicted_token_class = [self.id2label[t] for t in pred]
    #             predicted_token_prob = prob
    #             word_id_map = []
    #             for i, wi in enumerate(word_ids):
    #                 if wi is not None and (len(word_id_map) == 0 or (word_ids[i - 1] != wi)):
    #                     word_id_map.append(i)
    #
    #             signals = decode_labels([predicted_token_class[i] for i in word_id_map],
    #                                     [predicted_token_prob[i] for i in word_id_map])
    #             signals = [[tokens[i] for p, i in signal] for signal in signals]
    #             relations = [{
    #                 'tokens_idx': [t.idx for t in signal],
    #                 'tokens': [t.surface for t in signal],
    #                 'relation_type': self.relation_type,
    #             } for signal in signals]
    #
    #             document_signals.append({
    #                 'doc_id': doc.doc_id,
    #                 'paragraph_idx': par_i,
    #                 'tokens_idx': [t.idx for t in tokens],
    #                 'tokens': [t.surface for t in tokens],
    #                 'labels': [predicted_token_class[i] for i in word_id_map],
    #                 'probs': [round(predicted_token_prob[i], 4) for i in word_id_map],
    #                 'relations': relations,
    #             })
    #     return document_signals

    def predict_paragraphs(self, paragraphs):
        par_batch = []
        par_tokens = []
        for paragraph in paragraphs:
            tokens = [t for s in paragraph['sentences'] for t in s.tokens]
            par_tokens.append(tokens)
            par_batch.append([t.surface for t in tokens])

        inputs = self.tokenizer(par_batch, truncation=True, is_split_into_words=True,
                                padding="max_length", return_tensors='pt')
        if self.use_crf:
            probs, predictions = self.compute_ensemble_prediction(inputs)
        else:
            probs, predictions = self.compute_ensemble_label_prediction(inputs)
        # print("predictions", predictions)
        # print("predictions", len(par_tokens), len(predictions))
        # print(list(map(len, par_tokens)), list(map(len, predictions)))
        for b_i, (tokens, pred, prob) in enumerate(zip(par_tokens, predictions, probs)):
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
                'sentence_idx': list(set(t.sent_idx for t in signal)),
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
        if 'labels' in batch:
            del batch['labels']
        models_predictions = []
        with torch.no_grad():
            for model in self.signal_models:
                sequences = model.decode(**batch)
                models_predictions.append(sequences)
        predictions = []
        for preds in zip(*models_predictions):
            predictions.append(merge_list_majority(preds, default_value=0))
        probs = [[1.0 for _ in prediction] for prediction in predictions]
        return probs, sequences

    def compute_ensemble_label_prediction(self, batch):
        batch = {k: v.to(self.device) for k, v in batch.items()}
        predictions = []
        with torch.no_grad():
            for model in self.signal_models:
                logits = model.emissions(**batch)
                predictions.append(F.softmax(logits, dim=-1))
        probs, predictions = torch.max(torch.mean(torch.stack(predictions), dim=0), dim=-1)
        return probs.tolist(), predictions.tolist()


def merge_list_majority(preds, default_value=0):
    merged = []
    for line in np.stack(preds).T:
        values, counts = np.unique(line, return_counts=True)
        ind = np.argmax(counts)
        merged.append(values[ind] if counts[ind] > 1 else default_value)
    return merged


def compute_loss(logits, labels, device, majority_class_weight=1.0):
    num_labels = logits.size(dim=2)
    weights = [majority_class_weight] + ([1.0] * (num_labels - 1))
    loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device), ignore_index=7, label_smoothing=0.1)
    loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
    return loss


class RobertaWithCRF(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(0.4)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.linear_crf = LinearCRF(config.num_labels, batch_first=True)
        # Initialize weights and apply final processing
        self.init_weights()
        self.use_crf = True

        for param in self.roberta.embeddings.parameters():
            param.requires_grad = False
        for layer in self.roberta.encoder.layer[:6]:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, input_ids, labels, token_type_ids=None, attention_mask=None):
        emissions = self.emissions(input_ids=input_ids,
                                   token_type_ids=token_type_ids,
                                   attention_mask=attention_mask)
        loss = self.linear_crf(emissions, labels, mask=attention_mask > 0)
        return loss

    def emissions(self, input_ids, labels=None, token_type_ids=None, attention_mask=None, majority_class_weight=1.0):
        # attention_mask = input_ids.ne(self.vocab.token_to_idx[self.vocab.padding_token]).float()
        # outputs: (last_encoder_layer, pooled_output, attention_weight)
        outputs = self.roberta(input_ids=input_ids,
                               token_type_ids=token_type_ids,
                               attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        if labels is not None:
            return compute_loss(logits, labels, self.device, majority_class_weight), logits
        else:
            return logits

    def freeze(self, base_model=True, emissions=True):
        if base_model:
            for param in self.roberta.embeddings.parameters():
                param.requires_grad = False
            for layer in self.roberta.encoder.layer:
                for param in layer.parameters():
                    param.requires_grad = False
        if emissions:
            for param in self.classifier.parameters():
                param.requires_grad = False

    def predict(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        loss = None
        if self.use_crf:
            if labels is not None:
                loss, preds = self.decode(input_ids, labels, token_type_ids, attention_mask)
            else:
                preds = self.decode(input_ids, labels, token_type_ids, attention_mask)
        else:
            if labels is not None:
                loss, logits = self.emissions(input_ids, labels, token_type_ids, attention_mask)
            else:
                logits = self.emissions(input_ids, labels, token_type_ids, attention_mask)
            preds = torch.argmax(logits, dim=-1)
        return loss, preds

    #     pass
    # def emissions_predict(self, input_ids, labels, token_type_ids=None, attention_mask=None):
    #     logits = self.emissions(input_ids, labels, token_type_ids, attention_mask)
    #     preds = torch.argmax(logits, dim=-1)
    #     predictions = []
    #     references = []
    #     signals_pred = []
    #     signals_gold = []
    #     for pred, ref in zip(preds.tolist(), labels.tolist()):
    #         pred = [self.config.id2label[p] for i, p in enumerate(pred) if ref[i] != 7]
    #         ref = [self.config.id2label[i] for i in ref if i != 7]
    #         assert len(pred) == len(ref), f"PRED: {pred}, REF {ref}"
    #         predictions.append(pred)
    #         references.append(ref)
    #         signals_pred.append([(None, [i for p, i in signal], None) for signal in decode_labels(pred, pred)])
    #         signals_gold.append([(None, [i for p, i in signal], None) for signal in decode_labels(ref, ref)])
    #
    #     return logits, predictions, references, signals_pred, signals_gold

    def decode(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        emissions = self.emissions(input_ids=input_ids,
                                   token_type_ids=token_type_ids,
                                   attention_mask=attention_mask)
        sequences = self.linear_crf.decode(emissions, mask=attention_mask > 0)
        if labels is not None:
            loss = self.linear_crf(emissions, labels, mask=attention_mask > 0)
            return loss, sequences
        else:
            return sequences


class LinearCRF(nn.Module):
    def __init__(
            self,
            num_tags: int,
            batch_first: bool = False,
            impossible_starts: Optional[BoolTensor] = None,
            impossible_transitions: Optional[BoolTensor] = None,
            impossible_ends: Optional[BoolTensor] = None,
    ):
        super(LinearCRF, self).__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.starts = nn.Parameter(torch.empty(self.num_tags))
        self.transitions = nn.Parameter(torch.empty(self.num_tags, self.num_tags))
        self.ends = nn.Parameter(torch.empty(self.num_tags))
        self.reset_parameters(impossible_starts, impossible_transitions, impossible_ends)

    def reset_parameters(
            self,
            impossible_starts: Optional[BoolTensor] = None,
            impossible_transitions: Optional[BoolTensor] = None,
            impossible_ends: Optional[BoolTensor] = None,
    ) -> None:
        """
        Initialize the parameters of the model, impossible starts / transition / ends are set to
        -10000 to avoid being considered.

        Parameters:
        -----------
            impossible_starts: Optional[BoolTensor]
                shape: (num_tags,)
                impossible starting tags
            impossible_transitions: Optional[BoolTensor]
                shape: (num_tags, num_tags)
                impossible transition ([i,j] = True means tag i -> tag j is impossible)
            impossible_ends: Optional[BoolTensor]
                shape: (num_tags,)
                impossible ending tags
        """
        for param, impossible in zip(
                self.parameters(), [impossible_starts, impossible_transitions, impossible_ends]
        ):
            nn.init.uniform_(param, -0.1, 0.1)
            if impossible is not None:
                with torch.no_grad():
                    param.masked_fill_(impossible, -10000)

    def forward(
            self, emissions: FloatTensor, labels: LongTensor, mask: Optional[BoolTensor] = None
    ) -> FloatTensor:
        """
        Computes the negative log-likelihood given emission scores for a sequence of tags using the
        forward algorithm.

        Parameters:
        -----------
            emissions: FloatTensor
                shape: (seq_length, batch_size, num_tags) if batch_first is False
                emission score for each tag type and timestep
            labels: LongTensor
                shape: (seq_length, batch_size) if batch_first is False
                ground truth tag sequences
            mask: Optional[BoolTensor]
                shape: (seq_length, batch_size) if batch_first is False
                optional boolean mask for each sequence

        Returns:
        --------
            result: torch.FloatTensor
                shape: ()
                Negative log-likelihood normalized by the mask sum
        """
        if mask is None:
            mask = torch.ones_like(labels, dtype=torch.bool)
        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            labels = labels.transpose(0, 1)
            mask = mask.transpose(0, 1)

        numerator = self.starts[labels[0]]
        numerator += (emissions.gather(2, labels.unsqueeze(-1)).squeeze(-1) * mask).sum(dim=0)
        numerator += (self.transitions[labels[:-1], labels[1:]] * mask[1:]).sum(dim=0)
        seq_ends = mask.long().sum(dim=0) - 1
        last_tags = labels.gather(0, seq_ends.unsqueeze(0)).squeeze(0)
        numerator += self.ends[last_tags]

        denominator = self.starts + emissions[0]
        broadcast_emissions = emissions.unsqueeze(2)

        for i in range(1, labels.shape[0]):
            broadcast_denominator = denominator.unsqueeze(2)
            next_denominator = broadcast_denominator + self.transitions + broadcast_emissions[i]
            next_denominator = next_denominator.logsumexp(dim=1)
            denominator = next_denominator.where(mask[i].unsqueeze(1), denominator)

        denominator += self.ends
        denominator = denominator.logsumexp(dim=1)

        llh = numerator - denominator
        # divides the loss by tokens (token mean loss)
        return -llh.sum() / mask.sum()

    @torch.no_grad()
    def decode(self, emissions: FloatTensor, mask: Optional[BoolTensor] = None) -> List[List[int]]:
        """
        Computes the best tag sequence given emission scores using the Viterbi algorithm.

        Parameters:
        -----------
            emissions: FloatTensor
                shape: (seq_length, batch_size, num_tags) if batch_first is False
                emission score for each tag type and timestep
            mask: Optional[BoolTensor]
                shape: (seq_length, batch_size) if batch_first is False
                optional boolean mask for each sequence

        Returns:
        --------
            best_tags: List[List[int]]
                shape: (batch_size, real seq_length)
                best tag sequences
        """

        if mask is None:
            mask = emissions.new_ones(emissions.shape[:2], dtype=torch.bool)
        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        seq_length, batch_size = mask.shape
        broadcast_emissions = emissions.unsqueeze(2)
        score = self.starts + emissions[0]
        history = emissions.new_empty(seq_length - 1, batch_size, self.num_tags, dtype=torch.long)

        for i in range(1, seq_length):
            broadcast_score = score.unsqueeze(2)
            next_score = broadcast_score + self.transitions + broadcast_emissions[i]
            next_score, next_indices = next_score.max(dim=1)
            score = next_score.where(mask[i].unsqueeze(1), score)
            history[-i] = next_indices.where(mask[i - 1].unsqueeze(1), history[-i + 1])

        score += self.ends
        best_tags = torch.empty_like(mask, dtype=torch.long)
        _, best_last_tag = score.max(dim=1)
        best_prev_tag = best_last_tag

        for i in range(seq_length - 1):
            best_prev_tag = history[i].gather(1, best_prev_tag.unsqueeze(-1)).squeeze(-1)
            best_tags[seq_length - 2 - i] = best_prev_tag

        seq_ends = mask.long().sum(dim=0) - 1
        best_tags.scatter_(0, seq_ends.unsqueeze(0), best_last_tag.unsqueeze(0))
        return [line[: end + 1].tolist() for line, end in zip(best_tags.t(), seq_ends)]

# class CRF(nn.Module):
#     """
#     Linear-chain Conditional Random Field (CRF).
#
#     Args:
#         nb_labels (int): number of labels in your tagset, including special symbols.
#         bos_tag_id (int): integer representing the beginning of sentence symbol in
#             your tagset.
#         eos_tag_id (int): integer representing the end of sentence symbol in your tagset.
#         pad_tag_id (int, optional): integer representing the pad symbol in your tagset.
#             If None, the model will treat the PAD as a normal tag. Otherwise, the model
#             will apply constraints for PAD transitions.
#         batch_first (bool): Whether the first dimension represents the batch dimension.
#     """
#
#     def __init__(
#         self, nb_labels, bos_tag_id, eos_tag_id, pad_tag_id=None, batch_first=True
#     ):
#         super().__init__()
#
#         self.nb_labels = nb_labels
#         self.BOS_TAG_ID = bos_tag_id
#         self.EOS_TAG_ID = eos_tag_id
#         self.PAD_TAG_ID = pad_tag_id
#         self.batch_first = batch_first
#
#         self.transitions = nn.Parameter(torch.randn(self.nb_labels, self.nb_labels))
#         self.init_weights()
#
#     def init_weights(self):
#         # initialize transitions from a random uniform distribution between -0.1 and 0.1
#         # nn.init.uniform_(self.transitions, -0.1, 0.1)
#
#         # enforce contraints (rows=from, columns=to) with a big negative number
#         # so exp(-10000) will tend to zero
#
#         # no transitions allowed to the beginning of sentence
#         self.transitions.data[:, self.BOS_TAG_ID] = -10000.0
#         # no transition alloed from the end of sentence
#         self.transitions.data[self.EOS_TAG_ID, :] = -10000.0
#
#         if self.PAD_TAG_ID is not None:
#             # no transitions from padding
#             self.transitions.data[self.PAD_TAG_ID, :] = -10000.0
#             # no transitions to padding
#             self.transitions.data[:, self.PAD_TAG_ID] = -10000.0
#             # except if the end of sentence is reached
#             # or we are already in a pad position
#             self.transitions.data[self.EOS_TAG_ID, self.PAD_TAG_ID] = 0.0
#             self.transitions.data[self.PAD_TAG_ID, self.PAD_TAG_ID] = 0.0
#
#     def forward(self, emissions, tags, mask=None):
#         """Compute the negative log-likelihood. See `log_likelihood` method."""
#         nll = -self.log_likelihood(emissions, tags, mask=mask)
#         return nll
#
#     def log_likelihood(self, emissions, tags, mask=None):
#         """Compute the probability of a sequence of tags given a sequence of
#         emissions scores.
#
#         Args:
#             emissions (torch.Tensor): Sequence of emissions for each label.
#                 Shape of (batch_size, seq_len, nb_labels) if batch_first is True,
#                 (seq_len, batch_size, nb_labels) otherwise.
#             tags (torch.LongTensor): Sequence of labels.
#                 Shape of (batch_size, seq_len) if batch_first is True,
#                 (seq_len, batch_size) otherwise.
#             mask (torch.FloatTensor, optional): Tensor representing valid positions.
#                 If None, all positions are considered valid.
#                 Shape of (batch_size, seq_len) if batch_first is True,
#                 (seq_len, batch_size) otherwise.
#
#         Returns:
#             torch.Tensor: the log-likelihoods for each sequence in the batch.
#                 Shape of (batch_size,)
#         """
#
#         # fix tensors order by setting batch as the first dimension
#         if not self.batch_first:
#             emissions = emissions.transpose(0, 1)
#             tags = tags.transpose(0, 1)
#
#         if mask is None:
#             mask = torch.ones(emissions.shape[:2], dtype=torch.float)
#
#         scores = self._compute_scores(emissions, tags, mask=mask)
#         partition = self._compute_log_partition(emissions, mask=mask)
#         return torch.mean(scores - partition)
#
#     def decode(self, emissions, mask=None):
#         """Find the most probable sequence of labels given the emissions using
#         the Viterbi algorithm.
#
#         Args:
#             emissions (torch.Tensor): Sequence of emissions for each label.
#                 Shape (batch_size, seq_len, nb_labels) if batch_first is True,
#                 (seq_len, batch_size, nb_labels) otherwise.
#             mask (torch.FloatTensor, optional): Tensor representing valid positions.
#                 If None, all positions are considered valid.
#                 Shape (batch_size, seq_len) if batch_first is True,
#                 (seq_len, batch_size) otherwise.
#
#         Returns:
#             torch.Tensor: the viterbi score for the for each batch.
#                 Shape of (batch_size,)
#             list of lists: the best viterbi sequence of labels for each batch.
#         """
#         if mask is None:
#             mask = torch.ones(emissions.shape[:2], dtype=torch.float)
#
#         scores, sequences = self._viterbi_decode(emissions, mask)
#         return scores, sequences
#
#     def _compute_scores(self, emissions, tags, mask):
#         """Compute the scores for a given batch of emissions with their tags.
#
#         Args:
#             emissions (torch.Tensor): (batch_size, seq_len, nb_labels)
#             tags (Torch.LongTensor): (batch_size, seq_len)
#             mask (Torch.FloatTensor): (batch_size, seq_len)
#
#         Returns:
#             torch.Tensor: Scores for each batch.
#                 Shape of (batch_size,)
#         """
#         batch_size, seq_length = tags.shape
#         scores = torch.zeros(batch_size, device=emissions.device)
#
#         # save first and last tags to be used later
#         first_tags = tags[:, 0]
#         last_valid_idx = mask.int().sum(1) - 1
#         last_tags = tags.gather(1, last_valid_idx.unsqueeze(1)).squeeze()
#
#         # add the transition from BOS to the first tags for each batch
#         t_scores = self.transitions[self.BOS_TAG_ID, first_tags]
#
#         # add the [unary] emission scores for the first tags for each batch
#         # for all batches, the first word, see the correspondent emissions
#         # for the first tags (which is a list of ids):
#         # emissions[:, 0, [tag_1, tag_2, ..., tag_nblabels]]
#         e_scores = emissions[:, 0].gather(1, first_tags.unsqueeze(1)).squeeze()
#
#         # the scores for a word is just the sum of both scores
#         scores += e_scores + t_scores
#
#         # now lets do this for each remaining word
#         for i in range(1, seq_length):
#
#             # we could: iterate over batches, check if we reached a mask symbol
#             # and stop the iteration, but vecotrizing is faster due to gpu,
#             # so instead we perform an element-wise multiplication
#             is_valid = mask[:, i]
#
#             previous_tags = tags[:, i - 1]
#             current_tags = tags[:, i]
#
#             # calculate emission and transition scores as we did before
#             e_scores = emissions[:, i].gather(1, current_tags.unsqueeze(1)).squeeze()
#             t_scores = self.transitions[previous_tags, current_tags]
#
#             # apply the mask
#             e_scores = e_scores * is_valid
#             t_scores = t_scores * is_valid
#
#             scores += e_scores + t_scores
#
#         # add the transition from the end tag to the EOS tag for each batch
#         scores += self.transitions[last_tags, self.EOS_TAG_ID]
#
#         return scores
#
#     def _compute_log_partition(self, emissions, mask):
#         """Compute the partition function in log-space using the forward-algorithm.
#
#         Args:
#             emissions (torch.Tensor): (batch_size, seq_len, nb_labels)
#             mask (Torch.FloatTensor): (batch_size, seq_len)
#
#         Returns:
#             torch.Tensor: the partition scores for each batch.
#                 Shape of (batch_size,)
#         """
#         batch_size, seq_length, nb_labels = emissions.shape
#
#         # in the first iteration, BOS will have all the scores
#         alphas = self.transitions[self.BOS_TAG_ID, :].unsqueeze(0) + emissions[:, 0]
#
#         for i in range(1, seq_length):
#             # (bs, nb_labels) -> (bs, 1, nb_labels)
#             e_scores = emissions[:, i].unsqueeze(1)
#
#             # (nb_labels, nb_labels) -> (bs, nb_labels, nb_labels)
#             t_scores = self.transitions.unsqueeze(0)
#
#             # (bs, nb_labels)  -> (bs, nb_labels, 1)
#             a_scores = alphas.unsqueeze(2)
#
#             scores = e_scores + t_scores + a_scores
#             new_alphas = torch.logsumexp(scores, dim=1)
#
#             # set alphas if the mask is valid, otherwise keep the current values
#             is_valid = mask[:, i].unsqueeze(-1)
#             alphas = is_valid * new_alphas + (1 - is_valid) * alphas
#
#         # add the scores for the final transition
#         last_transition = self.transitions[:, self.EOS_TAG_ID]
#         end_scores = alphas + last_transition.unsqueeze(0)
#
#         # return a *log* of sums of exps
#         return torch.logsumexp(end_scores, dim=1)
#
#     def _viterbi_decode(self, emissions, mask):
#         """Compute the viterbi algorithm to find the most probable sequence of labels
#         given a sequence of emissions.
#
#         Args:
#             emissions (torch.Tensor): (batch_size, seq_len, nb_labels)
#             mask (Torch.FloatTensor): (batch_size, seq_len)
#
#         Returns:
#             torch.Tensor: the viterbi score for the for each batch.
#                 Shape of (batch_size,)
#             list of lists of ints: the best viterbi sequence of labels for each batch
#         """
#         batch_size, seq_length, nb_labels = emissions.shape
#
#         # in the first iteration, BOS will have all the scores and then, the max
#         alphas = self.transitions[self.BOS_TAG_ID, :].unsqueeze(0) + emissions[:, 0]
#
#         backpointers = []
#
#         for i in range(1, seq_length):
#             # (bs, nb_labels) -> (bs, 1, nb_labels)
#             e_scores = emissions[:, i].unsqueeze(1)
#
#             # (nb_labels, nb_labels) -> (bs, nb_labels, nb_labels)
#             t_scores = self.transitions.unsqueeze(0)
#
#             # (bs, nb_labels)  -> (bs, nb_labels, 1)
#             a_scores = alphas.unsqueeze(2)
#
#             # combine current scores with previous alphas
#             scores = e_scores + t_scores + a_scores
#
#             # so far is exactly like the forward algorithm,
#             # but now, instead of calculating the logsumexp,
#             # we will find the highest score and the tag associated with it
#             max_scores, max_score_tags = torch.max(scores, dim=1)
#
#             # set alphas if the mask is valid, otherwise keep the current values
#             is_valid = mask[:, i].unsqueeze(-1)
#             alphas = is_valid * max_scores + (1 - is_valid) * alphas
#
#             # add the max_score_tags for our list of backpointers
#             # max_scores has shape (batch_size, nb_labels) so we transpose it to
#             # be compatible with our previous loopy version of viterbi
#             backpointers.append(max_score_tags.t())
#
#         # add the scores for the final transition
#         last_transition = self.transitions[:, self.EOS_TAG_ID]
#         end_scores = alphas + last_transition.unsqueeze(0)
#
#         # get the final most probable score and the final most probable tag
#         max_final_scores, max_final_tags = torch.max(end_scores, dim=1)
#
#         # find the best sequence of labels for each sample in the batch
#         best_sequences = []
#         emission_lengths = mask.int().sum(dim=1)
#         for i in range(batch_size):
#
#             # recover the original sentence length for the i-th sample in the batch
#             sample_length = emission_lengths[i].item()
#
#             # recover the max tag for the last timestep
#             sample_final_tag = max_final_tags[i].item()
#
#             # limit the backpointers until the last but one
#             # since the last corresponds to the sample_final_tag
#             sample_backpointers = backpointers[: sample_length - 1]
#
#             # follow the backpointers to build the sequence of labels
#             sample_path = self._find_best_path(i, sample_final_tag, sample_backpointers)
#
#             # add this path to the list of best sequences
#             best_sequences.append(sample_path)
#
#         return max_final_scores, best_sequences
#
#     def _find_best_path(self, sample_id, best_tag, backpointers):
#         """Auxiliary function to find the best path sequence for a specific sample.
#
#             Args:
#                 sample_id (int): sample index in the range [0, batch_size)
#                 best_tag (int): tag which maximizes the final score
#                 backpointers (list of lists of tensors): list of pointers with
#                 shape (seq_len_i-1, nb_labels, batch_size) where seq_len_i
#                 represents the length of the ith sample in the batch
#
#             Returns:
#                 list of ints: a list of tag indexes representing the bast path
#         """
#
#         # add the final best_tag to our best path
#         best_path = [best_tag]
#
#         # traverse the backpointers in backwards
#         for backpointers_t in reversed(backpointers):
#
#             # recover the best_tag at this timestep
#             best_tag = backpointers_t[best_tag][sample_id].item()
#
#             # append to the beginning of the list so we don't need to reverse it later
#             best_path.insert(0, best_tag)
#
#         return best_path
