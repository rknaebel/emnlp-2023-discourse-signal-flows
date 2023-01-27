from collections import Counter

import click
import evaluate
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm.auto import tqdm
from transformers import AutoModelForTokenClassification
from transformers import get_scheduler

from helpers.data import load_docs, iter_document_paragraphs


def tokenize_and_align_labels(tokenizer, examples):
    tokenized_inputs = tokenizer(examples["tokens"],
                                 truncation=True, is_split_into_words=True, padding="max_length", return_tensors='pt')
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


def get_explicit_label_mapping(doc):
    token_tag_map = {}
    for r in doc.relations:
        if r.is_explicit():
            for t_i, t in enumerate(r.conn.tokens):
                if t.idx not in token_tag_map:
                    if len(r.conn.tokens) == 1:
                        label = 'S-CONN'
                    else:
                        if t_i == 0:
                            label = 'B-CONN'
                        elif t_i == (len(r.conn.tokens) - 1):
                            label = 'E-CONN'
                        else:
                            label = 'I-CONN'
                    token_tag_map[t.idx] = label
    return token_tag_map


def get_altlex_label_mapping(doc):
    token_tag_map = {}
    for r in doc.relations:
        if r.type == 'AltLex':
            for t_i, t in enumerate(r.conn.tokens):
                if t.idx not in token_tag_map:
                    if len(r.conn.tokens) == 1:
                        label = 'S-ALTLEX'
                    else:
                        if t_i == 0:
                            label = 'B-ALTLEX'
                        elif t_i == (len(r.conn.tokens) - 1):
                            label = 'E-ALTLEX'
                        else:
                            label = 'I-ALTLEX'
                    token_tag_map[t.idx] = label
    return token_tag_map


class ConnDataset(Dataset):

    def __init__(self, data_file, bert_model, relation_type='explicit'):
        self.items = []
        self.labels = {'O': 0}
        self.bert_model = bert_model

        for doc_i, doc in enumerate(load_docs(data_file)):
            if relation_type.lower() == 'explicit':
                label_mapping = get_explicit_label_mapping(doc)
            else:
                label_mapping = get_altlex_label_mapping(doc)
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
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("roberta-base", add_prefix_space=True)

        def collate(examples):
            examples = {
                'tokens': [example['tokens'] for example in examples],
                'tags': [example['tags'] for example in examples],
            }
            batch = tokenize_and_align_labels(tokenizer, examples)
            batch['labels'] = torch.LongTensor(batch['labels'])
            return batch

        return collate


def compute_loss(num_labels, logits, labels, device):
    weights = [1.0] + [3.0] * (num_labels - 1)
    loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device))
    loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
    return loss


@click.command()
@click.argument('corpus-path')
@click.argument('relation-type')
@click.option('-b', '--batch-size', type=int, default=8)
@click.option('--split-ratio', type=float, default=0.8)
@click.option('--bert-model', default="roberta-base")
@click.option('--save-path', default="model")
def main(corpus_path, relation_type, batch_size, split_ratio, bert_model, save_path):
    conn_dataset = ConnDataset(corpus_path, bert_model, relation_type=relation_type)
    print('SAMPLE', len(conn_dataset), conn_dataset[0])
    print('LABELS:', conn_dataset.labels)
    print('LABEL COUNTS:', conn_dataset.get_label_counts())
    dataset_length = len(conn_dataset)
    train_size = int(dataset_length * split_ratio)
    valid_size = dataset_length - train_size
    train_dataset, valid_dataset = random_split(conn_dataset, [train_size, valid_size])
    print(len(train_dataset), len(valid_dataset))

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size,
                                  collate_fn=ConnDataset.get_collate_fn())
    eval_dataloader = DataLoader(valid_dataset, batch_size=batch_size,
                                 collate_fn=ConnDataset.get_collate_fn())

    label2id = conn_dataset.labels
    id2label = {v: k for k, v in label2id.items()}

    model = AutoModelForTokenClassification.from_pretrained(bert_model,
                                                            num_labels=conn_dataset.get_num_labels(),
                                                            id2label=id2label, label2id=label2id)
    for param in model.base_model.embeddings.parameters():
        param.requires_grad = False
    for layer in model.base_model.encoder.layer[:6]:
        for param in layer.parameters():
            param.requires_grad = False
    optimizer = AdamW(model.parameters(), lr=5e-5)

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
            outputs = model(**batch)
            loss = compute_loss(conn_dataset.get_num_labels(), outputs.logits, batch['labels'], device)
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
                outputs = model(**batch)

            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            predictions = []
            references = []
            for pred, ref in zip(preds.tolist(), batch['labels'].tolist()):
                pred = [id2label[p] for i, p in enumerate(pred) if ref[i] != -100]
                ref = [id2label[i] for i in ref if i != -100]
                assert len(pred) == len(ref), f"PRED: {pred}, REF {ref}"
                predictions.append(pred)
                references.append(ref)
            metric.add_batch(predictions=predictions, references=references)

        results = metric.compute()
        for key, vals in results.items():
            if key == 'accuracy':
                print(f"{key:10}  {vals * 100:02.2f}")
            else:
                print(
                    f"{key:10}  {vals['precision'] * 100:02.2f}  {vals['recall'] * 100:02.2f}  {vals['f1-score'] * 100:02.2f}  {vals['support']}")

        current_score = results['macro avg']['f1-score']
        if current_score > best_score:
            print("Store new best model!")
            model.save_pretrained(save_path)


if __name__ == '__main__':
    main()
