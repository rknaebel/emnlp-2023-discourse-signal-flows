import csv
import os
import sys

import click
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModel

from helpers.data import get_corpus_path, iter_document_paragraphs, load_docs, get_doc_embeddings
from helpers.senses import get_bert_features, NeuralNetwork


def decode_labels(tokens, labels):
    conns = []
    for tok_i, (tok, label) in enumerate(zip(tokens, labels)):
        if label.startswith('S'):
            conns.append([(tok_i, tok)])
    conn_stack = []
    conn_cur = []
    for tok_i, (tok, label) in enumerate(zip(tokens, labels)):
        if label.startswith('B'):
            if conn_cur:
                conn_stack.append(conn_cur)
                conn_cur = []
            conn_cur.append((tok_i, tok))
        elif label.startswith('I'):
            if conn_cur:
                conn_cur.append((tok_i, tok))
            else:
                conn_cur = conn_stack.pop() if conn_stack else []
                conn_cur.append((tok_i, tok))
        elif label.startswith('E'):
            if conn_cur:
                conn_cur.append((tok_i, tok))
                conns.append(conn_cur)
            if conn_stack:
                conn_cur = conn_stack.pop()
            else:
                conn_cur = []
    return conns


@click.command()
@click.argument('corpus')
@click.argument('relation-type')
@click.option('--save-path', default=".")
def main(corpus, relation_type, save_path):
    corpus_path = get_corpus_path(corpus)
    save_path = os.path.join(save_path, f"best_model_{relation_type.lower()}_label")
    tokenizer = AutoTokenizer.from_pretrained("roberta-base", add_prefix_space=True)
    signal_model = AutoModelForTokenClassification.from_pretrained(save_path)
    sense_model_embed = AutoModel.from_pretrained("roberta-base")
    sense_model_state = torch.load(f"best_model_{relation_type.lower()}_lvl1_sense.pt")
    sense_model = NeuralNetwork(**sense_model_state['config'])
    sense_model.load_state_dict(sense_model_state['model'])

    csv_out = csv.writer(sys.stdout)

    label2id = sense_model_state['vocab']
    id2label = {v: k for k, v in label2id.items()}

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    signal_model.to(device)
    sense_model_embed.to(device)
    sense_model.to(device)
    for doc_i, doc in enumerate(load_docs(corpus_path)):
        sentence_embeddings = get_doc_embeddings(doc.sentences, tokenizer, sense_model_embed, device=device)
        for p_i, paragraph in enumerate(iter_document_paragraphs(doc)):
            tokens = [t for s in paragraph for t in s.tokens]
            input_strings = [t.surface for t in tokens]
            inputs = tokenizer(input_strings, truncation=True, is_split_into_words=True,
                               padding="max_length", return_tensors='pt')
            with torch.no_grad():
                _inputs = {k: v.to(device) for k, v in inputs.items()}
                logits = signal_model(**_inputs).logits
            predictions = torch.argmax(logits, dim=2).tolist()
            word_ids = inputs.word_ids()
            predicted_token_class = [signal_model.config.id2label[t] for t in predictions[0]]
            word_id_map = []
            for i, wi in enumerate(word_ids):
                if wi is not None and (len(word_id_map) == 0 or (word_ids[i - 1] != wi)):
                    word_id_map.append(i)
            signals = decode_labels(tokens, [predicted_token_class[i] for i in word_id_map])

            for sent_i, sent in enumerate(paragraph):
                sentence_signals = [signal for signal in signals if signal[0][1].sent_idx == sent.tokens[0].sent_idx]
                if sentence_signals:
                    features = np.stack([get_bert_features([t.idx for i, t in signal], sentence_embeddings)]
                                        for signal in sentence_signals)
                    features = torch.from_numpy(features).to(device)
                    with torch.no_grad():
                        outputs = sense_model(features)
                    sense_prediction = outputs.argmax(-1).tolist()

                    for signal, class_i in zip(sentence_signals, sense_prediction):
                        csv_out.writerow([
                            corpus, doc.doc_id, p_i, sent_i, signal[0][1].idx,
                            '-'.join(t[1].surface.lower() for t in signal),
                            id2label[class_i],
                        ])


if __name__ == '__main__':
    main()
