import csv
import os
import sys
from pathlib import Path

import click
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModel

from helpers.data import get_corpus_path, iter_document_paragraphs, load_docs, get_doc_embeddings
from helpers.senses import get_bert_features, NeuralNetwork


def decode_labels(tokens, labels, probs):
    conns = []
    for tok_i, (tok, label, prob) in enumerate(zip(tokens, labels, probs)):
        if label.startswith('S'):
            conns.append([(probs, tok)])
    conn_stack = []
    conn_cur = []
    for tok_i, (tok, label, prob) in enumerate(zip(tokens, labels, probs)):
        if label.startswith('B'):
            if conn_cur:
                conn_stack.append(conn_cur)
                conn_cur = []
            conn_cur.append((prob, tok))
        elif label.startswith('I'):
            if conn_cur:
                conn_cur.append((prob, tok))
            else:
                conn_cur = conn_stack.pop() if conn_stack else []
                conn_cur.append((prob, tok))
        elif label.startswith('E'):
            if conn_cur:
                conn_cur.append((prob, tok))
                conns.append(conn_cur)
            if conn_stack:
                conn_cur = conn_stack.pop()
            else:
                conn_cur = []
    return conns


@click.command()
@click.argument('corpus')
@click.argument('relation-type')
@click.argument('save-path')
@click.option('-r', '--replace', is_flag=True)
@click.option('-o', '--output-path', default='-')
def main(corpus, relation_type, save_path, replace, output_path):
    if output_path == '-':
        output = sys.stdout
    else:
        output_path = Path(output_path)
        if output_path.is_file() and output_path.stat().st_size > 100 and not replace:
            sys.stderr.write('File already exists: Exit without writing.')
            return
        else:
            output = output_path.open('w')

    corpus_path = get_corpus_path(corpus)
    label_save_path = os.path.join(save_path, f"best_model_{relation_type.lower()}_label")
    tokenizer = AutoTokenizer.from_pretrained("roberta-base", add_prefix_space=True)
    signal_model = AutoModelForTokenClassification.from_pretrained(label_save_path)
    sense_model_embed = AutoModel.from_pretrained("roberta-base")
    sense_model, label2id_coarse, label2id_fine = NeuralNetwork.load(save_path, relation_type)

    csv_out = csv.writer(output)

    id2label_coarse = {v: k for k, v in label2id_coarse.items()}
    id2label_fine = {v: k for k, v in label2id_fine.items()}

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    signal_model.to(device)
    sense_model_embed.to(device)
    sense_model.to(device)
    for doc_i, doc in enumerate(load_docs(corpus_path)):
        try:
            sentence_embeddings = get_doc_embeddings(doc.sentences, tokenizer, sense_model_embed, device=device)
        except RuntimeError as e:
            sys.stderr.write(f"Error {doc_i}: {e}")
            continue
        for p_i, paragraph in enumerate(iter_document_paragraphs(doc)):
            tokens = [t for s in paragraph for t in s.tokens]
            input_strings = [t.surface for t in tokens]
            inputs = tokenizer(input_strings, truncation=True, is_split_into_words=True,
                               padding="max_length", return_tensors='pt')
            with torch.no_grad():
                _inputs = {k: v.to(device) for k, v in inputs.items()}
                logits = signal_model(**_inputs).logits
                logits = F.softmax(logits, dim=-1)
            probs, predictions = torch.max(logits, dim=2)
            word_ids = inputs.word_ids()
            predicted_token_class = [signal_model.config.id2label[t] for t in predictions.tolist()[0]]
            predicted_token_prob = probs.tolist()[0]
            word_id_map = []
            for i, wi in enumerate(word_ids):
                if wi is not None and (len(word_id_map) == 0 or (word_ids[i - 1] != wi)):
                    word_id_map.append(i)
            signals = decode_labels(tokens,
                                    [predicted_token_class[i] for i in word_id_map],
                                    [predicted_token_prob[i] for i in word_id_map])
            for sent_i, sent in enumerate(paragraph):
                sentence_signals = [signal for signal in signals if signal[0][1].sent_idx == sent.tokens[0].sent_idx]
                if sentence_signals:
                    features = np.stack([get_bert_features([t.idx for i, t in signal], sentence_embeddings)]
                                        for signal in sentence_signals)
                    features = torch.from_numpy(features).to(device)
                    with torch.no_grad():
                        outputs_coarse, outputs_fine = sense_model(features)
                    sense_probs_coarse, sense_predictions_coarse = F.softmax(outputs_coarse, dim=-1).max(dim=-1)
                    sense_probs_fine, sense_predictions_fine = F.softmax(outputs_fine, dim=-1).max(dim=-1)

                    for signal, coarse_class_i, coarse_class_i_prob, fine_class_i, fine_class_i_prob in zip(
                            sentence_signals, sense_predictions_coarse.tolist(), sense_probs_coarse.tolist(),
                            sense_predictions_fine.tolist(), sense_probs_fine.tolist()):
                        csv_out.writerow([
                            corpus, relation_type.lower(),
                            doc.doc_id, p_i, sent_i,
                            np.mean([p for p, t in signal]).round(4),
                            '-'.join(str(t.idx) for i, t in signal),
                            '-'.join(t.surface.lower() for i, t in signal),
                            id2label_coarse[coarse_class_i],
                            round(coarse_class_i_prob, 4),
                            id2label_fine[fine_class_i],
                            round(fine_class_i_prob, 4)
                            # ' '.join(t.surface for s in [s for s in doc.sentences if ])
                        ])
        output.flush()


if __name__ == '__main__':
    main()
