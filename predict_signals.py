import csv
import itertools
import sys
from pathlib import Path

import click
import torch

from helpers.data import get_corpus_path, load_docs
from helpers.signals import DiscourseSignalModel
from predict_labels import iter_documents_paragraphs


# def decode_labels(tokens, labels, probs):
#     conns = []
#     for tok_i, (tok, label, prob) in enumerate(zip(tokens, labels, probs)):
#         if label.startswith('S'):
#             conns.append([(probs, tok)])
#     conn_stack = []
#     conn_cur = []
#     for tok_i, (tok, label, prob) in enumerate(zip(tokens, labels, probs)):
#         if label.startswith('B'):
#             if conn_cur:
#                 conn_stack.append(conn_cur)
#                 conn_cur = []
#             conn_cur.append((prob, tok))
#         elif label.startswith('I'):
#             if conn_cur:
#                 conn_cur.append((prob, tok))
#             else:
#                 conn_cur = conn_stack.pop() if conn_stack else []
#                 conn_cur.append((prob, tok))
#         elif label.startswith('E'):
#             if conn_cur:
#                 conn_cur.append((prob, tok))
#                 conns.append(conn_cur)
#             if conn_stack:
#                 conn_cur = conn_stack.pop()
#             else:
#                 conn_cur = []
#     return conns


@click.command()
@click.argument('corpus')
@click.argument('relation-type')
@click.argument('save-path')
@click.option('-r', '--replace', is_flag=True)
@click.option('-o', '--output-path', default='-')
@click.option('-l', '--limit', default=0, type=int)
@click.option('-b', '--batch-size', default=32, type=int)
@click.option('-f', '--output-format', default='json')
def main(corpus, relation_type, save_path, replace, output_path, limit, batch_size, output_format):
    if output_path == '-':
        output = sys.stdout
    else:
        output_path = Path(output_path)
        if output_path.is_file() and output_path.stat().st_size > 100 and not replace:
            sys.stderr.write('File already exists: Exit without writing.')
            return
        else:
            output_path.parent.mkdir(exist_ok=True, parents=True)
            output = output_path.open('w')

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    signal_model = DiscourseSignalModel.load_model(save_path, relation_type, device=device)

    # corpus_path = get_corpus_path(corpus)
    # save_paths = glob.glob(save_path)
    # label_save_path = os.path.join(save_paths[0], f"best_model_{relation_type.lower()}_label")
    # tokenizer = AutoTokenizer.from_pretrained("roberta-base", add_prefix_space=True)
    # signal_model = AutoModelForTokenClassification.from_pretrained(label_save_path)
    # sense_model_embed = AutoModel.from_pretrained("roberta-base")
    # if len(save_paths) == 1:
    #     sense_model = DiscourseSenseClassifier.load(save_paths[0], relation_type=relation_type.lower())
    # else:
    #     sense_model = DiscourseSenseEnsembleClassifier.load(save_paths, device, relation_type=relation_type.lower())

    csv_out = csv.writer(output)

    corpus_path = get_corpus_path(corpus)
    paragraphs = filter(lambda p: sum(len(s.tokens) for s in p['sentences']) > 7,
                        iter_documents_paragraphs(load_docs(corpus_path, limit=limit)))
    while True:
        batch = list(itertools.islice(paragraphs, batch_size))
        if len(batch) == 0:
            break
        par_preds = signal_model.predict_paragraphs(batch)
        for par in filter(lambda p: len(p['relations']) > 0, par_preds):
            # json.dump(s, output)
            # output.write('\n')
            # {"doc_id": "nyt_00009", "paragraph_idx": 1,
            # "tokens_idx": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
            # 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51],
            # "tokens": ["We", "have", "known", "for", "well", "over", "five", "years", "that", "C", "-", "reactiv
            # e", "protein", "is", "more", "critical", "than", "cholesterol", ".", "There", "are", "at", "least", "a", "dozen", "nondrug", "ways", "to", "lower", "CRP", ",", "all", "safer
            # ", "than", "statin", "drugs", "!", "And", "there", "are", "even", "more", "safe", "methods", "for", "lowering", "cholesterol", "."],
            # "labels": ["O", "O", "O", "O", "O", "O",
            #  "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "S", "O", "O", "O
            # ", "O", "O", "O", "O", "O", "O", "O"],
            # "probs": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            #  1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            #  "relations": [{"tokens_idx": [41], "tokens": ["And"],
            #  "coarse": "Expansion", "coarse_probs": 0.9536, "fine": "Expansion.Conjunction", "fine_probs": 0.9695}]}
            for rel in par['relations']:
                csv_out.writerow([
                    corpus, relation_type.lower(),
                    par['doc_id'], par['paragraph_idx'],
                    # np.mean([p for p, t in signal]).round(4),
                    '-'.join(str(i) for i in rel['tokens_idx']),
                    '-'.join(t.lower() for t in rel['tokens']),
                    rel['coarse'], rel['coarse_probs'],
                    rel['fine'], rel['fine_probs']
                ])

        output.flush()

    # signal_model.to(device)
    # sense_model_embed.to(device)
    # sense_model.to(device)
    # for doc_i, doc in enumerate(load_docs(corpus_path)):
    #     if limit and doc_i >= limit:
    #         break
    #     try:
    #         sentence_embeddings = get_doc_embeddings(doc, tokenizer, sense_model_embed, device=device)
    #     except RuntimeError as e:
    #         sys.stderr.write(f"Error {doc_i}: {e}")
    #         continue
    #     for p_i, paragraph in enumerate(iter_document_paragraphs(doc)):
    #         tokens = [t for s in paragraph for t in s.tokens]
    #         input_strings = [t.surface for t in tokens]
    #         inputs = tokenizer(input_strings, truncation=True, is_split_into_words=True,
    #                            padding="max_length", return_tensors='pt')
    #         with torch.no_grad():
    #             _inputs = {k: v.to(device) for k, v in inputs.items()}
    #             logits = signal_model(**_inputs).logits
    #             logits = F.softmax(logits, dim=-1)
    #         probs, predictions = torch.max(logits, dim=2)
    #         word_ids = inputs.word_ids()
    #         predicted_token_class = [signal_model.config.id2label[t] for t in predictions.tolist()[0]]
    #         predicted_token_prob = probs.tolist()[0]
    #         word_id_map = []
    #         for i, wi in enumerate(word_ids):
    #             if wi is not None and (len(word_id_map) == 0 or (word_ids[i - 1] != wi)):
    #                 word_id_map.append(i)
    #         signals = decode_labels(tokens,
    #                                 [predicted_token_class[i] for i in word_id_map],
    #                                 [predicted_token_prob[i] for i in word_id_map])
    #         for sent_i, sent in enumerate(paragraph):
    #             sentence_signals = [signal for signal in signals if signal[0][1].sent_idx == sent.tokens[0].sent_idx]
    #             if sentence_signals:
    #                 features = np.stack([get_bert_features([t.idx for i, t in signal], sentence_embeddings)]
    #                                     for signal in sentence_signals)
    #                 features = torch.from_numpy(features).to(device)
    #                 pred = sense_model.predict(features)
    #
    #                 for signal, coarse_class_i, coarse_class_i_prob, fine_class_i, fine_class_i_prob in zip(
    #                         sentence_signals,
    #                         pred['coarse'], pred['probs_coarse'], pred['fine'], pred['probs_fine']):
    #                     csv_out.writerow([
    #                         corpus, relation_type.lower(),
    #                         doc.doc_id, p_i, sent_i,
    #                         np.mean([p for p, t in signal]).round(4),
    #                         '-'.join(str(t.idx) for i, t in signal),
    #                         '-'.join(t.surface.lower() for i, t in signal),
    #                         coarse_class_i, round(coarse_class_i_prob, 4),
    #                         fine_class_i, round(fine_class_i_prob, 4)
    #                         # ' '.join(t.surface for s in [s for s in doc.sentences if ])
    #                     ])
    #     output.flush()


if __name__ == '__main__':
    main()
