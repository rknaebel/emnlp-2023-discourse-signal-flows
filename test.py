import click
import sklearn
import torch
from tqdm import tqdm

from helpers.data import get_corpus_path, load_docs, get_sense
from helpers.evaluate import evaluate_docs, print_results
from helpers.signals import DiscourseSignalModel


@click.command()
@click.argument('corpus')
@click.argument('relation-type')
@click.option('--save-path', default=".")
@click.option('--random-seed', default=42, type=int)
def main(corpus, relation_type, save_path, random_seed):
    corpus_path = get_corpus_path(corpus)
    train_docs = list(load_docs(corpus_path))
    train_docs, test_docs = sklearn.model_selection.train_test_split(train_docs, test_size=0.1,
                                                                     random_state=random_seed)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    signal_model = DiscourseSignalModel.load_model(save_path, relation_type, device=device)

    gold_docs = []
    pred_docs = []
    for doc_i, doc in tqdm(enumerate(test_docs)):
        doc_predictions = signal_model.predict(doc)
        pred_rels = [(relation_type, tuple(rel['tokens_idx']), rel['fine'])
                     for par in doc_predictions for rel in par['relations']]
        gold_rels = [(rel.type, tuple(tok.idx for tok in rel.conn.tokens),
                      [get_sense(s, 2, simple_sense=True) for s in rel.senses])
                     for rel in doc.relations if rel.type.lower() == relation_type.lower()]
        gold_docs.append(gold_rels)
        pred_docs.append(pred_rels)

    for threshold in [0.7, 0.9]:
        results = evaluate_docs(gold_docs, pred_docs, threshold=threshold)
        # if output_format == 'latex':
        #     model_name = os.path.basename(save_path)
        #     print_results_latex(res_explicit, res_non_explicit, title=f'{model_name}-{title}-{threshold}')
        # else:
        print_results(results, title=f'{relation_type.upper()}-{threshold}')

    # results = {}
    # precision, recall, f1 = score_paragraphs(signals_gold, signals_pred, threshold=0.7)
    # results['partial-match'] = {
    #     'precision': precision,
    #     'recall': recall,
    #     'f1-score': f1,
    #     'support': 0,
    # }
    # precision, recall, f1 = score_paragraphs(signals_gold, signals_pred, threshold=0.9)
    # results['exact-match'] = {
    #     'precision': precision,
    #     'recall': recall,
    #     'f1-score': f1,
    #     'support': 0,
    # }

    # print(f"==>"
    #       f"  {results['partial-match']['precision']}"
    #       f"  {results['partial-match']['recall']}"
    #       f"  {results['partial-match']['f1-score']}"
    #       f"  {results['exact-match']['precision']}"
    #       f"  {results['exact-match']['recall']}"
    #       f"  {results['exact-match']['f1-score']}"
    #       )

    # for key, vals in results.items():
    #     if key == 'accuracy':
    #         print(f"{key:10}  {vals * 100:02.2f}")
    #     else:
    #         print(
    #             f"{key:10}  {vals['precision'] * 100:02.2f}  {vals['recall'] * 100:02.2f}  {vals['f1-score'] * 100:02.2f}  {vals['support']}")


if __name__ == '__main__':
    main()
