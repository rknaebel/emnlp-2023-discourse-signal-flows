import glob
import math
from collections import defaultdict, Counter
from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, f1_score
from sklearn.model_selection import train_test_split

column_names = ('corpus,type,doc_id,paragraph,indices,signal,'
                'sense1,sense1_prob,sense2,sense2_prob').split(',')

corpora = ['pdtb3', 'essay', 'ted', 'unsc', 'bbc', 'anthology']


def load_dataframe(corpus, relation_type, model_results='v2'):
    paths = glob.glob(f"results/{model_results}/{corpus}.{relation_type}.csv")
    dfs = []
    for path in paths:
        try:
            dfs.append(pd.read_csv(path, names=column_names))
        except Exception as e:
            print(e)
            continue
    df = pd.concat(dfs)
    df['index0'] = df.indices.apply(lambda i: int(i.split('-')[0]))
    return df


def iter_dataframes():
    for c in corpora:
        df = load_dataframe(c, '*')
        if df is not None:
            yield df


def get_sense_counts(df, sense_level='sense2'):
    df_ctr = df.groupby(['corpus', 'doc_id'])[sense_level].value_counts()

    df_tmp = defaultdict(dict)
    for (corpus, doc_id, relation), val in df_ctr.iteritems():
        df_tmp[corpus, doc_id][relation] = val

    return pd.DataFrame.from_dict(df_tmp).transpose().fillna(0)


def get_sense_distributions(df, sense_level='sense2', confidence=0.7, normalize=True):
    df_ctr = get_sense_counts(df[df[f'{sense_level}_prob'] > confidence], sense_level)
    if normalize:
        return df_ctr.div(df_ctr.sum(axis=1), axis=0)
    return df_ctr


def aggregate_dists(df):
    df = df.agg([np.mean, np.std])
    df = df.reindex(sorted(df), axis=1)
    return df.transpose()


def get_ngrams(seq, window_size):
    ngrams = []
    for i in range(len(seq) - window_size + 1):
        gram = tuple(seq[i: i + window_size])
        if not all(g == '<pad>' for g in gram):
            ngrams.append(gram)
    #         n_gram = seq[i: i + window_size]
    #         if len(n_gram) == 3:
    #             ngrams.append((n_gram[0], ('X', 'X'), n_gram[-1]))
    return ngrams


def get_change_flows(seq):
    return [seq[0]] + [j for i, j in enumerate(seq[1:]) if seq[i] != j]


def extract_flows_from_df(df, ngrams_sizes=(1, 2, 3), use_change_flows=False, min_flow_count=5, max_flow_count=100):
    documents = defaultdict(Counter)
    for doc_i, (name, group) in enumerate(df.groupby(['corpus', 'doc_id'])):
        sense_flow = group.sort_values(['index0'])['sense2'].to_list()
        # if len(sense_flow) < min_flow_count:
        #     continue
        if use_change_flows:
            sense_flow = get_change_flows(sense_flow)
        sense_flow = [s[s.find('.') + 1:] for s in sense_flow]
        sense_flow = ['<pad>'] + sense_flow + ['<pad>']
        for ngram_size in ngrams_sizes:
            ngrams = map(lambda g: '-'.join(g), get_ngrams(sense_flow, ngram_size))
            # bigrams = map(lambda g: '-'.join(g), get_ngrams(sense_flow, 2))
            # trigrams = map(lambda g: '-'.join(g), get_ngrams(sense_flow, 3))
            # # trigrams_holes = (f'{s1}-X-{s3}' for s1, _, s3 in get_ngrams(sense_change_flow, 3))
            documents[name] += Counter(
                ngrams)  # + Counter(bigrams) + Counter(trigrams)  # + Counter(trigrams_holes)
    return documents


def get_flow_counts(df, normalize=True, ngrams=(1, 2, 3), use_change_flows=False):
    documents = extract_flows_from_df(df, ngrams_sizes=ngrams, use_change_flows=use_change_flows)
    df_ctr = pd.DataFrame.from_dict(documents).fillna(0.0).transpose()
    if normalize:
        df_ctr = df_ctr.div(df_ctr.sum(axis=1), axis=0)
    #     df_ctr[df_ctr >= 1.0] = 1.0

    return df_ctr


def load_datasets(corpora, ngrams=(1, 2, 3), use_change_flows=True, corpus_name_mapping=None):
    dfs = []
    for corpus in corpora:
        df = load_dataframe(corpus, '*')
        if corpus_name_mapping:
            df.corpus = corpus_name_mapping.get(corpus, corpus)
        df_dist = get_flow_counts(df, ngrams=ngrams, use_change_flows=use_change_flows, normalize=False)
        dfs.append(df_dist)
    dfs = pd.concat(dfs).fillna(0)
    return dfs


# from scipy.stats import pearsonr
#
#
# def calc_signi(dfs):
#     df = pd.get_dummies(dfs, columns=['topic'])
#     rho = df.corr()
#     pval = df.corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(*rho.shape)
#     p = pval.applymap(lambda x: ''.join(['*' for t in [.05, .01, .001] if x<=t]))
#     return rho.round(2).astype(str) + p
#
# tmp = dfs.div(dfs.sum(axis=1), axis=0)
# tmp = tmp.transpose()[(tmp.var() > tmp.var().quantile(0.80))].transpose()
# tmp = pd.concat([tmp, doc_classes], axis=1, join='inner')
#
# tmp_corr = calc_signi(tmp)
# for topic in 'topic_business,topic_entertainment,topic_politics,topic_sport,topic_tech'.split(','):
#     print(tmp_corr[topic].sort_values(ascending=False)[:20])


# def get_training_data(df):
#     dfs_feat = []
#     for corpus in ['pdtb3', 'essay', 'ted', 'unsc', 'bbc', 'anthology']:
#         # for corpus in ['pdtb3', 'essay', 'ted']:
#         df1 = load_dataframe(corpus, 'explicit')
#         df2 = load_dataframe(corpus, 'altlex')
#         df = pd.concat([df1, df2])
#         df_dist = get_flow_counts(df[df.sense2_prob > 0.7])
#         df_dist = df_dist.div(df_dist.sum(axis=1), axis=0)
#         dfs_feat.append(df_dist)
#     dfs_feat = pd.concat(dfs_feat).fillna(0.0)
#
#     for index, item in dfs_feat.iterrows():
#         print(index, item.to_numpy())
#         break

def get_relation_type_results(relation_type, sense_level):
    dfs = []
    for path in glob.glob(f'results/v2/*.{relation_type}.csv'):
        df = pd.read_csv(path, names=column_names)
        df_dist = get_sense_distributions(df, normalize=False, sense_level=f'sense{sense_level}')
        #         df_dist = aggregate_dists(df_dist)
        df_dist['corpus'] = path[len('results/v2/'):].split('.')[0]
        dfs.append(df_dist)
    # dfs = pd.concat(dfs)
    # return dfs.pivot_table(index='corpus', columns=dfs.index).transpose().fillna(0.0) * 100
    return dfs


def get_overall_counts(relation_type, sense_level):
    dfs = get_relation_type_results(relation_type, sense_level)
    df_counts = pd.concat(dfs).fillna(0).groupby(level=0).sum()
    df_counts = (df_counts.div(df_counts.sum(axis=1), axis=0) * 100).transpose().round(2)
    df_counts = df_counts.sort_index()
    print(df_counts.to_latex())
    return df_counts


# def train_model(dfs):
#     X, y = [], []
#     for (corpus, doc_id), item in dfs.iterrows():
#         X.append(item.to_numpy())
#         y.append(corpus)
#     X = np.stack(X)
#     y = np.stack(y)
#
#     x_train, x_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)
#
#     clf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', max_samples=0.8, n_jobs=-1)
#     # clf = DecisionTreeClassifier()
#     clf.fit(x_train, y_train)
#
#     y_test_pred = clf.predict(x_test)
#     print(classification_report(y_test, y_test_pred))
#
#     clf_dummy = DummyClassifier(strategy="stratified")
#     clf_dummy.fit(x_train, y_train)
#     y_test_dummy = clf_dummy.predict(x_test)
#     print(classification_report(y_test, y_test_dummy))

def print_metrics_results(results):
    for key, vals in results.items():
        if key == 'accuracy':
            print(f"{key:32}  {vals * 100:>6.2f}")
        else:
            print(
                f"{key:32}  "
                f"{vals['precision'] * 100:>6.2f}  "
                f"{vals['recall'] * 100:>6.2f}  "
                f"{vals['f1-score'] * 100:>6.2f}  "
                f"{vals['support']:>5d}")
    print('## ' + '= ' * 50)


def print_final_results(loss, results):
    print("\n===")
    print(f'=== Final Validation Score: {loss}')
    print(f'=== Final Validation Macro AVG: {results.get("macro avg")}')
    print(f'=== Final Validation Weighted AVG: {results.get("weighted avg")}')
    print("===")


from helpers.connlex import single_connectives, multi_connectives_first, multi_connectives, distant_connectives


def get_connective_candidates(tokens: List[str]):
    candidates = []
    sentence = [w.lower().strip("'") for w in tokens]
    for word_idx, word in enumerate(sentence):
        for conn in distant_connectives:
            if word == conn[0]:
                if all(c in sentence for c in conn[1:]):
                    candidate = [(word_idx, conn[0])]
                    i = word_idx
                    ci = 1
                    while i < len(sentence):
                        if ci >= len(conn):
                            return candidates
                        c = conn[ci]
                        w = sentence[i]
                        if c == w:
                            candidate.append((i, c))
                            ci += 1
                        i += 1
                    if ci == len(conn):
                        candidates.append(candidate)
        if word in multi_connectives_first:
            for multi_conn in multi_connectives:
                if (word_idx + len(multi_conn)) <= len(sentence) and all(
                        c == sentence[word_idx + i] for i, c in enumerate(multi_conn)):
                    candidates.append([(word_idx + i, c) for i, c in enumerate(multi_conn)])
        if word in single_connectives:
            candidates.append([(word_idx, word)])
    return candidates


def get_linear_feature_importance(linear_clf, feature_names, top_k=20, save_path=''):
    for c_i, c in enumerate(linear_clf.classes_):
        feature_importance = pd.DataFrame(feature_names, columns=["feature"])
        feature_importance["importance"] = pow(math.e, linear_clf.coef_[c_i])
        feature_importance = feature_importance.sort_values(by=["importance"], ascending=False)
        ax = feature_importance[:top_k].plot.barh(x='feature', y='importance', figsize=(8, 5))
        # plt.title(c)
        print(c)
        plt.tight_layout()
        if save_path:
            plt.savefig(f'plots/{save_path}_feat_{c}.pdf')
        plt.show()


def get_features(dfs_topics, test_size=0.2, corpus_predictions=False):
    X, y = [], []

    for (corpus, doc_id), item in dfs_topics.iterrows():
        if corpus_predictions:
            X.append(item.to_numpy())
            y.append(corpus)
        else:
            X.append(item[:-1].to_numpy())
            y.append(item[-1])
    X = np.stack(X)
    y = np.stack(y)
    # print(np.unique(y, return_counts=True))
    # print(X, y)

    transform = TfidfTransformer()
    X_transformed = transform.fit_transform(X)

    x_train, x_test, y_train, y_test = train_test_split(X_transformed, y, stratify=y, test_size=test_size)

    return x_train, y_train, x_test, y_test


def plot_confusion_senseflows(clf, x_test, y_test, save_path='', labels=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    y_test_pred = clf.predict(x_test)
    cm = ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred, ax=ax, normalize='true', colorbar=False,
                                                 cmap='YlGnBu',
                                                 values_format='.2f',
                                                 labels=labels)
    # plt.tight_layout()
    if save_path:
        plt.savefig(f'plots/clf-senseflow-{save_path}.pdf')
    plt.show()


from sklearn.linear_model import LogisticRegression


def get_models(x_train, y_train, x_test, y_test):
    clf = LogisticRegression(class_weight='balanced', n_jobs=-1, max_iter=250, fit_intercept=False)
    clf.fit(x_train, y_train)
    clf_dummy = DummyClassifier(strategy="stratified")
    clf_dummy.fit(x_train, y_train)
    return clf, clf_dummy


def evaluate_model(clf, x, y, print_results=True):
    y_pred = clf.predict(x)
    if print_results:
        print(classification_report(y, y_pred))
    domain_genre_map = {
        'AES': 'Essay', 'PEC': 'Essay', 'ACL': 'Abstract', 'MED': 'Abstract', 'BBC': 'News', 'NYT': 'News',
        'WSJ': 'News', 'TED': 'Speech', 'UN': 'Speech'
    }
    domain_genre_map = np.vectorize(domain_genre_map.get)
    pred_genre = domain_genre_map(y_pred)
    y_genre = domain_genre_map(y)
    if print_results:
        print(classification_report(y_genre, pred_genre))
    return f1_score(y, y_pred, average='macro'), f1_score(y_genre, pred_genre, average='macro')


def evaluate_model_topic(clf, x, y, print_results=True):
    y_pred = clf.predict(x)
    if print_results:
        print(classification_report(y, y_pred))
    return f1_score(y, y_pred, average='macro')

