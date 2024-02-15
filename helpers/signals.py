import glob
import sys

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from helpers.crf import DiscourseSignalExtractor
from helpers.data import get_doc_embeddings, get_paragraph_embeddings
from helpers.senses import get_bert_features, DiscourseSenseClassifier, DiscourseSenseEnsembleClassifier


class DiscourseSignalModel:
    def __init__(self, tokenizer, signal_model, sense_model_embed, sense_model, no_none_class=False, device='cpu'):
        self.tokenizer = tokenizer
        self.signal_model: DiscourseSignalExtractor = signal_model
        self.sense_model_embed = sense_model_embed
        self.sense_model = sense_model
        self.no_none_class = no_none_class
        self.device = device

    @staticmethod
    def load_model(save_path, relation_type, no_none_class=False, device='cpu', use_crf=True):
        tokenizer = AutoTokenizer.from_pretrained("roberta-base", add_prefix_space=True, local_files_only=True)
        signal_model = DiscourseSignalExtractor.load_model(save_path, relation_type, device=device, use_crf=use_crf)

        save_paths = glob.glob(save_path)
        sense_model_embed = AutoModel.from_pretrained("roberta-base", local_files_only=True)
        if len(save_paths) == 1:
            sense_model = DiscourseSenseClassifier.load(save_paths[0], device, relation_type='both')
        else:
            sense_model = DiscourseSenseEnsembleClassifier.load(save_paths, device, relation_type='both')
        sense_model_embed.to(device)
        sense_model.to(device)
        return DiscourseSignalModel(tokenizer, signal_model, sense_model_embed, sense_model, no_none_class, device)

    def predict(self, doc):
        try:
            sentence_embeddings = get_doc_embeddings(doc, self.tokenizer, self.sense_model_embed, device=self.device)
        except RuntimeError as e:
            sys.stderr.write(f"Error {doc.doc_id}: {e}")
            return []

        doc_signals = self.signal_model.predict(doc)
        for par_i, paragraph in enumerate(doc_signals):
            if paragraph['relations']:
                features = np.stack([get_bert_features(signal['tokens_idx'], sentence_embeddings,
                                                       used_context=self.sense_model.used_context)
                                     for signal in paragraph['relations']])
                features = torch.from_numpy(features).to(self.device)
                pred = self.sense_model.predict(features, no_none=self.no_none_class)

                relations = []
                for signal, coarse_class_i, coarse_class_i_prob, fine_class_i, fine_class_i_prob, coarse_class_i_prob_all in zip(
                        paragraph['relations'],
                        pred['coarse'], pred['probs_coarse'], pred['fine'], pred['probs_fine'],
                        pred['probs_coarse_all']):
                    relations.append({
                        'tokens_idx': signal['tokens_idx'],
                        'tokens': signal['tokens'],
                        'coarse': coarse_class_i,
                        'coarse_probs': round(coarse_class_i_prob, 4),
                        'fine': fine_class_i,
                        'fine_probs': round(fine_class_i_prob, 4),
                        # 'is_relation': (1.0 - coarse_class_i_prob_all[self.sense_model.label2id_coarse['None']]),
                    })
                relations = [r for r in relations if r['coarse'] != 'None']
                doc_signals[par_i]['relations'] = relations
        return doc_signals

    def predict_paragraphs(self, paragraphs, is_relation_threshold=0.4):
        try:
            paragraph_embeddings = get_paragraph_embeddings(paragraphs, self.tokenizer, self.sense_model_embed,
                                                            device=self.device)
        except RuntimeError as e:
            sys.stderr.write(f"Error: {e}")
            return []

        doc_signals = self.signal_model.predict_paragraphs(paragraphs)
        for par_i, paragraph in enumerate(doc_signals):
            if paragraph['relations']:
                par_offset = paragraph['tokens_idx'][0]
                features = np.stack([get_bert_features([i - par_offset for i in signal['tokens_idx']],
                                                       paragraph_embeddings[par_i], self.sense_model.used_context)
                                     for signal in paragraph['relations']])
                features = torch.from_numpy(features).to(self.device)
                pred = self.sense_model.predict(features, no_none=self.no_none_class)

                relations = []
                for signal, coarse_class_i, coarse_class_i_prob, fine_class_i, fine_class_i_prob, coarse_class_i_prob_all in zip(
                        paragraph['relations'],
                        pred['coarse'], pred['probs_coarse'], pred['fine'], pred['probs_fine'],
                        pred['probs_coarse_all']):
                    relations.append({
                        'tokens_idx': signal['tokens_idx'],
                        'tokens': signal['tokens'],
                        'sentence_idx': signal['sentence_idx'],
                        'coarse': coarse_class_i,
                        'coarse_probs': round(coarse_class_i_prob, 4),
                        'fine': fine_class_i,
                        'fine_probs': round(fine_class_i_prob, 4),
                    })
                paragraph['relations'] = relations
                yield paragraph
