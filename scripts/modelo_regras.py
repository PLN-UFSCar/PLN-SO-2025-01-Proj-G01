import pandas as pd
import pickle
import os
import sys
from typing import Union, Dict, Any
from faker import Faker
import re
import numpy as np
from seqeval.metrics import classification_report as seqeval_classification_report
from seqeval.metrics import f1_score as seqeval_f1
from seqeval.metrics import accuracy_score as seqeval_accuracy
from sklearn.metrics import classification_report as sklearn_classification_report
import spacy
from spacy.language import Language
from spacy.matcher import Matcher
from spacy.tokens import Span
import re

def carregar_senhas_rockyou(arquivo_path="rockyou.txt", max_senhas=10000):
    senhas = set()
    try:
        with open(arquivo_path, 'r', encoding='utf-8', errors='ignore') as f:
            for i, linha in enumerate(f):
                if i >= max_senhas:
                    break
                senha = linha.strip()
                if len(senha) >= 4 and not senha.isdigit():
                    senhas.add(senha)
    except FileNotFoundError:
        print(f"Arquivo {arquivo_path} não encontrado. Usando senhas comuns padrão.")
        senhas = {
            "password", "123456789", "qwerty", "abc123", "Password", "password123",
            "admin", "letmein", "welcome", "monkey", "dragon", "master",
            "sunshine", "princess", "football", "baseball", "shadow",
            "superman", "michael", "jessica", "charlie", "jordan"
        }
    return senhas

def make_regex_component(label, pattern):
    @Language.component(f"regex_{label.lower()}_detector")
    def regex_component(doc):
        regex = re.compile(pattern)
        new_spans = []

        for match in regex.finditer(doc.text):
            span = doc.char_span(*match.span(), label=label, alignment_mode="contract")
            if span:
                new_spans.append(span)

        all_spans = list(doc.ents) + new_spans

        def has_overlap(span1, span2):
            return span1.start < span2.end and span2.start < span1.end

        filtered = []
        for span in sorted(all_spans, key=lambda s: (s.start, -s.end)):
            if not any(has_overlap(span, s) for s in filtered):
                filtered.append(span)

        doc.ents = filtered
        return doc

    return f"regex_{label.lower()}_detector"

def get_bio_tags_regex(row, tokenizer):
    text = row["source_text_pt"]
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
    )
    offset_mapping = encoding['offset_mapping']
    bio_tags = ['O'] * len(offset_mapping)
    
    char_spans = row["spans_from_source_pt"]
    for start_char, end_char, label in char_spans:
        is_first_token = True
        for i, (token_start, token_end) in enumerate(offset_mapping):
            if token_start == token_end == 0:
                continue

            if token_start >= start_char and token_end <= end_char:
                if is_first_token:
                    bio_tags[i] = f'B-{label}'
                    is_first_token = False
                else:
                    bio_tags[i] = f'I-{label}'

    return bio_tags[1:-1]

def calculate_metrics_regex(true_labels_col, pred_labels_col):
    true_labels = true_labels_col.tolist()
    true_predictions = pred_labels_col.tolist()

    entity_report = seqeval_classification_report(
        true_labels, true_predictions, output_dict=True, zero_division=0
    )

    flat_true = [tag for seq in true_labels for tag in seq]
    flat_pred = [tag for seq in true_predictions for tag in seq]
    
    token_report = sklearn_classification_report(
        flat_true, flat_pred, output_dict=True, zero_division=0
    )

    results = {
        "eval_entity_f1": seqeval_f1(true_labels, true_predictions),
        "eval_entity_accuracy": seqeval_accuracy(true_labels, true_predictions),
        "eval_entity_precision": entity_report["macro avg"]["precision"],
        "eval_entity_recall": entity_report["macro avg"]["recall"],
        "eval_token_f1": token_report["weighted avg"]["f1-score"],
        "eval_token_accuracy": token_report["accuracy"],
        "eval_token_precision": token_report["weighted avg"]["precision"],
        "eval_token_recall": token_report["weighted avg"]["recall"],
    }
    
    return results

def process_text_regex(text, spans):
    if not spans:
        return text
    
    spans = sorted(spans, key=lambda s: s[0], reverse=True)
    
    for start, end, label in spans:
        text = text[:start] + f"[{label}]" + text[end:]
    return text

def main():
    pass

if __name__ == '__main__':
    main()