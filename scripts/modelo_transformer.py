import re
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
import torch
from .preprocessamento import tags_validas_adaptadas

label_list = ['O'] + [f'B-{lbl}' for lbl in tags_validas_adaptadas] + [f'I-{lbl}' for lbl in tags_validas_adaptadas]
label_to_id = {label: i for i, label in enumerate(label_list)}
id_to_label = {i: label for i, label in enumerate(label_list)}

def process_text_transformer(text, entities):
    entities = sorted(entities, key=lambda e: e['start'], reverse=True)
    for ent in entities:
        tag = f"[{ent['entity_group']}]"
        text = text[:ent['start']] + tag + text[ent['end']:]
    return text

def main():
    pass

if __name__ == '__main__':
    main()