import sklearn.mixture
import numpy as np
import torch
import sys

from utils import timed_func


# Load BEA sentences to train GMM
@timed_func
def parse_grammar_dataset():
    m2_path = "../data/BEA14/WI_LOCNESS/wi+locness/m2/A.train.gold.bea19.m2"
    with open(m2_path, "r") as f:
        raw_lines = f.readlines()

    data = []
    curr_item = {}
    for line in raw_lines:
        if line.startswith("S "):
            if len(curr_item) > 0 and len(curr_item["errors"]) > 0:
                curr_item["corrected_sent"] = _correct_sentence(curr_item)
                data.append(curr_item)
            curr_item = {"sentence": line[2:].strip(), "errors": []}
        elif line.startswith("A "):
            items = line.split("|||")
            start_pos_str, end_pos_str = items[0][2:].split()
            start_pos, end_pos = int(start_pos_str), int(end_pos_str)
            error = {
                "start_pos": start_pos,  # These are in terms of word locs
                "end_pos": end_pos,
                "correct": items[2] 
            }
            curr_item["errors"].append(error)

    print("Collected {} sentences".format(len(data)))
    sentences = [item['sentence'] for item in data]
    return data, sentences 


def _correct_sentence(item):
    """
    E.g., sent is "My town is a medium size city ..."
    errors is [{'start_pos': 5, 'end_pos': 6, 'correct': '- sized'}]
    Assume errors are sorted by start_pos.
    """
    sent = item['sentence']
    L = [w for w in sent.split()]
    newline = []
    i = 0
    for err in item['errors']:
        if err['start_pos'] == -1:  # This sentence is correct
            return sent
        newline.extend(L[i:err['start_pos']])
        newline.append(err['correct'])
        i = err['end_pos']
    newline.extend(L[i:])
    return _remove_repeated_spaces(" ".join(newline))


def _remove_repeated_spaces(s):
    return " ".join(s.split())
    

