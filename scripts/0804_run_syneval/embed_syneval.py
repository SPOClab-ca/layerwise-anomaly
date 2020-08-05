import argparse
import numpy as np 
import os, sys, time 
import pickle 
import scipy
import sklearn
from sklearn.decomposition import PCA
import torch 
from utils import timed_func

import transformers
from transformers import BertModel, BertTokenizer


lm_syneval_dir = "../../data/LM_syneval/data"
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
bert_model = BertModel.from_pretrained('bert-base-cased', output_hidden_states=False)


def evaluate_contextual_diff(pair):
    source, target = pair[0], pair[1]
    src_ids = torch.tensor(bert_tokenizer.encode(source)).unsqueeze(0)
    src_vec = bert_model(src_ids)[0].mean(dim=1)[0]  # (768,) torch.tensor
    
    tgt_ids = torch.tensor(bert_tokenizer.encode(target)).unsqueeze(0)
    tgt_vec = bert_model(tgt_ids)[0].mean(dim=1)[0]
    
    d_emb = len(src_vec)  # 768
    diff = (src_vec - tgt_vec).detach()
    return diff  # 768-dimensional np.array


@timed_func
def get_syneval_data():
    syneval_names = [
        "vp_coord", "subj_rel", "simple_reflexives", 
        "simple_npi_inanim", "simple_npi_anim",
        "simple_agrmt", "sent_comp", "reflexives_across",
        "reflexive_sent_comp", "prep_inanim", "prep_anim",
        "obj_rel_within_inanim", "obj_rel_within_anim",
        "obj_rel_no_comp_within_inanim", "obj_rel_no_comp_within_anim",
        "obj_rel_no_comp_across_inanim", "obj_rel_no_comp_across_anim",
        "obj_rel_across_inanim", "obj_rel_across_anim",
        "npi_across_inanim", "npi_across_anim",
        "long_vp_coord"
    ]

    syneval_data_2 = {}
    syneval_data_multiple = {}

    for name in syneval_names:
        with open(os.path.join(lm_syneval_dir, "templates/{}.pickle".format(name)), "rb") as f:
            data = pickle.load(f)
            
            clean_data = {}  # tense -> list of str
            for tense in data.keys():
                clean_data[tense] = []
                for sents in data[tense]:
                    correct = sents[0]
                    clean_data[tense].append(correct)

            if len(data.keys()) == 2:
                syneval_data_2[name] = clean_data 
            else:
                print ("\t category {} has {} tenses".format(name, len(data.keys())))
                syneval_data_multiple[name] = clean_data 
            
    print ("2-tense categories: {}. Total: {}".format(syneval_data_2.keys(), len(syneval_data_2)))
    return syneval_data_2, syneval_data_multiple 



def name2vecs(syneval_data_2, args):
    if not os.path.exists(args.export):
        os.makedirs(args.export)

    for name in syneval_data_2:
        start_time = time.time()
        data = syneval_data_2[name]
        k1, k2 = list(data.keys())
        diffs = []
        for s1, s2 in zip(data[k1], data[k2]):
            diffs.append(evaluate_contextual_diff([s1, s2]))
        diffs = torch.stack(diffs, dim=0)  # (N, 784)
        
        fname = os.path.join(args.export, name + ".pkl")
        with open(fname, "wb+") as f:
            pickle.dump(diffs, f)

        print ("name2vec {} done in {:.2f} seconds".format(name, time.time() - start_time))
        

def pca_to_embed(args):
    all_vecs = []
    for fname in os.listdir(args.export):
        if fname.startswith("pcamodel"):
            continue
        with open(os.path.join(args.export, fname), "rb") as f:
            diffs = pickle.load(f)
            all_vecs.append(diffs)
    
    pca = PCA(n_components=2)
    X = torch.cat(all_vecs, dim=0).detach().numpy()
    pca.fit(X)
    with open(os.path.join(args.export, "pcamodel.pkl"), "wb+") as f:
        pickle.dump(pca, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", type=str, default="0804_embedded")
    args = parser.parse_args()
    print(args)

    syneval_data_2, syneval_data_multiple = get_syneval_data()

    name2vecs(syneval_data_2, args)
    pca_to_embed(args)

main()