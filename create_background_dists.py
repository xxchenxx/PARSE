import os
import time
import torch
import parse
import utils
import urllib
import pickle
import argparse
import numpy as np
import pandas as pd
import requests as r
import collections as col
from tqdm import tqdm
from Bio import SeqIO
from io import StringIO
from esm import pretrained
from CLEAN.model import MoCo
from fastdist import fastdist
from torch_geometric.loader import DataLoader
from collapse import initialize_model, atom_info
from collapse.data import SiteDataset, SiteNNDataset

model = MoCo(512, 128, torch.device('cuda'), torch.float, esm_model_dim=480).cuda()
model.load_state_dict(torch.load("best_demo_1702870147.059707_checkpoints.pth.tar")['model_state_dict'])
parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str)
parser.add_argument('--source', type=str, default='M-CSA')
parser.add_argument('--pdb_dir', type=str, default='/scratch/users/aderry/pdb')
parser.add_argument('--use_neighbors', action='store_true')
parser.add_argument('--db', type=str, default='./data/csa_site_db_nn.pkl', help='Reference database embeddings')
parser.add_argument('--function_sets', type=str, default='./data/csa_function_sets_nn.pkl', help='Reference function sets')
parser.add_argument('--cutoff', type=float, default=0.001, help='FDR cutoff for reporting results')
parser.add_argument('--use_gpu', action='store_true', help='Use GPU to embed proteins')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

src_dataset = pd.read_csv(args.dataset, converters={'locs': lambda x: eval(x)})

all_emb = []

mapping = pd.read_csv("pdb_chain_uniprot.csv", header=0)
esm_model, alphabet = pretrained.load_model_and_alphabet('esm2_t12_35M_UR50D')
esm_model = esm_model.to('cuda')
batch_converter = alphabet.get_batch_converter()
esm_model.eval()

start = time.time()
device = 'cuda' if args.use_gpu else 'cpu'

db = utils.deserialize(args.db)
function_sets = utils.deserialize(args.function_sets)

def read_fasta_to_dict(fasta_file):
    sequence_dict = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        uniprot_id = record.description.split('|')[1]
        sequence_dict[uniprot_id] = str(record.seq)
    return sequence_dict

fasta_file = 'uniprot_sprot.fasta'
fasta_dict = read_fasta_to_dict(fasta_file)
seen = set()
background_dists = {}
with torch.no_grad():
    for i in range(len(src_dataset)):
        print(i)
        row = src_dataset.iloc[i]
        if row['uniprot'] not in fasta_dict or row['uniprot'] in seen:
            continue
        d = [
            ("protein1", fasta_dict[row['uniprot']])
        ]            
        batch_labels, batch_strs, batch_tokens = batch_converter(d)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)    
        with torch.no_grad():
            batch_tokens = batch_tokens.cuda()
            results = esm_model(batch_tokens, repr_layers=[12], return_contacts=False)
        token_representations = results["representations"][12][:,1:-1]
        sequence_embedding = model.encoder_q(token_representations)[0]
        seen.add(row['uniprot'])
        rnk = parse.compute_rank_df_for_background(sequence_embedding.squeeze().cpu().numpy(), db)
        in_df = rnk[['site', 'score']]
        in_df.columns = [0, 1]
        result = parse.enrichment(in_df, function_sets)
        for j in range(len(result)):
            row = result.iloc[j]
            if row['function'] in background_dists:
                background_dists[row['function']].append(row['score'])
            else:
                background_dists[row['function']] = [row['score']]
print('Saving...')
with open('function_score_dists_temp_2.pkl', 'wb') as f:
    pickle.dump(background_dists, f)
