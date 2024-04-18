import numpy as np
import pandas as pd
import os
import argparse
from fastdist import fastdist
import pickle
from tqdm import tqdm
import torch
import collections as col
from collapse.data import SiteDataset, SiteNNDataset
from torch_geometric.loader import DataLoader
from utils import generate_struct_to_seq_map
from esm import pretrained
from transformers import AutoTokenizer, AutoModel
from modeling_esm import ESM_PLM
import urllib
parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str)
parser.add_argument('embedding_outfile', type=str)
parser.add_argument('funcsets_outfile', type=str)
parser.add_argument('--source', type=str, default='M-CSA')
parser.add_argument('--queue_size', type=int, default=1024)
parser.add_argument('--pdb_dir', type=str, default='/scratch/users/aderry/pdb')
parser.add_argument('--use_neighbors', action='store_true')
parser.add_argument('--checkpoint', type=str, default='best_demo_1702870147.059707_checkpoints.pth.tar')
parser.add_argument('--model', type=str, default='MoCo', choices=['MoCo', 'SimSiam', "MoCo_positive_only", "Triplet"])
parser.add_argument('--esm_checkpoint', type=str, default=None)
parser.add_argument('--esm_model', type=str, default="facebook/esm2_t12_35M_UR50D")
args = parser.parse_args()

# os.makedirs(args.outfile, exist_ok=True)

torch.backends.cudnn.benchmark = False
# deteministic
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

src_dataset = pd.read_csv(args.dataset, converters={'locs': lambda x: eval(x)})
src_dataset = src_dataset[src_dataset['source'] == args.source]

if args.use_neighbors:
    dataset = SiteNNDataset(src_dataset, args.pdb_dir, train_mode=False)
else:
    dataset = SiteDataset(src_dataset, args.pdb_dir, train_mode=False)
loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

# model = initialize_model(device=device)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

print('Computing embeddings...')
all_emb = []
prosite_labels = []
all_pdb = []
all_sites = []
all_sources = []
all_resids = []

mapping = pd.read_csv("pdb_chain_uniprot.csv", header=0)
current_pdb = None
current_chain = None
embeddings = None
sequence_embedding = None
of = open("mcsa_to_clean.csv", "w")

of.write("Entry	EC number	Sequence\n")
import os
import urllib
import requests as r
from Bio import SeqIO
from io import StringIO
from collapse.data import process_pdb, atom_info

baseUrl="http://www.uniprot.org/uniprot/"

from tqdm import tqdm
with torch.no_grad():
    for g, pdb, source, desc in tqdm(loader):
        g = g.to(device)

        pdb_id = pdb[0][:4]
        chain = pdb[0][4:]
        if not (pdb_id == current_pdb and current_chain == chain):
            filtered = mapping[(mapping['PDB'] == pdb_id) & (mapping['CHAIN'] == chain)]
        
        for i in range(min(filtered.shape[0], 1)):
            row = filtered.iloc[i]
            resid = int(g.resid[0][1:])
                
            atom_df = process_pdb(os.path.join(args.pdb_dir, pdb_id+'.pdb'), chain, False)
            if True:
                if pdb_id == current_pdb and current_chain == chain:
                    pass
                else:
                    # print(row['SP_PRIMARY'])
                    if not os.path.exists(f"fasta_cache/{row['SP_PRIMARY']}.fasta"):
                        currentUrl=baseUrl + row['SP_PRIMARY'] + ".fasta"
                        # currentUrl=baseUrl + pdb_id.split("_")[1] + ".fasta"
                        response = r.post(currentUrl)
                        cData=''.join(response.text)

                        Seq=StringIO(cData)
                        pSeq=SeqIO.parse(Seq, 'fasta')
                        pSeq = list(pSeq)
                        pSeq = str(pSeq[0].seq)
                        with open(f"fasta_cache/{row['SP_PRIMARY']}.fasta", "w") as f:
                            f.write(pSeq)
                    else:
                        with open(f"fasta_cache/{row['SP_PRIMARY']}.fasta", "r") as f:
                            pSeq = f.read()
                    
                    of.write(f"{row['SP_PRIMARY']}	{pdb_id}	{pSeq}\n")
                    current_pdb = pdb_id
                    current_chain = chain

of.close()