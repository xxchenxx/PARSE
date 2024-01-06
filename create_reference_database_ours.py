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
from collapse import initialize_model, atom_info
from esm import pretrained

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str)
parser.add_argument('embedding_outfile', type=str)
parser.add_argument('funcsets_outfile', type=str)
parser.add_argument('--source', type=str, default='M-CSA')
parser.add_argument('--pdb_dir', type=str, default='/scratch/users/aderry/pdb')
parser.add_argument('--use_neighbors', action='store_true')
args = parser.parse_args()

# os.makedirs(args.outfile, exist_ok=True)

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
esm_model, alphabet = pretrained.load_model_and_alphabet('esm2_t12_35M_UR50D')
esm_model = esm_model.to('cuda')
batch_converter = alphabet.get_batch_converter()
esm_model.eval()

from CLEAN.model import MoCo

model = MoCo(512, 128, torch.device('cuda'), torch.float, esm_model_dim=480).cuda()
model.load_state_dict(torch.load("best_demo_1702870147.059707_checkpoints.pth.tar")['model_state_dict'])
import os
import urllib
import requests as r
from Bio import SeqIO
from io import StringIO


baseUrl="http://www.uniprot.org/uniprot/"

from tqdm import tqdm
with torch.no_grad():
    for g, pdb, source, desc in tqdm(loader):
        g = g.to(device)

        pdb_id = pdb[0][:4]
        chain = pdb[0][4:]
        # print(pdb_id, chain)
        filtered = mapping[(mapping['PDB'] == pdb_id) & (mapping['CHAIN'] == chain)]
    
        for i in range(filtered.shape[0]):
            row = filtered.iloc[i]
            resid = int(g.resid[0][1:])
            # print(g.resid[0], row['RES_BEG'], row['RES_END'])
            # print(resid, row['RES_BEG'], row['RES_END'])
            if resid >= row['RES_BEG'] and resid <= row['RES_END']:
                index = resid - row['RES_BEG'] + row['SP_BEG'] - 1
                # print(index) - 1
                if pdb_id == current_pdb and current_chain == chain:
                    embeddings = sequence_embedding[index]
                else:
                    currentUrl=baseUrl + row['SP_PRIMARY'] + ".fasta"
                    response = r.post(currentUrl)
                    cData=''.join(response.text)

                    Seq=StringIO(cData)
                    pSeq=SeqIO.parse(Seq, 'fasta')
                    pSeq = list(pSeq)
                    pSeq = str(pSeq[0].seq)
                    d = [
                        ("protein1", pSeq),
                    ]
                    batch_labels, batch_strs, batch_tokens = batch_converter(d)
                    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)    
                    with torch.no_grad():
                        batch_tokens = batch_tokens.cuda()
                        results = esm_model(batch_tokens, repr_layers=[12], return_contacts=False)
                    token_representations = results["representations"][12][:,1:-1]
                    sequence_embedding = model.encoder_q(token_representations)[0]
                    embeddings = sequence_embedding[index]
                    current_pdb = pdb_id
                    current_chain = chain

        # embeddings, _ = model.online_encoder(g, return_projection=False)
        all_emb.append(embeddings.squeeze().cpu().numpy())
        all_pdb.append(pdb[0])
        all_sites.append(desc[0])
        all_sources.append(source[0])
        all_resids.append(g.resid[0])
     
print('Saving...')
all_emb = np.stack(all_emb)
outdata = {'embeddings': all_emb.copy(), 'pdbs': all_pdb, 'resids': all_resids, 'sites': all_sites, 'sources': all_sources}
pdb_resids = [x+'_'+y for x,y in zip(all_pdb, all_resids)]

with open(args.embedding_outfile, 'wb') as f:
    pickle.dump(outdata, f)
    
fn_lists = col.defaultdict(set)
for fn, site in zip(all_sites, pdb_resids):
    fn_lists[f'{fn}'].add(str(site))
with open(args.funcsets_outfile, 'wb') as f:
    pickle.dump({k: list(v) for k,v in fn_lists.items()}, f)
