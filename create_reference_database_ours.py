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
if args.esm_checkpoint is None:
    esm_model, alphabet = pretrained.load_model_and_alphabet('esm2_t12_35M_UR50D')
    esm_model = esm_model.to('cuda')
    batch_converter = alphabet.get_batch_converter()
    esm_model.eval()
    is_lora_esm = False
else:
    esm_model = ESM_PLM('', esm_checkpoint=args.esm_checkpoint, num_params="official_35m")
    tokenizer = AutoTokenizer.from_pretrained(args.esm_model)
    is_lora_esm = True
    esm_model = esm_model.to('cuda')
    esm_model.eval()
from CLEAN.model import MoCo, MoCo_positive_only, LayerNormNet
from CLEAN.simsiam import SimSiam

if args.model == 'MoCo':
    model = MoCo(512, 128, torch.device('cuda'), torch.float, esm_model_dim=480, queue_size=args.queue_size).cuda()
elif args.model == 'SimSiam':
    model = SimSiam(512, 128, torch.device('cuda'), torch.float, esm_model_dim=480, queue_size=args.queue_size).cuda()
elif args.model == 'MoCo_positive_only':
    model = MoCo_positive_only(512, 128, torch.device('cuda'), torch.float, esm_model_dim=480, queue_size=args.queue_size).cuda()
elif args.model == 'Triplet':
    model = LayerNormNet(512, 128, torch.device('cuda'), torch.float, esm_model_dim=480).cuda()
try:
    model.load_state_dict(torch.load(args.checkpoint)['model_state_dict'])
except:
    model.load_state_dict(torch.load(args.checkpoint))

model.eval()
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
            pdb_sequence = []
            count = 0
            current_resid = None
            resid_to_index = {}
            for i in range(len(atom_df)):
                if atom_df.iloc[i]['residue'] != current_resid and atom_df.iloc[i]['chain'] == chain:
                    pdb_sequence.append(atom_info.aa_to_letter(atom_df.iloc[i]['resname']))
                    # print(pdb_sequence)
                    current_resid = atom_df.iloc[i]['residue']
                    resid_to_index[current_resid] = count + 1 # 1-indexed
                    count += 1
            pdb_sequence = ''.join(pdb_sequence)
            # print(g.resid[0], row['RES_BEG'], row['RES_END'])
            # print(resid, row['RES_BEG'], row['RES_END'])
            if True:
                # print(index) - 1
                # print(pdb_id)
                if pdb_id == current_pdb and current_chain == chain:
                    resid_index = resid_to_index[resid]
                    if resid_index not in reverse_structure_seq_map:
                        print("A mismatch is detected at PDB {} and chain {} and resid {}. ".format(pdb_id, chain, resid))
                        if resid_index - 1 in reverse_structure_seq_map:
                            index = reverse_structure_seq_map[resid_index - 1]
                            print("Using the left-side fix")
                        elif resid_index + 1 in reverse_structure_seq_map:
                            index = reverse_structure_seq_map[resid_index + 1] - 2
                            print("Using the right-side fix")

                    else:
                        index = reverse_structure_seq_map[resid_index] - 1
                    
                    embeddings = sequence_embedding[index]
                    current_resid = resid
                    current_chain = chain
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
                    try:
                        structure_seq_map = generate_struct_to_seq_map(pdb_sequence, pSeq, range(1, len(pSeq) + 1), range(1, len(pSeq) + 1))
                    except StopIteration:
                        # for some reasons we fail here
                        print("Warning: failed to generate structure to sequence map for PDB {} and chain {}.".format(pdb_id, chain))
                        pSeq = pdb_sequence
                        structure_seq_map = generate_struct_to_seq_map(pdb_sequence, pSeq, range(1, len(pSeq) + 1), range(1, len(pSeq) + 1))
                    reverse_structure_seq_map = {v: k for k, v in structure_seq_map.items()}
                    d = [
                        ("protein1", pSeq),
                    ]
                    if not is_lora_esm:
                        batch_labels, batch_strs, batch_tokens = batch_converter(d)
                        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)    
                        with torch.no_grad():
                            batch_tokens = batch_tokens.cuda()
                            results = esm_model(batch_tokens, repr_layers=[12], return_contacts=False)
                        token_representations = results["representations"][12][:,1:-1]
                    else:
                        inputs = tokenizer(pSeq, return_tensors="pt", padding="longest", truncation=True, max_length=2048)
                        inputs = {k: v.to('cuda') for k, v in inputs.items()}
                        with torch.no_grad():
                            outputs = esm_model(**inputs)
                        token_representations = outputs
                    if 'MoCo' in args.model:
                        sequence_embedding = model.encoder_q(token_representations)[0]
                    elif args.model == 'SimSiam':
                        # print(token_representations.shape)
                        sequence_embedding = model.projector(token_representations.squeeze(0))
                    elif args.model == 'Triplet':
                        sequence_embedding = model(token_representations)[0]
                    resid_index = resid_to_index[resid]
                    if resid_index not in reverse_structure_seq_map:
                        print("A mismatch is detected at PDB {} and chain {} and resid {}. Skip this residue.".format(pdb_id, chain, resid))
                        current_resid = resid
                        current_chain = chain
                        continue
                    else:
                        index = reverse_structure_seq_map[resid_index] - 1
                    if index >= len(sequence_embedding):
                        print("A mismatch is detected at PDB {} and chain {} and resid {}. Skip this residue.".format(pdb_id, chain, resid))
                        current_resid = resid
                        current_chain = chain
                        continue
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
