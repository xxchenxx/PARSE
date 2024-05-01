import argparse
import time
import utils
import collections as col
import pandas as pd
import torch
from esm import pretrained
import requests as r
from Bio import SeqIO
from io import StringIO
from utils import generate_struct_to_seq_map
import os
# from transformers import AutoTokenizer, AutoModel
# from modeling_esm import ESM_PLM
global esm_model, batch_converter, is_lora_esm, tokenizer
torch.backends.cudnn.benchmark = False
# deteministic
torch.backends.cudnn.deterministic = True
import gemmi
import numpy as np
# read and parse a CIF file


mapping = pd.read_csv("pdb_chain_uniprot.csv", header=0)
esm_model, alphabet = pretrained.load_model_and_alphabet('esm2_t12_35M_UR50D')
esm_model = esm_model.to('cuda')
batch_converter = alphabet.get_batch_converter()
esm_model.eval()

from tqdm import tqdm

three_to_one_mappings = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
     'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
     'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
     'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb', type=str, default=None, help='Input PDB file')
    parser.add_argument('--precomputed_id', type=str, default=None, help='ID for accessing precomputed embeddings')
    parser.add_argument('--precomputed_lmdb', type=str, default=None, help='Precomputed embeddings in LMDB format')
    parser.add_argument('--chain', type=str, default=None, help='Input PDB chain to annotate')
    parser.add_argument('--db', type=str, default='./data/csa_site_db_nn.pkl', help='Reference database embeddings')
    parser.add_argument('--function_sets', type=str, default='./data/csa_function_sets_nn.pkl', help='Reference function sets')
    parser.add_argument('--background', type=str, default='./data/function_score_dists.pkl', help='Function-specific background distributions')
    parser.add_argument('--cutoff', type=float, default=0.001, help='FDR cutoff for reporting results')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU to embed proteins')
    parser.add_argument('--checkpoint', type=str, default='best_demo_1702870147.059707_checkpoints.pth.tar')
    parser.add_argument('--model', type=str, default='MoCo', choices=['MoCo', 'SimSiam', "MoCo_positive_only", "Triplet"])
    parser.add_argument('--queue_size', type=int, default=1024)
    parser.add_argument('--prefix', type=str, default=None)
    parser.add_argument('--esm_checkpoint', type=str, default=None)
    parser.add_argument('--esm_model', type=str, default="facebook/esm2_t12_35M_UR50D")
    parser.add_argument('--pdb_dir', type=str, default="")
    parser.add_argument('--csv_file', type=str, default="")
    parser.add_argument('--output_dir', type=str, default=None)

    args = parser.parse_args()
    
    
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
    start = time.time()
    device = 'cuda' if args.use_gpu else 'cpu'
    
    db = utils.deserialize(args.db)
    function_sets = utils.deserialize(args.function_sets)
    background_dists = utils.deserialize(args.background)
    
    from CLEAN.model import MoCo, MoCo_positive_only, LayerNormNet
    # from CLEAN.simsiam import SimSiam

    if args.model == 'MoCo':
        model = MoCo(512, 128, torch.device('cuda'), torch.float, esm_model_dim=480, queue_size=args.queue_size).cuda()
    elif args.model == 'SimSiam':
        # model = SimSiam(512, 128, torch.device('cuda'), torch.float, esm_model_dim=480).cuda()
        pass
    elif args.model == 'MoCo_positive_only':
        model = MoCo_positive_only(512, 128, torch.device('cuda'), torch.float, esm_model_dim=480, queue_size=args.queue_size).cuda()
    elif args.model == 'Triplet':
        model = LayerNormNet(512, 128, torch.device('cuda'), torch.float, esm_model_dim=480).cuda()
    try:
        model.load_state_dict(torch.load(args.checkpoint)['model_state_dict'])
    except:
        model.load_state_dict(torch.load(args.checkpoint))
        
    model.eval()
    from glob import glob

    # doc = cif.read_file('components.cif')
    dataset = pd.read_csv(args.csv_file)
    current_chain = None
    current_rcsb_code = None
    embeddings = []
    
    for i in tqdm(range(len(dataset))):
        
        row = dataset.iloc[i]
        rcsb_code = row['rcsb_code']
        protein_auth_chain_id, protein_auth_seq_num = row['protein_auth_chain_id'], row['protein_auth_seq_num']
        protein_auth_res_name = row['protein_auth_res_name']
        protein_auth_seq_num = int(protein_auth_seq_num)
        # print(row)
        rcsb_code = rcsb_code.upper()
        
        if current_rcsb_code != rcsb_code or current_chain != protein_auth_chain_id:
            current_rcsb_code = rcsb_code
            current_chain = protein_auth_chain_id
            cif_path = f"{args.pdb_dir}/{rcsb_code}.cif"
            # print(cif_path)
            if not os.path.exists(cif_path):
                continue
            cif_structure = gemmi.read_structure(cif_path)
            for cif_model in cif_structure:
                for chain in cif_model:
                    if chain.name == protein_auth_chain_id:
                        break
                break

            # get the sequence
            positions = [res.seqid.num for res in chain]
            sequence = [three_to_one_mappings.get(res.name, "*") for res in chain]
            # three to one
            # print(row)
            # print(sequence)
            positions = [res for (res, seq) in zip(positions, sequence) if seq != "*"]
            sequence = [seq for seq in sequence if seq != "*"]
            sequence = "".join(sequence)
            # print(positions)
            index = positions.index(protein_auth_seq_num)
            # print(sequence[index])
            # print(protein_auth_res_name)
            
            d = [
                ("protein1", sequence), 
            ]
            
            batch_labels, batch_strs, batch_tokens = batch_converter(d)
            batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)    
            with torch.no_grad():
                batch_tokens = batch_tokens.cuda()
                results = esm_model(batch_tokens, repr_layers=[12], return_contacts=False)
            token_representations = results["representations"][12][:,1:-1]
            sequence_embedding = model.encoder_q(token_representations)[0].detach().cpu().numpy()
            embeddings.append(sequence_embedding)
        else:
            # print(row)
            # print(positions)
            pass
    
    # residue_embeddings = np.stack(residue_embeddings, 0)
    # os.makedirs(args.output_dir, exist_ok=True)
    import pickle
    pickle.dump(embeddings, open(f"{args.output_dir}/protein_embeddings.npy", "wb"))