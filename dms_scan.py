from collapse import process_pdb, embed_protein, initialize_model
from atom3d.datasets import load_dataset
import argparse
import time
import parse
import utils
import collections as col
import pandas as pd
import torch
from esm import pretrained
from collapse.data import atom_info
import urllib
import requests as r
from Bio import SeqIO
from io import StringIO
import os
from utils import generate_struct_to_seq_map
import numpy as np
mapping = pd.read_csv("pdb_chain_uniprot.csv", header=0)
esm_model, alphabet = pretrained.load_model_and_alphabet('esm2_t12_35M_UR50D')
esm_model = esm_model.to('cuda')
batch_converter = alphabet.get_batch_converter()

from tqdm import tqdm



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb_dir', type=str, default=None, help='Input PDB file')
    parser.add_argument('--precomputed_id', type=str, default=None, help='ID for accessing precomputed embeddings')
    parser.add_argument('--precomputed_lmdb', type=str, default=None, help='Precomputed embeddings in LMDB format')
    parser.add_argument('--chain', type=str, default=None, help='Input PDB chain to annotate')
    parser.add_argument('--db', type=str, default='./data/csa_site_db_nn.pkl', help='Reference database embeddings')
    parser.add_argument('--function_sets', type=str, default='./data/csa_function_sets_nn.pkl', help='Reference function sets')
    parser.add_argument('--background', type=str, default='./data/function_score_dists.pkl', help='Function-specific background distributions')
    parser.add_argument('--cutoff', type=float, default=0.001, help='FDR cutoff for reporting results')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU to embed proteins')
    parser.add_argument('--checkpoint', type=str, default='best_demo_1702870147.059707_checkpoints.pth.tar')
    parser.add_argument('--model', type=str, default='MoCo', choices=['MoCo', 'SimSiam', "MoCo_positive_only", 'Triplet'])
    parser.add_argument('--output_name', type=str)
    parser.add_argument('--queue_size', type=int, default=1024)
    parser.add_argument('--dms_file', type=str, required=True)
    args = parser.parse_args()
    
    start = time.time()
    device = 'cuda' if args.use_gpu else 'cpu'
    
    db = utils.deserialize(args.db)
    function_sets = utils.deserialize(args.function_sets)
    background_dists = utils.deserialize(args.background)
    
    # we don't need pdb input here
    
    from CLEAN.model import MoCo, MoCo_positive_only, LayerNormNet
    # from CLEAN.simsiam import SimSiam

    if args.model == 'MoCo':
        model = MoCo(512, 128, torch.device('cuda'), torch.float, esm_model_dim=480, queue_size=args.queue_size).cuda()
    elif args.model == 'SimSiam':
        # model = SimSiam(512, 128, torch.device('cuda'), torch.float, esm_model_dim=480).cuda()
        pass
    elif args.model == 'Triplet':
        model = LayerNormNet(512, 128, torch.device('cuda'), torch.float, esm_model_dim=480).cuda()
    elif args.model == 'MoCo_positive_only':
        model = MoCo_positive_only(512, 128, torch.device('cuda'), torch.float, esm_model_dim=480, queue_size=args.queue_size).cuda()
    try:
        model.load_state_dict(torch.load(args.checkpoint)['model_state_dict'])
    except:
        model.load_state_dict(torch.load(args.checkpoint))
    model.eval()


    similarity = []
    dms_file = pd.read_csv(args.dms_file)
    from tqdm import tqdm
    args.output_name = args.dms_file.split('/')[-1].split('.')[0]
    if os.path.exists(args.output_name + "_predicted.csv"):
        exit(0)
    for idx, row in tqdm(dms_file.iterrows(), total=dms_file.shape[0]):
        mutated_sequence = row['mutated_sequence']
        mutant = row['mutant']
        wt, pos, mut = mutant[0], int(mutant[1:-1]), mutant[-1]
        original_sequence = mutated_sequence[:pos - 1] + wt + mutated_sequence[pos:]

        d = [
            ("protein1", original_sequence),
        ]
        batch_labels, batch_strs, batch_tokens = batch_converter(d)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)    
        with torch.no_grad():
            batch_tokens = batch_tokens.cuda()
            results = esm_model(batch_tokens, repr_layers=[12], return_contacts=False)
        token_representations = results["representations"][12][:,1:-1]
        if 'MoCo' in args.model:
            sequence_embedding = model.encoder_q(token_representations)[0].detach().cpu().numpy()
        elif args.model == 'SimSiam':
            # print(token_representations.shape)
            sequence_embedding = model.projector(token_representations.squeeze(0)).detach().cpu().numpy()
        elif args.model == 'Triplet':
            sequence_embedding = model(token_representations)[0].detach().cpu().numpy()

        original_residue_embedding = sequence_embedding[pos - 1]

        d = [
            ("protein1", mutated_sequence),
        ] 
        batch_labels, batch_strs, batch_tokens = batch_converter(d)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)    
        with torch.no_grad():
            batch_tokens = batch_tokens.cuda()
            results = esm_model(batch_tokens, repr_layers=[12], return_contacts=False)
        token_representations = results["representations"][12][:,1:-1]
        if 'MoCo' in args.model:
            sequence_embedding = model.encoder_q(token_representations)[0].detach().cpu().numpy()
        elif args.model == 'SimSiam':
            # print(token_representations.shape)
            sequence_embedding = model.projector(token_representations.squeeze(0)).detach().cpu().numpy()
        elif args.model == 'Triplet':
            sequence_embedding = model(token_representations)[0].detach().cpu().numpy()

        mutated_residue_embedding = sequence_embedding[pos - 1]

        similarity.append(np.dot(original_residue_embedding, mutated_residue_embedding) / (np.linalg.norm(original_residue_embedding) * np.linalg.norm(mutated_residue_embedding)))

    dms_file['Ours_score'] = pd.Series(similarity)
    

    dms_file.to_csv(args.output_name + "_predicted.csv", index=False)