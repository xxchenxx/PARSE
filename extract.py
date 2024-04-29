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

def extract_protein_embedding(model, atom_df, model_fn, pdb_id, device='cpu', include_hets=True, env_radius=10.0, uniprot=None):
    emb_data = col.defaultdict(list)
    graphs = []
    if not include_hets:
        atom_df = atom_df[atom_df.resname.isin(atom_info.aa)].reset_index(drop=True)
    # print(atom_df)
    current_chain = None
    current_resid = None
    baseUrl = "http://www.uniprot.org/uniprot/"
    for j in tqdm(range(atom_df.shape[0])):
        # g = g.to(device)
        chain = atom_df.iloc[j]['chain']
        # print(pdb_id, chain)
        # print(atom_df.iloc[j]) 
        # assert False
        resname = atom_df.iloc[j]['resname']
        resid = int(atom_df.iloc[j]['residue'])
        if current_resid == resid and current_chain == chain:
            continue
        else:
            # print(pdb_id, chain, resid)
            if not 'AF' in pdb_id:
                if current_chain == chain:
                # print(resid)
                # print(resid_to_index)
                # print(reverse_structure_seq_map)
                    resid_index = resid_to_index[resid]
                    if resid_index not in reverse_structure_seq_map:
                        print("A mismatch is detected at PDB {} and chain {} and resid {}. Skip this residue.".format(pdb_id, chain, resid))
                        current_resid = resid
                        current_chain = chain
                        continue
                    else:
                        index = reverse_structure_seq_map[resid_index] - 1
                    # print(pdb_sequence)
                    # print(pSeq)
                    # print(pdb_sequence[47 - 1], pSeq[50 - 1])
                    # print(pdb_sequence[48 - 1], pSeq[51 - 1])
                    # print(pdb_sequence[49 - 1], pSeq[52 - 1])
                    # if index >= len(sequence_embedding): 
                    #     continue
                    embeddings = sequence_embedding[index]
                    emb_data['chains'].append(chain)
                    emb_data['resids'].append(atom_info.aa_to_letter(resname) + str(resid))
                    emb_data['confidence'].append(1)
                    emb_data['embeddings'].append(embeddings)

                    current_resid = resid
                    current_chain = chain
                else:
                    filtered = mapping[(mapping['PDB'] == pdb_id) & (mapping['CHAIN'] == chain)]
                    pdb_sequence = []
                    resid_to_index = {}
                    current_resid = -1
                    start_resid = int(atom_df.iloc[0]['residue'])
                    count = 0
                    for i in range(len(atom_df)):
                        if atom_df.iloc[i]['residue'] != current_resid and atom_df.iloc[i]['chain'] == chain:
                            pdb_sequence.append(atom_info.aa_to_letter(atom_df.iloc[i]['resname']))
                            # print(pdb_sequence)
                            current_resid = atom_df.iloc[i]['residue']
                            resid_to_index[current_resid] = count + 1 # 1-indexed
                            count += 1
                    pdb_sequence = ''.join(pdb_sequence)
                    for i in range(min(filtered.shape[0], 1)):
                        row = filtered.iloc[i]

                        if not os.path.exists(f"fasta_cache/{uniprot}.fasta"):
                            currentUrl=baseUrl + row['SP_PRIMARY'] + ".fasta"
                            # currentUrl=baseUrl + pdb_id.split("_")[1] + ".fasta"
                            response = r.post(currentUrl)
                            cData=''.join(response.text)

                            Seq=StringIO(cData)
                            pSeq=SeqIO.parse(Seq, 'fasta')
                            pSeq = list(pSeq)
                            pSeq = str(pSeq[0].seq)
                            with open(f"fasta_cache/{uniprot}.fasta", "w") as f:
                                f.write(pSeq)
                        else:
                            with open(f"fasta_cache/{uniprot}.fasta", "r") as f:
                                pSeq = f.read()
                        # print()
                        # print(pSeq)
                        # print(pdb_sequence)
                        structure_seq_map = generate_struct_to_seq_map(pdb_sequence, pSeq, range(1, len(pSeq) + 1), range(1, len(pSeq) + 1))
                        reverse_structure_seq_map = {v: k for k, v in structure_seq_map.items()}
                        pSeq_new = pSeq
                        if args.prefix is not None:
                            pSeq_new = args.prefix + pSeq_new
                        d = [
                            ("protein1", pSeq_new),
                        ]
                        if not is_lora_esm:
                            batch_labels, batch_strs, batch_tokens = batch_converter(d)
                            batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)    
                            with torch.no_grad():
                                batch_tokens = batch_tokens.cuda()
                                results = esm_model(batch_tokens, repr_layers=[12], return_contacts=False)
                            if args.prefix is None:
                                token_representations = results["representations"][12][:,1:-1]
                            else:
                                token_representations = results["representations"][12][:,1 + len(args.prefix):-1]
                            print(token_representations)
                        else:
                            inputs = tokenizer(pSeq, return_tensors="pt", padding="longest", truncation=True, max_length=2048)
                            inputs = {k: v.to('cuda') for k, v in inputs.items()}
                            with torch.no_grad():
                                outputs = esm_model(**inputs)
                            token_representations = outputs
                            print(token_representations)
                        
                        if 'MoCo' in args.model:
                            sequence_embedding = model.encoder_q(token_representations)[0].detach().cpu().numpy()
                        elif args.model == 'SimSiam':
                            # print(token_representations.shape)
                            sequence_embedding = model.projector(token_representations.squeeze(0)).detach().cpu().numpy()
                        elif args.model == 'Triplet':
                            sequence_embedding = model(token_representations)[0].detach().cpu().numpy()
                        
                        
                        resid_index = resid_to_index[resid]
                        if resid_index not in reverse_structure_seq_map:
                            print("A mismatch is detected at PDB {} and chain {} and resid {}. Skip this residue.".format(pdb_id, chain, resid))
                            current_resid = resid
                            current_chain = chain
                            continue
                        else:
                            index = reverse_structure_seq_map[resid_index] - 1
                        embeddings = sequence_embedding[index]
                        emb_data['chains'].append(chain)
                        emb_data['resids'].append(atom_info.aa_to_letter(resname) + str(resid))
                        emb_data['confidence'].append(1)
                        emb_data['embeddings'].append(embeddings)

                        current_resid = resid
                        current_chain = chain
                        break
            else:
                index = resid - 1
                if current_chain == chain:
                    embeddings = sequence_embedding[index]
                else:
                    if not os.path.exists(f"fasta_cache/{uniprot}.fasta"):
                        currentUrl=baseUrl + uniprot + ".fasta"
                        # currentUrl=baseUrl + pdb_id.split("_")[1] + ".fasta"
                        response = r.post(currentUrl)
                        cData=''.join(response.text)

                        Seq=StringIO(cData)
                        pSeq=SeqIO.parse(Seq, 'fasta')
                        pSeq = list(pSeq)
                        pSeq = str(pSeq[0].seq)
                        with open(f"fasta_cache/{uniprot}.fasta", "w") as f:
                            f.write(pSeq)
                    else:
                        with open(f"fasta_cache/{uniprot}.fasta", "r") as f:
                            pSeq = f.read()

                    if args.prefix is not None:
                        pSeq = args.prefix + pSeq
                    d = [
                        ("protein1", pSeq),
                    ]

                    if not is_lora_esm:
                        batch_labels, batch_strs, batch_tokens = batch_converter(d)
                        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)    
                        with torch.no_grad():
                            batch_tokens = batch_tokens.cuda()
                            results = esm_model(batch_tokens, repr_layers=[12], return_contacts=False)
                        if args.prefix is None:
                            token_representations = results["representations"][12][:,1:-1]
                        else:
                            token_representations = results["representations"][12][:,1 + len(args.prefix):-1]
                        print(token_representations)
                        
                    else:
                        inputs = tokenizer(pSeq, return_tensors="pt", padding="longest", truncation=True, max_length=2048)
                        inputs = {k: v.to('cuda') for k, v in inputs.items()}
                        with torch.no_grad():
                            outputs = esm_model(**inputs)
                        token_representations = outputs
                        print(token_representations)
                    if 'MoCo' in args.model:
                        sequence_embedding = model.encoder_q(token_representations)[0].detach().cpu().numpy()
                    elif args.model == 'SimSiam':
                        # print(token_representations.shape)
                        sequence_embedding = model.projector(token_representations.squeeze(0)).detach().cpu().numpy()
                    elif args.model == 'Triplet':
                        sequence_embedding = model(token_representations)[0].detach().cpu().numpy()
                    embeddings = sequence_embedding[index]
                    
                
                emb_data['chains'].append(chain)
                emb_data['resids'].append(atom_info.aa_to_letter(resname) + str(resid))
                emb_data['confidence'].append(1)
                emb_data['embeddings'].append(embeddings)
                current_resid = resid
                current_chain = chain
        # emb_data['embeddings'] = np.stack(embs.cpu().numpy(), 0)
    return emb_data

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
    residue_embeddings = []
    
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
            try:
                assert sequence[index] == three_to_one_mappings[protein_auth_res_name]
                residue_embedding = sequence_embedding[index]
                residue_embeddings.append(residue_embedding)
            except AssertionError:
                print("Mismatch detected")
                residue_embeddings.append(np.ones_like(residue_embeddings[0]) * -10000)
        else:
            # print(row)
            # print(positions)
            index = positions.index(protein_auth_seq_num)
            # print(sequence[index])
            # print(protein_auth_res_name)
            try:
                assert sequence[index] == three_to_one_mappings[protein_auth_res_name]
                residue_embedding = sequence_embedding[index]
                residue_embeddings.append(residue_embedding)
            except AssertionError:
                print("Mismatch detected")
                residue_embeddings.append(np.ones_like(residue_embeddings[0]) * -10000)
    
    residue_embeddings = np.stack(residue_embeddings, 0)
    os.makedirs(args.output_dir, exist_ok=True)
    np.save(f"{args.output_dir}/residue_embeddings.npy", residue_embeddings)