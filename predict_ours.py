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
from utils import generate_struct_to_seq_map
import os

torch.backends.cudnn.benchmark = False
# deteministic
torch.backends.cudnn.deterministic = True

mapping = pd.read_csv("pdb_chain_uniprot.csv", header=0)
esm_model, alphabet = pretrained.load_model_and_alphabet('esm2_t12_35M_UR50D')
esm_model = esm_model.to('cuda')
batch_converter = alphabet.get_batch_converter()
esm_model.eval()

from tqdm import tqdm


def embed_protein_custom(model, atom_df, model_fn, pdb_id, device='cpu', include_hets=True, env_radius=10.0, uniprot=None):
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
                        batch_labels, batch_strs, batch_tokens = batch_converter(d)
                        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)    
                        with torch.no_grad():
                            batch_tokens = batch_tokens.cuda()
                            results = esm_model(batch_tokens, repr_layers=[12], return_contacts=False)
                        if args.prefix is not None:
                            token_representations = results["representations"][12][:,1:-1]
                        else:
                            token_representations = results["representations"][12][:,1 + len(args.prefix):-1]
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

                    batch_labels, batch_strs, batch_tokens = batch_converter(d)
                    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)    
                    with torch.no_grad():
                        batch_tokens = batch_tokens.cuda()
                        results = esm_model(batch_tokens, repr_layers=[12], return_contacts=False)
                    if args.prefix is not None:
                        token_representations = results["representations"][12][:,1:-1]
                    else:
                        token_representations = results["representations"][12][:,1 + len(args.prefix):-1]
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
    args = parser.parse_args()
    
    start = time.time()
    device = 'cuda' if args.use_gpu else 'cpu'
    
    db = utils.deserialize(args.db)
    function_sets = utils.deserialize(args.function_sets)
    background_dists = utils.deserialize(args.background)
    
    from CLEAN.model import MoCo, MoCo_positive_only, LayerNormNet
    from CLEAN.simsiam import SimSiam

    if args.model == 'MoCo':
        model = MoCo(512, 128, torch.device('cuda'), torch.float, esm_model_dim=480, queue_size=args.queue_size).cuda()
    elif args.model == 'SimSiam':
        model = SimSiam(512, 128, torch.device('cuda'), torch.float, esm_model_dim=480).cuda()
    elif args.model == 'MoCo_positive_only':
        model = MoCo_positive_only(512, 128, torch.device('cuda'), torch.float, esm_model_dim=480, queue_size=args.queue_size).cuda()
    elif args.model == 'Triplet':
        model = LayerNormNet(512, 128, torch.device('cuda'), torch.float, esm_model_dim=480).cuda()
    try:
        model.load_state_dict(torch.load(args.checkpoint)['model_state_dict'])
    except:
        model.load_state_dict(torch.load(args.checkpoint))
        
    model.eval()
    if args.pdb:
        # model = initialize_model(device=device)
        pdb_df = process_pdb(args.pdb, chain=args.chain, include_hets=False)
        if not 'AF' in args.pdb:
            embed_data = embed_protein_custom(model, pdb_df, None, args.pdb.split("/")[-1].split(".")[0], device, include_hets=False)
        else:
            embed_data = embed_protein_custom(model, pdb_df, None, args.pdb.split("/")[-1].split(".")[0], device, include_hets=False, uniprot=args.pdb.split("-")[1])
        embed_data['id'] = args.pdb
        print(f'Time to embed PDB: {time.time() - start:.2f} seconds')
    else:
        assert False

    rnk = parse.compute_rank_df(embed_data, db)
    full_result = parse.parse(rnk, function_sets, background_dists, cutoff=1)
    pdb_id = args.pdb.split("/")[-1].split(".")[0]
    rnk.to_csv(f"ours_rnk_{pdb_id}.csv")
    full_result.to_csv(f'ours_{pdb_id}_full_result.csv')
    results = parse.parse(rnk, function_sets, background_dists, cutoff=args.cutoff)
    results.to_csv(f'ours_{pdb_id}_cutoff_{args.cutoff}.csv')

    print(results)
    print(f'Finished in {time.time() - start:.2f} seconds')
    