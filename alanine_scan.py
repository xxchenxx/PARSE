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

mapping = pd.read_csv("pdb_chain_uniprot.csv", header=0)
esm_model, alphabet = pretrained.load_model_and_alphabet('esm2_t12_35M_UR50D')
esm_model = esm_model.to('cuda')
batch_converter = alphabet.get_batch_converter()

from tqdm import tqdm

def embed_protein_custom_with_target_resid(atom_df, model_fn, pdb_id, change_resid, change_aa, model, device='cpu', include_hets=True, env_radius=10.0, uniprot=None):
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
            filtered = mapping[(mapping['PDB'] == pdb_id) & (mapping['CHAIN'] == chain)]
            # print(filtered)
            for i in range(filtered.shape[0]):
                row = filtered.iloc[i]
                
                if resid >= row['RES_BEG'] and resid <= row['RES_END']:
                    index = resid - row['RES_BEG'] + row['SP_BEG'] - 1
                    # print(f"Current chain: {current_chain}, chain: {chain}")
                    # TODO: understand why this is slow. 
                    if current_chain == chain:
                        embeddings = sequence_embedding[index]
                    else:
                        currentUrl=baseUrl + row['SP_PRIMARY'] + ".fasta"
                        response = r.post(currentUrl)
                        cData=''.join(response.text)

                        Seq=StringIO(cData)
                        pSeq=SeqIO.parse(Seq, 'fasta')
                        pSeq = list(pSeq)
                        pSeq = str(pSeq[0].seq)
                        
                        if change_aa is None:
                            pass
                        else:
                            assert pSeq[change_resid - row['RES_BEG'] + row['SP_BEG'] - 1] == change_aa
                            pSeq = pSeq[:change_resid - row['RES_BEG'] + row['SP_BEG'] - 1] + 'A' + pSeq[change_resid - row['RES_BEG'] + row['SP_BEG']:]
                        d = [
                            ("protein1", pSeq),
                        ]
                        batch_labels, batch_strs, batch_tokens = batch_converter(d)
                        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)    
                        with torch.no_grad():
                            batch_tokens = batch_tokens.cuda()
                            results = esm_model(batch_tokens, repr_layers=[12], return_contacts=False)
                        token_representations = results["representations"][12][:,1:-1]
                        if 'MoCo' in args.model:
                            sequence_embedding = model.encoder_q(token_representations)[0]
                        elif args.model == 'SimSiam':
                            # print(token_representations.shape)
                            sequence_embedding = model.projector(token_representations.squeeze(0))
                        elif args.model == 'Triplet':
                            sequence_embedding = model(token_representations)[0]
                        embeddings = sequence_embedding[index]
                        current_chain = chain
                    emb_data['chains'].append(chain)
                    emb_data['resids'].append(atom_info.aa_to_letter(resname) + str(resid))
                    emb_data['confidence'].append(1)
                    emb_data['embeddings'].append(embeddings.detach().cpu().numpy())
                        # print(len(emb_data['embeddings']))
                        # print(resid)
                    current_resid = resid
                    current_chain = chain
    return emb_data

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
    parser.add_argument('--model', type=str, default='MoCo', choices=['MoCo', 'SimSiam', "MoCo_positive_only"])
    parser.add_argument('--queue_size', type=int, default=1024)
    args = parser.parse_args()
    
    start = time.time()
    device = 'cuda' if args.use_gpu else 'cpu'
    
    db = utils.deserialize(args.db)
    function_sets = utils.deserialize(args.function_sets)
    background_dists = utils.deserialize(args.background)
    
    # we don't need pdb input here
    
    from CLEAN.model import MoCo, MoCo_positive_only
    from CLEAN.simsiam import SimSiam

    if args.model == 'MoCo':
        model = MoCo(512, 128, torch.device('cuda'), torch.float, esm_model_dim=480, queue_size=args.queue_size).cuda()
    elif args.model == 'SimSiam':
        model = SimSiam(512, 128, torch.device('cuda'), torch.float, esm_model_dim=480).cuda()
    elif args.model == 'MoCo_positive_only':
        model = MoCo_positive_only(512, 128, torch.device('cuda'), torch.float, esm_model_dim=480, queue_size=args.queue_size).cuda()
    model.load_state_dict(torch.load(args.checkpoint)['model_state_dict'])
    model.eval()


    original_results = {}
    changed_results = {}
    for pdb, resid in zip(db['pdbs'], db['resids']):
        pdb, chain = pdb[:4], pdb[4]
        aa, pos = resid[0], int(resid[1:])
        pdb_df = process_pdb(os.path.join(args.pdb_dir, pdb + ".pdb"), chain=args.chain, include_hets=False)

        if pdb not in original_results:
            embed_data = embed_protein_custom_with_target_resid(pdb_df, None, pdb, None, None, model, device, include_hets=False)
            embed_data['id'] = pdb
            print(f'Time to embed PDB: {time.time() - start:.2f} seconds')
            rnk = parse.compute_rank_df(embed_data, db)
            original_results[pdb] = rnk
        
        embed_data = embed_protein_custom_with_target_resid(pdb_df, None, pdb, pos, aa, model, device, include_hets=False)
        embed_data['id'] = pdb
        print(f'Time to embed PDB: {time.time() - start:.2f} seconds')
        rnk = parse.compute_rank_df(embed_data, db)
        changed_results[f"{pdb}_{pos}"] = rnk
        break
        # print(rnk)
        # full_result = parse.parse(rnk, function_sets, background_dists, cutoff=1)
        # pdb_id = args.pdb.split("/")[-1].split(".")[0]
        # rnk.to_csv(f"rnk_1.csv")
        # full_result.to_csv(f'full_result_1.csv')
        # results = parse.parse(rnk, function_sets, background_dists, cutoff=args.cutoff)
        # results.to_csv(f'result_1.csv')

        # print(results)
        # print(f'Finished in {time.time() - start:.2f} seconds')
    print(original_results)
    print(changed_results)