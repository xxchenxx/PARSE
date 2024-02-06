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

mapping = pd.read_csv("pdb_chain_uniprot.csv", header=0)
esm_model, alphabet = pretrained.load_model_and_alphabet('esm2_t12_35M_UR50D')
esm_model = esm_model.to('cuda')
batch_converter = alphabet.get_batch_converter()
esm_model.eval()
from CLEAN.model import MoCo
model = MoCo(512, 128, torch.device('cuda'), torch.float, esm_model_dim=480, queue_size=1000).cuda()
# model.load_state_dict(torch.load("best_demo_1702870147.059707_checkpoints.pth.tar")['model_state_dict'])
model.load_state_dict(torch.load("/home/xuxi/PARSE/best_large_lr5e-4_wd1e-3_ep1000_1705825663.9917889_checkpoints.pth.tar")['model_state_dict'])

from tqdm import tqdm
model.eval()
def embed_protein_custom(atom_df, model_fn, pdb_id, device='cpu', include_hets=True, env_radius=10.0, uniprot=None):
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
                filtered = mapping[(mapping['PDB'] == pdb_id) & (mapping['CHAIN'] == chain)]
                # print(filtered)
                for i in range(filtered.shape[0]):
                    row = filtered.iloc[i]
                    
                    # print(g.resid[0], row['RES_BEG'], row['RES_END'])
                    # print(resid, row['RES_BEG'], row['RES_END'])
                
                    if resid >= row['RES_BEG'] and resid <= row['RES_END']:
                        index = resid - row['RES_BEG'] + row['SP_BEG'] - 1
                        # print(index) - 1
                        if current_chain == chain:
                            embeddings = sequence_embedding[index]
                        else:
                            pSeq = "MLKYKPLLKISKNCEAAILRASKTRLNTIRAYGSTVPKSKSFEQDSRKRTQSWTALRVGAILAATSSVAYLNWHNGQIDNEPKLDMNKQKISPAEVAKHNKPDDCWVVINGYVYDLTRFLPNHPGGQDVIKFNAGKDVTAIFEPLHAPNVIDKYIAPEKKLGPLQGSMPPELVCPPYAPGETKEDIARKEQLKSLLPPLDNIINLYDFEYLASQTLTKQAWAYYSSGANDEVTHRENHNAYHRIFFKPKILVDVRKVDISTDMLGSHVDVPFYVSATALCKLGNPLEGEKDVARGCGQGVTKVPQMISTLASCSPEEIIEAAPSDKQIQWYQLYVNSDRKITDDLVKNVEKLGVKALFVTVDAPSLGQREKDMKLKFSNTKAGPKAMKKTNVEESQGASRALSKFIDPSLTWKDIEELKKKTKLPIVIKGVQRTEDVIKAAEIGVSGVVLSNHGGRQLDFSRAPIEVLAETMPILEQRNLKDKLEVFVDGGVRRGTDVLKALCLGAKGVGLGRPFLYANSCYGRNGVEKAIEILRDEIEMSMRLLGVTSIAELKPDLLDLSTLKARTVGVPNDVLYNEVYEGPTLTEFEDA"
                            print(pSeq[282 - row['RES_BEG'] + row['SP_BEG'] - 1])
                            pSeq = pSeq[:282 - row['RES_BEG'] + row['SP_BEG'] - 1] + 'A' + pSeq[282 - row['RES_BEG'] + row['SP_BEG']:]
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
                            current_chain = chain
                        emb_data['chains'].append(chain)
                        
                        emb_data['resids'].append(atom_info.aa_to_letter(resname) + str(resid))
                        emb_data['confidence'].append(1)
                        emb_data['embeddings'].append(embeddings.detach().cpu().numpy())
                        # print(len(emb_data['embeddings']))
                        # print(resid)
                        current_resid = resid
                        current_chain = chain
            else:
                index = resid - 1
                if current_chain == chain:
                    embeddings = sequence_embedding[index]
                else:
                    currentUrl=baseUrl + uniprot + ".fasta"
                    # currentUrl=baseUrl + pdb_id.split("_")[1] + ".fasta"
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
                    
                
                emb_data['chains'].append(chain)
                emb_data['resids'].append(atom_info.aa_to_letter(resname) + str(resid))
                emb_data['confidence'].append(1)
                emb_data['embeddings'].append(embeddings.detach().cpu().numpy())
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
    args = parser.parse_args()
    
    start = time.time()
    device = 'cuda' if args.use_gpu else 'cpu'
    
    db = utils.deserialize(args.db)
    function_sets = utils.deserialize(args.function_sets)
    background_dists = utils.deserialize(args.background)
    
    if args.pdb:
        # model = initialize_model(device=device)
        pdb_df = process_pdb(args.pdb, chain=args.chain, include_hets=False)
        if not 'AF' in args.pdb:
            embed_data = embed_protein_custom(pdb_df, None, args.pdb.split("/")[-1].split(".")[0], device, include_hets=False)
        else:
            embed_data = embed_protein_custom(pdb_df, None, args.pdb.split("/")[-1].split(".")[0], device, include_hets=False, uniprot=args.pdb.split("-")[1])
        embed_data['id'] = args.pdb
        print(f'Time to embed PDB: {time.time() - start:.2f} seconds')
    else:
        assert False

    rnk = parse.compute_rank_df(embed_data, db)
    print(rnk)
    full_result = parse.parse(rnk, function_sets, background_dists, cutoff=1)
    pdb_id = args.pdb.split("/")[-1].split(".")[0]
    rnk.to_csv(f"rnk_2.csv")
    full_result.to_csv(f'full_result_2.csv')
    results = parse.parse(rnk, function_sets, background_dists, cutoff=args.cutoff)
    results.to_csv(f'result_2.csv')

    print(results)
    print(f'Finished in {time.time() - start:.2f} seconds')
    