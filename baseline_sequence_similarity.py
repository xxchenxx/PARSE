from Bio import pairwise2
import pickle
from glob import glob
import os
import pandas as pd
import numpy as np
from Bio import SeqIO
from io import StringIO
import requests as r

score = pairwise2
reference = pickle.load(open("msa_mcsa_rcsb_pos_weak_neg_aug_sampled100K_0402_version2_16000/csa_site_db_nn.pkl", "rb"))
mapping = pd.read_csv("pdb_chain_uniprot.csv", header=0)
pdbs = list(set(reference['pdbs']))
sequence_dict_reference = {}
sequence_dict_query = {}
baseUrl = "http://www.uniprot.org/uniprot/"
from tqdm import tqdm
query_proteins = pd.read_csv("test_pdb_hard/ids.csv")

for row in tqdm(query_proteins.iterrows()):
    pdb = row[1]['label']
    uniprot = row[1]['id']
    
    if not os.path.exists(f"fasta_cache/{uniprot}.fasta"):
        currentUrl=baseUrl + uniprot + ".fasta"
        # currentUrl=baseUrl + pdb_id.split("_")[1] + ".fasta"
        print(uniprot)
        response = r.post(currentUrl)
        cData=''.join(response.text)
        Seq=StringIO(cData)
        pSeq=SeqIO.parse(Seq, 'fasta')
        pSeq = list(pSeq)
        if len(pSeq) == 0:
            continue
        pSeq = str(pSeq[0].seq)
        with open(f"fasta_cache/{uniprot}.fasta", "w") as f:
            f.write(pSeq)
    else:
        with open(f"fasta_cache/{uniprot}.fasta", "r") as f:
            pSeq = f.read()
    sequence_dict_query[uniprot] = pSeq
    
print(len(sequence_dict_query))
for pdb in tqdm(pdbs):
    # get the uniprot id
    pdb = pdb[:4]
    print(pdb)
    try:
        uniprot = mapping[mapping['PDB'] == pdb]['SP_PRIMARY'].values[0]
        # get the sequence from the reference
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
    except:
        # get the structure sequence
        continue
    sequence_dict_reference[pdb] = pSeq

similarity_matrix = np.zeros((len(sequence_dict_reference), len(sequence_dict_query)))
for i, pdb1 in enumerate(tqdm(sequence_dict_reference.keys())):
    for j, pdb2 in enumerate(tqdm(sequence_dict_query.keys())):
        # similarity_matrix[i, j] = score.align.globalxx(sequence_dict_query[pdb], sequence_dict_reference[pdb2])
        similarity_matrix[i, j] = similarity_matrix[i, j] = score.align.globalxx(sequence_dict_reference[pdb1], sequence_dict_query[pdb2])[0].score

np.save("similarity_matrix.npy", similarity_matrix)
pdbs = [x[:4] for x in pdbs]
df = pd.DataFrame(similarity_matrix, columns=list(sequence_dict_query.keys()), index=list(sequence_dict_reference.keys()))
df.to_csv("similarity_matrix.csv")