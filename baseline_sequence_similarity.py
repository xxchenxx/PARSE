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

query_proteins = pd.read_csv("test_pdb_hard/ids.csv")

for row in tqdm(query_proteins.iterrows()):
    pdb = row[1]['label']
    uniprot = row[1]['id']
    try:
        with open(f"fasta_cache/{uniprot}.fasta", "r") as f:
            pSeq = f.read()
        sequence_dict_query[pdb] = pSeq
    except:
        continue

similarity_matrix = np.zeros((len(pdbs), len(glob("test_pdb_hard/*.pdb")))).T
for i, row in query_proteins.iterrows():
    for j, pdb2 in enumerate(pdbs):
        pdb = row['label']
        pdb2 = pdb2[:4]
        # similarity_matrix[i, j] = score.align.globalxx(sequence_dict_query[pdb], sequence_dict_reference[pdb2])
        try:
            similarity_matrix[i, j] = score.align.globalxx(sequence_dict_query[pdb], sequence_dict_reference[pdb2])[0].score
        except:
            similarity_matrix[i, j] = -1

np.save("similarity_matrix.npy", similarity_matrix)
pdbs = [x[:4] for x in pdbs]
df = pd.DataFrame(similarity_matrix, columns=pdbs, index=query_proteins['label'])
df.to_csv("similarity_matrix.csv")