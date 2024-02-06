import os
import urllib
import urllib.request
import pandas as pd
data = pd.read_csv("data/csa_functional_sites.csv")

for i in range(data.shape[0]):
    pdb = data.iloc[i]['pdb']
    pdb_id = pdb[:4]
    if not os.path.exists(f"pdb/{pdb_id}.pdb"):
        print(pdb_id)
        try:
            urllib.request.urlretrieve(f'https://files.rcsb.org/download/{pdb_id}.pdb', f'pdb/{pdb_id}.pdb')
        except Exception as e:
            print(e)
            pass