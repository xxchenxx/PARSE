import pandas as pd
import requests
from Bio import SeqIO, PDB
from io import StringIO
import os

pdbl = PDB.PDBList()
pdb_parser = PDB.PDBParser()

pdb_dir = './pdb_files'
os.makedirs(pdb_dir, exist_ok=True)

# Define the paths to your files
csa_file_path = 'data/csa_functional_sites.csv'
mapping_file_path = 'pdb_chain_uniprot.csv'
output_fasta_path = 'protein_sequences.fasta'

csa_df = pd.read_csv(csa_file_path)
mapping_df = pd.read_csv(mapping_file_path)

three_to_one = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
    'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
    'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
    'THR': 'T', 'VAL': 'V', 'TRLIP': 'W', 'TYR': 'Y',
}

def fetch_uniprot_sequence(uniprot_id):
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    response = requests.get(url)
    if response.status_code == 200:
        fasta_io = StringIO(response.text)
        for record in SeqIO.parse(fasta_io, "fasta"):
            return str(record.seq)
    else:
        return None

def extract_sequence_from_pdb(pdb_id, chain_id):
    try:
        pdb_file_path = pdbl.retrieve_pdb_file(pdb_id, file_format='pdb', pdir=pdb_dir, overwrite=False)
        structure = pdb_parser.get_structure(pdb_id, pdb_file_path)
        for model in structure:
            for chain in model:
                if chain.id == chain_id:
                    sequence = ''.join(three_to_one.get(residue.resname, 'X') for residue in chain if residue.id[0] == ' ')
                    return sequence
    except Exception as e:
        print(f"An error occurred while processing {pdb_id}: {e}")
    return None

with open(output_fasta_path, 'w') as fasta_file:
    for index, row in csa_df.iterrows():
        pdb_id = row['pdb'][:4]
        chain_id = row['pdb'][4:]
        sequence = extract_sequence_from_pdb(pdb_id, chain_id)
        
        if sequence:
            fasta_file.write(f">{pdb_id}_{chain_id}\n{sequence}\n")
        else:
            mapping_row = mapping_df[(mapping_df['PDB'] == pdb_id) & (mapping_df['CHAIN'] == chain_id)]
            if not mapping_row.empty:
                uniprot_id = mapping_row.iloc[0]['SP_PRIMARY']
                sequence = fetch_uniprot_sequence(uniprot_id)
                if sequence:
                    fasta_file.write(f">{pdb_id}_{chain_id}_uniprot\n{sequence}\n")
                    print(f"Wrote {sequence}")
                else:
                    print(f"Failed to fetch sequence for UniProt ID: {uniprot_id}")
            else:
                print(f"No mapping found for PDB ID: {pdb_id}, Chain: {chain_id}")

print(f"Sequences saved to {output_fasta_path}")