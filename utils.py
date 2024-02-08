import pickle
import numpy as np

def serialize(obj, path):
    with open(path, 'wb') as fh:
        pickle.dump(obj, fh)

def deserialize(path):
    with open(path, 'rb') as fh:
        return pickle.load(fh)

def pdb_from_fname(fname):
    af_flag = False
    if fname.endswith('.ent.gz'):
        pdb = fname[3:7]
    elif fname.endswith('.pdb'):
        pdb = fname[:-4]
    elif 'AF' in fname:
        af_flag = True
        pdb = fname.split('-')[1]
    else:
        pdb = fname
    return pdb, af_flag

def get_db_site_map(rnk):
    res_match = np.array([x.split('_')[1][0] for x in rnk['site']]) == rnk['location'].str[0]
    return dict(zip(rnk['site'], rnk['location'])), dict(zip(rnk['site'], res_match))

def calc_rmsd(A, B):
    D = len(A[0])
    N = len(A)
    rmsd = 0.0
    for v, w in zip(A, B):
        rmsd += sum([(v[i] - w[i])**2.0 for i in range(D)])
    return np.sqrt(rmsd/N)

def align(A, B):
    A -= A.mean(axis=0)
    B -= B.mean(axis=0)
    
    C = np.dot(np.transpose(A), B)
    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0
    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]
    U = np.dot(V, W)
    A_rot = np.dot(A, U)
    rmsd = calc_rmsd(A_rot, B)
    return U, A.mean(axis=0), B.mean(axis=0), rmsd

from typing import List, Literal, Dict, Iterable
def isalpha(char: str) -> bool:
    return char.isalpha()
from numpy import isnan
import gemmi

def alignment_score(alignment: gemmi.AlignmentResult, ref: Literal["first", "second", "shortest"]="shortest") -> float:
    assert isinstance(alignment, gemmi.AlignmentResult)
    assert ref in ["first", "second", "shortest"]
    if ref == "first":
        return round(alignment.calculate_identity(1), 3)
    elif ref == "second":
        return round(alignment.calculate_identity(2), 3)
    else:
        return round(alignment.calculate_identity(), 3)
def check_aligment_score(sequence: str,  struct_sequence: str) -> float:
    alignment = gemmi.align_string_sequences(list(sequence), list(struct_sequence), [])
    return alignment_score(alignment)
def align_sequences(seq1: str, seq2: str) -> str:
    alignment = gemmi.align_string_sequences(list(seq1), list(seq2), [])
    return alignment.formatted(seq1, seq2)
def generate_seq_to_struct_map(
    sequence: str,  struct_sequence: str, struct_pos: List[int], seq_pos: List[int]=None
) -> Dict[int, int]:
    """
    Returns a dict where:
        the key is the residue position from the sequence
        the value is the residue position from structure sequence
    """
    alignment = gemmi.align_string_sequences(list(sequence), list(struct_sequence), [])
    seq_gaps = alignment.add_gaps(sequence, 1)
    struc_gaps = alignment.add_gaps(struct_sequence, 2)
    # print(alignment.calculate_identity(), alignment.score)
    # print(seq_gaps)
    # print(alignment.match_string.replace(" ", "-"))
    # print(struc_gaps)
    if isinstance(seq_pos, Iterable) and len(seq_pos) > 0:
        seq_pos_gen = (num for num in seq_pos)
    elif seq_pos is None:
        seq_pos_gen = (num for num in range(1, len(sequence) + 1))
    else:
        raise TypeError("Invalid type passed for seq_pos parameter")
    struct_pos_gen = (num for num in struct_pos)
    seq_struct_map = dict()
    for idx in range(len(seq_gaps)):
        seq_char = seq_gaps[idx]
        struct_char = struc_gaps[idx]
        struct_seq_id = None
        if isalpha(struct_char):
            struct_seq_id = next(struct_pos_gen)
        if isalpha(seq_char):
            seq_id = next(seq_pos_gen)
        if (
            isalpha(struct_char)
            and isalpha(seq_char)
            and struct_char.upper() == seq_char.upper()
            and not isnan(struct_seq_id)
        ):
            seq_struct_map[seq_id] = int(struct_seq_id)
    return seq_struct_map

def generate_struct_to_seq_map(
    sequence: str,  struct_sequence: str, struct_pos: List[int], seq_pos: List[int]=None
) -> Dict[int, int]:
    """
    Returns a dict where:
        the key is the residue position from the structure sequence
        the value is the residue position from sequence
    """
    alignment = gemmi.align_string_sequences(list(sequence), list(struct_sequence), [])
    seq_gaps = alignment.add_gaps(sequence, 1)
    struc_gaps = alignment.add_gaps(struct_sequence, 2)
    if isinstance(seq_pos, Iterable) and len(seq_pos) > 0:
        seq_pos_gen = (num for num in seq_pos)
    elif seq_pos is None:
        seq_pos_gen = (num for num in range(1, len(sequence) + 1))
    else:
        raise TypeError("Invalid type passed for seq_pos parameter")
    struct_pos_gen = (num for num in struct_pos)
    struc_seq_map = dict()
    for idx in range(len(seq_gaps)):
        seq_char = seq_gaps[idx]
        struct_char = struc_gaps[idx]
        struct_seq_id = None
        if isalpha(struct_char):
            struct_seq_id = next(struct_pos_gen)
        if isalpha(seq_char):
            seq_id = next(seq_pos_gen)
        if (
            isalpha(struct_char)
            and isalpha(seq_char)
            and struct_char.upper() == seq_char.upper()
            and not isnan(struct_seq_id)
        ):
            struc_seq_map[struct_seq_id] = int(seq_id)
    return struc_seq_map