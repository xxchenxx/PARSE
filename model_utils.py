import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from txplm.data.data_utils import DATA_DIR
from txplm.data.dataset import ProteinEvalDataset


def create_mlp(n_layers, in_features, out_features, hidden_features = 256, dropout_rate = 0.25):
    """
    From ChatGPT:
    Create a PyTorch sequential module with 'n' linear layers and ReLU activations.

    Parameters:
        n_layers (int): Number of linear layers in the sequential module.

    Returns:
        nn.Sequential: PyTorch sequential module.
    """
    layers = []

    for i in range(n_layers):
        # Add linear layer with ReLU activation except for the last layer
        in_size = hidden_features if i > 0 else in_features 
        if i < n_layers - 1:
            layers.append(nn.Linear(in_size, hidden_features))
            if dropout_rate is not None:
                layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.GELU())
        else:
            # For the last layer, don't apply ReLU activation
            layers.append(nn.Linear(in_size, out_features))

    return nn.Sequential(*layers)

ALL_PROTEINS_FILE = os.path.join(DATA_DIR, "integrated_data/v1/protein/protein_info_filtered.pkl")
def get_all_protein_embeddings(
        model,
        batch_size = 16,
        all_proteins_file = ALL_PROTEINS_FILE
    ):
    '''
    Gets embeddings for every protein in our stored protein file
    '''
    df_all_prots = pd.read_pickle(ALL_PROTEINS_FILE)
    protein_dataset = ProteinEvalDataset(df_all_prots['index'])
    protein_dataloader = DataLoader(
        protein_dataset,
        batch_size = batch_size,
        num_workers = 2,
        drop_last = False,
        shuffle = False,
        pin_memory = True,
    )

    # Run inference loop:
    # Passing proteins via dataloader:
    model_device = model.device
    extra_protein_embeddings = []
    all_prot_inds = []
    for i, model_inputs in enumerate(protein_dataloader):
        if isinstance(model_inputs, dict):
            model_inputs["data"] = model_inputs["data"].to(model_device)
            protein_inputs = model_inputs
        else:
            all_prot_inds.append(model_inputs)
            protein_inputs = model_inputs.to(model_device)

        out = model.forward_sequences(protein_inputs)
        extra_protein_embeddings.append(out["shared"].detach().clone().cpu())

    extra_prot_embeds = torch.cat(extra_protein_embeddings, dim = 0)
    all_prot_inds = torch.cat(all_prot_inds).flatten()

    return extra_prot_embeds, all_prot_inds