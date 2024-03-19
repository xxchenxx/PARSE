import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# from txplm.data.data_utils import DATA_DIR
# from txplm.data.dataset import ProteinEvalDataset


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

# ALL_PROTEINS_FILE = os.path.join(DATA_DIR, "integrated_data/v1/protein/protein_info_filtered.pkl")
# def get_all_protein_embeddings(
#         model,
#         batch_size = 16,
#         all_proteins_file = ALL_PROTEINS_FILE
#     ):
#     '''
#     Gets embeddings for every protein in our stored protein file
#     '''
#     df_all_prots = pd.read_pickle(ALL_PROTEINS_FILE)
#     protein_dataset = ProteinEvalDataset(df_all_prots['index'])
#     protein_dataloader = DataLoader(
#         protein_dataset,
#         batch_size = batch_size,
#         num_workers = 2,
#         drop_last = False,
#         shuffle = False,
#         pin_memory = True,
#     )

#     # Run inference loop:
#     # Passing proteins via dataloader:
#     model_device = model.device
#     extra_protein_embeddings = []
#     all_prot_inds = []
#     for i, model_inputs in enumerate(protein_dataloader):
#         if isinstance(model_inputs, dict):
#             model_inputs["data"] = model_inputs["data"].to(model_device)
#             protein_inputs = model_inputs
#         else:
#             all_prot_inds.append(model_inputs)
#             protein_inputs = model_inputs.to(model_device)

#         out = model.forward_sequences(protein_inputs)
#         extra_protein_embeddings.append(out["shared"].detach().clone().cpu())

#     extra_prot_embeds = torch.cat(extra_protein_embeddings, dim = 0)
#     all_prot_inds = torch.cat(all_prot_inds).flatten()

#     return extra_prot_embeds, all_prot_inds

def batched_split_long_seq(toks: torch.Tensor, padding_idx: int, eos_idx:int, long_protein_strategy: str = 'split', max_protein_len: int = 1024):
    '''
    toks: torch.Tensor
        - Batched token input
    padding_idx: int
        - Index of padding token in sequence
    eos_idx: int
        - Index of EOS index
    max_protein_len: Maximum sequence length BEFORE adding CLS and EOS tokens
    '''

    cls_idx = toks[0,0].item()

    if long_protein_strategy == 'split':
        # First identify all sequences that need treatment:
        eos_loc = (toks == eos_idx)
        # Compute EOS locations:
        eos_loc = [eos_loc[i,:].nonzero(as_tuple=True)[0] for i in range(eos_loc.shape[0])]
        inds_to_split = [i for i in range(len(eos_loc)) if eos_loc[i] > (max_protein_len + 1)]
        #inds_to_split = torch.tensor(eos_over).nonzero(as_tuple=True)[0]
        #inds_to_split = (eos_loc > (max_protein_len + 1)).nonzero(as_tuple=True)[0]
        to_add_list = []
        batch_keys = list(range(toks.shape[0]))
        for i in inds_to_split:
            # Get overage amount:
            overage = (toks[i,:] == eos_idx).nonzero(as_tuple=True)[0][0].item()

            # Get number of additional seq's you'll have to make:
            num_add_splits = (overage // (max_protein_len + 1))

            for j in range(num_add_splits):
                # ***Must adjust for cls and eos tokens***
                bot = (j + 1) * max_protein_len + 1 # IGNORE first one (already in place)
                new_empty = torch.ones(1, toks.shape[1], dtype = int) * padding_idx # All fill with padding idx
                new_empty = new_empty.to(toks.device)

                # Copy over from prev. iteration
                new_tmp = toks[i,bot:].clone().unsqueeze(0)
                #print('tok, {}, {}, {}'.format(i, j, toks[i,(bot-5):(bot+5)]))
                #rem_shape = toks.shape[1] - new_tmp.shape[1]
                new_empty[0,1:(new_tmp.shape[1]+1)] = new_tmp
                new_empty[0,0] = cls_idx
                if j < (num_add_splits - 1):
                    new_empty[0,max_protein_len+1] = eos_idx # Else, the EOS index should already be there
                    new_empty[0,(max_protein_len+2):] = 1
                
                to_add_list.append(new_empty)
                # print('new_emp, {}, {}, {} (front)'.format(i, j, new_empty[0,:7]))
                # print('new_emp, {}, {}, {} (end)'.format(i, j, new_empty[0,(max_protein_len-3):(max_protein_len+5)]))
                # print('new_emp full', new_empty)
                # print('')
                batch_keys.append(i) # Ind in original toks

            
                cur_ind = toks.shape[1] + len(to_add_list)
        
            toks[i,(max_protein_len+2):] = padding_idx 
            toks[i,(max_protein_len+1)] = eos_idx

        new_toks = torch.cat([toks] + to_add_list, dim = 0)
        new_toks = new_toks[:,:(max_protein_len+2)] # Cut down size
        batch_keys = torch.tensor(batch_keys, dtype=int)


    elif long_protein_strategy == 'truncate':
        # Simple truncation:
        if toks.shape[1] > (max_protein_len + 2):
            new_toks = toks[:,:(max_protein_len+2)] # Truncate to appropriate length
            no_pad = (new_toks[:,-1] != padding_idx) # Get all samples that DO NOT have padding at the end of their seqs
            new_toks[:, no_pad] = eos_idx # Replace ending with EOS token if needed
        else:
            new_toks = toks
        batch_keys = None
        eos_loc = None
    
    #eos_loc = [(new_toks[i,:] == eos_idx).nonzero(as_tuple=True)[0] for i in range(new_toks.shape[0])]

    # for i in range(new_toks.shape[0]):
    #     print(f'new_tok {i}, ', new_toks[i,:])
    # print(new_toks.shape)

    return new_toks, batch_keys, eos_loc


def reverse_batched_split(protein_embeds, batch_keys, eos_locs: list):
    '''
    We know that protein_embeds have CLS tokens at each starting spot, EOS at each ending spot
    '''
    max_ind = batch_keys.max().item()
    full_protein_embeds = []
    #max_size = 0
    for i in range(max_ind + 1):
        iship = (batch_keys == i)
        if iship.sum() == 0: # Allow for breaks in continuously increasing integers
            continue
        iship_inds = iship.nonzero(as_tuple=True)[0].sort()[0] # SORT TO REMAIN CONSISTENT WITH batched_split_long_seq PROCESS
        # Reshape (#,S,d) -> (1,S + #,d), an essential flattening along dimension 1
        common_prot = protein_embeds[iship_inds,:,:]
        eos_inprot = torch.ones(common_prot.shape[0], common_prot.shape[1], dtype=bool)
        eos_inprot[:-1,-1] = False # All but last index in sub-batch (contains actual EOS token)
        cls_inprot = torch.ones(common_prot.shape[0], common_prot.shape[1], dtype=bool)
        cls_inprot[1:,0] = False # All but first sequence in sub-batch (contains actual CLS token)

        common_prot = common_prot.reshape(1, -1, protein_embeds.shape[-1]).squeeze(0)
        eos_inprot = eos_inprot.flatten()
        cls_inprot = cls_inprot.flatten()

        # Trim common_prot by eos_inprot and cls_inprot
        common_prot = common_prot[(eos_inprot & cls_inprot),:]

        # Remove CLS and EOS for middles
        full_protein_embeds.append(common_prot)
        #max_size = common_prot.shape[0] if common_prot.shape[0] > max_size else max_size

    # Pad to max size:
    max_size = max(eos_locs) + 1
    for i in range(len(full_protein_embeds)):
        diff_size = max_size - full_protein_embeds[i].shape[0]
        if diff_size > 0:
            tocat = torch.zeros(diff_size, protein_embeds.shape[-1]).to(protein_embeds.device)
            full_protein_embeds[i] = torch.cat([full_protein_embeds[i], tocat], dim=0)
        elif diff_size < 0:
            full_protein_embeds[i] = full_protein_embeds[i][:max_size,:]

    new_prot_embeds = torch.stack(full_protein_embeds)

    return new_prot_embeds

def concat_tensor_dict(Ld, dim = 0):
    '''
    Util fn to support concatenating an arbitrary dictionary, 
        as long as it's depth 1 and contains all tensor values.
    
    Assumes all dicts have same keys

    Args:
        Ld: list of dict of tensors
    '''
    if len(Ld) == 1:
        return Ld[0]

    a = Ld[0].keys()

    concat_dict = {k:Ld[0][k] for k in a}

    for i in range(1, len(Ld)):
        for k in a:
            concat_dict[k] = torch.cat([concat_dict[k], Ld[i][k]], dim = dim)

    return concat_dict