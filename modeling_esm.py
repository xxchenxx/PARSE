#import esm
import torch
from torch import nn

from transformers.models.esm.modeling_esm import *

from peft import PeftModelForTokenClassification

from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

from flash_attn import flash_attn_func
from model_utils import batched_split_long_seq, reverse_batched_split, concat_tensor_dict
import esm

from transformers import (
    AutoModelForMaskedLM,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)

from functools import partial


def set_attention_type(model: nn.Module, attention_type: str):
    for mn, m in model.named_modules():
        if hasattr(m, 'attention_type'):
            # if type(m.attention_type) is str:
            # print("[ATTENTION]", mn)
            m.attention_type = attention_type
    # exit()
    return model

class ESMPrefix(nn.Module):
    def __init__(self, num_hidden_layers, decoder_attention_heads, d_model, prefix_dropout=0.0, prefix_attn_bn='concat', prefix_attn_composition="add", prefix_mid_dim=800):
        super().__init__()

        # self.match_n_layer = config.decoder_layers if args.num_bias_layers < 0 else args.num_bias_layers
        self.match_n_layer = num_hidden_layers
        self.match_n_head = decoder_attention_heads
        self.n_embd = d_model

        self.match_n_embd = self.n_embd // self.match_n_head

        self.mid_dim = prefix_mid_dim
        self.attn_bn = prefix_attn_bn
        self.prefix_dropout = prefix_dropout

        self.dropout = nn.Dropout(self.prefix_dropout)

        self.input_tokens = torch.arange(self.attn_bn).long()
        self.wte = nn.Embedding(self.attn_bn, self.n_embd)
        self.control_trans = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))

        self.wte_enc = nn.Embedding(self.attn_bn, self.n_embd)
        self.control_trans_enc = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))

        self.wte2 = nn.Embedding(self.attn_bn, self.n_embd)
        self.control_trans2 = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))

        # if args.lisa_option == "cross_attn":
        #     self.apply(init_lisa_params)

    def forward(self, bsz, nsamples=1):
        old_bsz = bsz
        bsz = bsz * nsamples
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1).to(self.wte2.weight.device)
        temp_control = self.wte(input_tokens)
        past_key_values = self.control_trans(temp_control) #bsz, seqlen, layer*emb
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                               self.match_n_embd)
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)

        temp_control2 = self.wte2(input_tokens)
        past_key_values2 = self.control_trans2(temp_control2)  # bsz, seqlen, layer*emb
        past_key_values2 = past_key_values2.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                                 self.match_n_embd)
        past_key_values2 = self.dropout(past_key_values2)
        past_key_values2 = past_key_values2.permute([2, 0, 3, 1, 4]).split(2)

        input_tokens_enc = self.input_tokens.unsqueeze(0).expand(old_bsz, -1).to(self.wte2.weight.device)
        temp_control_enc = self.wte_enc(input_tokens_enc)
        past_key_values_enc = self.control_trans_enc(temp_control_enc)  # bsz, seqlen, layer*emb
        bsz_enc, seqlen, _ = past_key_values_enc.shape
        past_key_values_enc = past_key_values_enc.view(bsz_enc, seqlen, self.match_n_layer * 2, self.match_n_head,
                                                       self.match_n_embd)
        past_key_values_enc = self.dropout(past_key_values_enc)
        past_key_values_enc = past_key_values_enc.permute([2, 0, 3, 1, 4]).split(2)

        result = []
        for i, key_val in enumerate(past_key_values):
            temp_dict = {'self': {"prev_key": key_val[0].contiguous().view(bsz*self.match_n_head, -1, self.match_n_embd),
                                  "prev_value": key_val[1].contiguous().view(bsz*self.match_n_head, -1, self.match_n_embd),
                                  "prev_key_padding_mask": torch.zeros(bsz, seqlen).to(key_val.device) #bsz, attn_bn
                                  },
                         }

            key_val2 = past_key_values2[i]
            temp_dict['encoder_decoder'] = {"prev_key": key_val2[0].contiguous().view(bsz*self.match_n_head, -1, self.match_n_embd),
                                            "prev_value": key_val2[1].contiguous().view(bsz*self.match_n_head, -1, self.match_n_embd),
                                            "prev_key_padding_mask": torch.zeros(bsz, seqlen).to(key_val2.device)
                                            }
            key_val_enc = past_key_values_enc[i]
            # at generation time, this is expanded automatically to the beam size
            temp_dict['encoder'] = {"prev_key": key_val_enc[0].contiguous().view(old_bsz*self.match_n_head, -1, self.match_n_embd),
                                    "prev_value": key_val_enc[1].contiguous().view(old_bsz*self.match_n_head, -1, self.match_n_embd),
                                    "prev_key_padding_mask": torch.zeros(bsz_enc, seqlen).to(key_val_enc.device)
                                    }
            result.append(temp_dict)
        return result

class ProteinPooler(nn.Module):
    '''
    Similar setup to PfamPooler but with different options
    - Uses flattening along sequence dimension before pooling multiple tokens on batch key
    '''
    def __init__(self, pooling_method = 'mean'):
        super(ProteinPooler, self).__init__()
        self.pooling_method = pooling_method.lower()

        if self.pooling_method == 'max':
            self.pooler = lambda x: x.max(dim=-2)[0]
        elif self.pooling_method == 'mean':
            self.pooler = lambda x: x.nanmean(dim=-2)
        elif self.pooling_method == 'cls_token': # IF YOU USE SPLITTING OF SEQUENCES, DO NOT USE CLS TOKEN
            self.pooler = lambda x: x[:,0,:] # TODO: Can pool across multiple sub-splits of proteins by max/mean pooling of CLS tokens
        else:
            raise NotImplementedError(f'Protein pooling method {self.pooling_method} is not implemented')

    def forward(self, protein_embeds, batch_keys = None, padding_mask = None):
        # start = time.time()
        if (padding_mask is not None) and (self.pooling_method == 'max'):
            protein_embeds[padding_mask] = -float("inf") # Set to neg inf so that max ignores it
        if batch_keys is not None:
            max_ind = batch_keys.max().item()
            pooled_reps = []
            for i in range(max_ind + 1):
                iship = (batch_keys == i)
                if iship.sum() == 0: # Allow for breaks in continuously increasing integers
                    continue
                # Reshape (#,S,d) -> (1,S + #,d), an essential flattening along dimension 1
                common_prot = protein_embeds[batch_keys == i,:,:].reshape(1, -1, protein_embeds.shape[-1]).squeeze(0)
                rep = self.pooler(common_prot)
                pooled_reps.append(rep)
        
            result1 = torch.stack(pooled_reps)
        # end = time.time()
        # print(f'Pooler time: {end - start}')
        
        # start = time.time()
        # if padding_mask is not None:
        #     protein_embeds[padding_mask] = -float("inf") # Set to neg inf so that max ignores it
        # if batch_keys is not None:
        #     # Get indices of unique batch keys and their counts
        #     start = time.time()
        #     pooled_reps = scatter_max(protein_embeds, batch_keys.to(protein_embeds.device), dim=0, out=-torch.ones(torch.unique(batch_keys).shape[0], protein_embeds.shape[1], protein_embeds.shape[-1]).to(protein_embeds.device)*1e7)[0].max(1)[0]
        #     end = time.time()
        #     print(f'Pooler 3 time: {end - start}')    
            
        # result3 = pooled_reps
        
        # if batch_keys is not None:
        #     start = time.time()
        #     # Get indices of unique batch keys and their counts
        #     unique_keys, counts = batch_keys.unique(return_counts=True)
        #     end = time.time()
        #     print(f'unique key time: {end - start}')

        #     # Compute pooled representations for each unique batch key
        #     start = time.time()
        #     start_indices = torch.cat((torch.tensor([0]), counts[:-1].cumsum(dim=0)))
        #     end_indices = counts.cumsum(dim=0)
        #     end = time.time()
        #     print(f'indices time: {end - start}')
            
        #     start = time.time()
        #     pooled_reps = torch.stack([protein_embeds[start:end].masked_select(~padding_mask[start:end].unsqueeze(-1).repeat(1, 1, protein_embeds.shape[-1])).reshape(-1, protein_embeds.shape[-1]).max(0)[0] for start, end in zip(start_indices, end_indices)])
        #     end = time.time()
        #     print(f'pool time: {end - start}')
            
        #     # return pooled_reps
        
        # # else:
        #     # return self.pooler(protein_embeds)
        
        # result2 = pooled_reps
        
        return result1

class EsmSelfOutputQuant(EsmSelfOutput):
    def __init__(self, config):
        super().__init__(config)
    
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # print(hidden_states.dtype, input_tensor.dtype)
        # hidden_states += input_tensor
        ret_hidden_states = hidden_states + input_tensor
        return ret_hidden_states

class EsmOutputQuant(EsmOutput):
    def __init__(self, config):
        super().__init__(config)
        
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # hidden_states += input_tensor
        ret_hidden_states = hidden_states + input_tensor
        return ret_hidden_states

class EsmSelfAttentionQuat(EsmSelfAttention):
    def __init__(self, config, position_embedding_type=None):
        super().__init__(config, position_embedding_type)
        self.attention_probs_dropout_prob = config.attention_probs_dropout_prob
        self.attention_type = 'vanilla'
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Matt: Our BERT model (which this code was derived from) scales attention logits down by sqrt(head_dim).
        # ESM scales the query down by the same factor instead. Modulo numerical stability these are equivalent,
        # but not when rotary embeddings get involved. Therefore, we scale the query here to match the original
        # ESM code and fix rotary embeddings.
        query_layer = query_layer * self.attention_head_size**-0.5

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        if self.position_embedding_type == "rotary":
            query_layer, key_layer = self.rotary_embeddings(query_layer, key_layer)

        if self.attention_type == 'vanilla':
            # Take the dot product between "query" and "key" to get the raw attention scores.
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

            if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
                seq_length = hidden_states.size()[1]
                position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
                position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
                distance = position_ids_l - position_ids_r
                positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
                positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

                if self.position_embedding_type == "relative_key":
                    relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                    attention_scores = attention_scores + relative_position_scores
                elif self.position_embedding_type == "relative_key_query":
                    relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                    relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                    attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

            if attention_mask is not None:
                # Apply the attention mask is (precomputed for all layers in EsmModel forward() function)
                attention_scores = attention_scores + attention_mask

            # Normalize the attention scores to probabilities.
            attention_probs = nn.functional.softmax(attention_scores, dim=-1)

            # This is actually dropping out entire tokens to attend to, which might
            # seem a bit unusual, but is taken from the original Transformer paper.
            attention_probs = self.dropout(attention_probs)

            # Mask heads if we want to
            if head_mask is not None:
                attention_probs = attention_probs * head_mask

            context_layer = torch.matmul(attention_probs, value_layer)
            
        elif self.attention_type == 'flash_attn':
            # print("USING FLASH ATTENTION IN ESM!!!")
            context_layer, attention_probs, _ = flash_attn_func(query_layer, key_layer, value_layer, dropout_p=self.attention_probs_dropout_prob, causal=True, return_attn_probs=True)
        else:
            raise NotImplementedError#, f"unsupported attention type {self.attention_type}"

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs
        

class EsmAttentionQuant(EsmAttention):
    def __init__(self, config):
        super().__init__(config)
        self.output = EsmSelfOutputQuant(config)
        self.self = EsmSelfAttentionQuat(config)

class EsmLayerQuant(EsmLayer):
    def __init__(self, config):
        super().__init__(config)
        self.attention = EsmAttentionQuant(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise RuntimeError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = EsmAttentionQuant(config)
        self.output = EsmOutputQuant(config)

class ESMEncoderQuant(EsmEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.ModuleList([EsmLayerQuant(config) for _ in range(config.num_hidden_layers)])
    
class EsmModelQuant(EsmModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        self.encoder = ESMEncoderQuant(config)
        
class EsmForMaskedLMQuant(EsmForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        self.esm = EsmModelQuant(config, add_pooling_layer=False)
        

class ESM_PLM(torch.nn.Module):
    def __init__(
        self, 
        pretrained_weights_dir,
        num_params = '3b', 
        padding_idx = 1, 
        eos_idx = 2,
        max_protein_len = 1024,
        long_protein_strategy = 'split',
        max_batch_forward_pass = None,
        use_lora = False, 
        use_q_lora = False,
        lora_alpha = 8, 
        lora_r = 8,
        use_adapter = False,
        adapter_rank = 8,
        use_prefix=False,
        prefix_dropout=0.0,
        prefix_mid_dim=800,
        prefix_attn_bn=30,
        protein_attention_type = 'vanilla',
        esm_checkpoint = None,

    ):
        super(ESM_PLM, self).__init__()
        
        self.num_params = num_params.lower()
        self.padding_idx = padding_idx
        # self.pooler = ProteinPooler(pooling_method = self.pooling_method)
        self.long_protein_strategy = long_protein_strategy
        self.padding_idx, self.eos_idx = padding_idx, eos_idx
        self.max_protein_len = max_protein_len
        self.max_batch_forward_pass = max_batch_forward_pass

        self.use_prefix = use_prefix
        
        
        self.seq_proc = partial(batched_split_long_seq, 
            padding_idx = self.padding_idx,
            eos_idx = self.eos_idx,
            long_protein_strategy = self.long_protein_strategy,
            max_protein_len = self.max_protein_len)

        extra_model_kwargs = {
            'use_lora': use_lora, 
            'lora_alpha': lora_alpha, 
            'lora_r': lora_r,
            'use_adapter': use_adapter,
            'adapter_rank': adapter_rank
        }
        

        if self.num_params == '8m':
            # self.model, _ = esm.pretrained.esm2_t6_8M_UR50D()
            self.model, _ = esm.pretrained.load_model_and_alphabet_local(pretrained_weights_dir+'/esm2_t6_8M_UR50D.pt')
            self.repr_layer = 6
            self.embedding_size = 320
        elif self.num_params == '35m':
            # self.model, _ = esm.pretrained.esm2_t12_35M_UR50D()
            self.model, _ = esm.pretrained.load_model_and_alphabet_local(pretrained_weights_dir+'/esm2_t12_35M_UR50D.pt')
            self.repr_layer = 12
            self.embedding_size = 480
        elif self.num_params == '650m':
            self.model, _ = esm.pretrained.esm2_t33_650M_UR50D(**extra_model_kwargs)
            # self.model, _ = esm.pretrained.load_model_and_alphabet_local(pretrained_weights_dir+'/esm2_t33_650M_UR50D.pt')
            self.repr_layer = 33
            self.embedding_size = 1280
        elif self.num_params == '3b':
            self.model, _ = esm.pretrained.esm2_t36_3B_UR50D(**extra_model_kwargs)
            # FIXME: This local loading is not working: https://github.com/facebookresearch/esm/discussions/514.  Investigate later.
            # self.model, _ = esm.pretrained.load_model_and_alphabet_local(pretrained_weights_dir+'/esm2_t36_3B_UR50D.pt')
            self.repr_layer = 36
            self.embedding_size = 2560
        elif self.num_params == '15b':
            self.model, _ = esm.pretrained.esm2_t48_15B_UR50D(**extra_model_kwargs)
            # self.model, _ = esm.pretrained.load_model_and_alphabet_local(pretrained_weights_dir+'/esm2_t48_15B_UR50D.pt')
            self.repr_layer = 48
            self.embedding_size = 5120
        elif 'official' in self.num_params:
            if '650m' in self.num_params:
                model_name = "facebook/esm2_t33_650M_UR50D"
                self.repr_layer = 33
                self.embedding_size = 1280
            elif '35m' in self.num_params:
                model_name = "facebook/esm2_t12_35M_UR50D"
                self.repr_layer = 12
                self.embedding_size = 480
            elif '3b' in self.num_params:
                model_name = "facebook/esm2_t36_3B_UR50D"
                self.repr_layer = 36
                self.embedding_size = 2560
            elif '15b' in self.num_params:
                model_name = "facebook/esm2_t48_15B_UR50D"
                self.repr_layer = 48
                self.embedding_size = 5120
            else:
                model_name = "facebook/esm2_t30_150M_UR50D"
                self.repr_layer = 30
                self.embedding_size = 640
            
            peft_config = LoraConfig(
                target_modules='.*11.*(query|key|value)$',
                task_type=TaskType.TOKEN_CLS, 
                inference_mode=False, 
                r=lora_r, 
                lora_alpha=lora_alpha, 
                # target_modules=["query", "key", "value"], # also try "dense_h_to_4h" and "dense_4h_to_h"
                lora_dropout=0.1,
                bias="none" # or "all" or "lora_only" 
            )
            
            if use_q_lora and not use_lora:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit = True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type='nf4',
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
            else:
                bnb_config = None
            
            
            self.model = EsmForMaskedLMQuant.from_pretrained(model_name, quantization_config=bnb_config)
            self.model = set_attention_type(self.model, protein_attention_type)
            # self.model.encoder.layer
            
            # self.model.gradient_checkpointing_enable()
            # self.model.enable_input_require_grads()
            if use_lora and not use_q_lora:
                self.model = get_peft_model(self.model, peft_config)
            elif use_q_lora:
                
                self.model = prepare_model_for_kbit_training(self.model, use_gradient_checkpointing=False)
                
                self.model = get_peft_model(self.model, peft_config)
            
            if esm_checkpoint is not None:
                self.model.load_adapter(esm_checkpoint)

                # self.model = self.model.to(dtype=torch.float16)
            # self.model.esm.encoder.layer
            # if model_splitting:
            #     self.model.model.esm.encoder.layer = PipelineModuleEsm(layers = self.model.model.esm.encoder.layer,
            #                                                        num_stages = n_model_pieces)
            
            
        else:
            
            raise ValueError(f'ESM model with {self.num_params} parameters is not implemented')

        if self.use_prefix:
            self.prefix_model = ESMPrefix(self.repr_layer, self.model.attention_heads, self.embedding_size, prefix_dropout, prefix_attn_bn, prefix_mid_dim=prefix_mid_dim)

    def forward(self, input_ids, attention_mask):
        # Modified forward from ESM_PLM_basic
        # ESM API to process forward pass
        # IF aggregate=True, return is shape (B,E), else (B,len,E)

        # Split into chunks here ------:
        
        prefix_states = {'self': None}
            # FIXME
            # Everything passed in one batch
        if "official" in self.num_params:
                results = self.model(input_ids, attention_mask=attention_mask, output_hidden_states = True)
                results['representations'] = results['hidden_states']
        else:
                results = self.model(input_ids, attention_mask=attention_mask, repr_layers=[self.repr_layer], return_contacts = False)

        z = results['representations'][self.repr_layer]
        # print(results['representations'].keys())
        # z = results['representations'][-1]

        return z[:, 1:-1]