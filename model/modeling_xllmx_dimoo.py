import functools
import logging
import math
from typing import List
import torch.nn.functional as F
import torch
from torch import nn
from transformers import AutoTokenizer, AutoConfig
from .modeling_llada import LLaDAModelLM
from .configuration_llada import LLaDAConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
__all__ = ["LLaDAForMultiModalGeneration"]

def create_attention_mask(original_lengths, max_tokens, device):
    batch_size = len(original_lengths)
    attention_mask = torch.zeros(batch_size, max_tokens, dtype=torch.bool, device=device)
    for i, length in enumerate(original_lengths):
        attention_mask[i, :length] = 1  # 有效位置设为1
    return attention_mask

class LLaDAForMultiModalGeneration(LLaDAModelLM):
    config_class = LLaDAConfig
    base_model_prefix = "model"
    def __init__(self, config: LLaDAConfig, *args, **kwargs):
        print(f"Initializing MMadaModelLM with config: {config}")
        super().__init__(config, *args, **kwargs)
    
    def forward(self,
            input_ids=None, labels=None, infer=False,
            added_hidden_state=None, mask_initial=None,
            input_emb=None, attention_mask=None,
                    target_object_indices=[],
                    image_mask=None,
                    att_score_list=None,
                    use_attention_control=False,
                **kwargs):
        if input_ids != None:
            input_ids = input_ids.tolist()
            # ========================================================
            # padding input batch len & attention bias for attention mask
            # ========================================================
            max_tokens = max([len(_) for _ in input_ids])
            original_lengths = [len(example) for example in input_ids] # every sample len --> record for attention mask
            input_ids = [example + [0] * (max_tokens - len(example)) for example in input_ids] # padding 0 to right --> max length
            input_ids = torch.tensor(input_ids, dtype=torch.int64, device=self.device)
            # attn mask
            if attention_mask == None:
                attention_mask = create_attention_mask(original_lengths, max_tokens, self.device)
            else:
                raise NotImplementedError
        else:
            if attention_mask == None:
                raise NotImplementedError

        # ========================================================
        # model output 
        # ========================================================
        output = LLaDAModelLM.forward(self,
                                use_attention_control=use_attention_control,
                                target_object_indices=target_object_indices,
                                image_mask=image_mask,
                                att_score_list=att_score_list,
                input_ids=input_ids, inputs_embeds=input_emb,
                  attention_mask=attention_mask, mask_initial=mask_initial,
                  output_hidden_states=True, added_hidden_state=added_hidden_state)
        if infer:
            return output

    
    def get_fsdp_wrap_module_list(self) -> List:
        modules = [*list(self.model.transformer.blocks), self.model.transformer.ff_out]
        return modules