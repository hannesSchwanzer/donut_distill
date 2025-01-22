import torch
from torch import nn
import torch.nn.functional as F
from typing import List

mse_loss_fn = nn.MSELoss()
ce_loss_fn = nn.CrossEntropyLoss() # probably mean? becuase two outputs
kl_loss_fn = nn.KLDivLoss(reduction='batchmean')

def calculate_loss_and_accuracy(outputs: dict[str,torch.Tensor], 
                    teacher_outputs: dict[str,torch.Tensor],
                    labels: torch.Tensor,
                    is_first_distillation_phase: bool,
                    is_1phase_distillation: bool,
                    encoder_layer_map: List[List[int]],     # e.g. [[1,2], [1,2], [1,2,4,5,7,8,10,11,13,14], [1,2]] Teacher has form [2,2,14,2]
                    decoder_layer_map: List[int], # Teacher has 4 Layers
                    device) -> torch.Tensor:
    
    total_loss = torch.tensor([0.0], device=device)
    if is_first_distillation_phase or is_1phase_distillation:
        # Distillation: First one complete training only on hidden_states and attentions... in the encoder
        for stage_idx, stage in enumerate(encoder_layer_map): # TODO: Fix layer mapping
            for student_layer_number, teacher_layer_number in enumerate(stage): # TODO: Fix layer mapping
                teacher_layer_idx = teacher_layer_number # TODO: Fix
                student_layer_idx = student_layer_number # TODO: Fix
                # We distill the attention scores...
                total_loss += mse_loss_fn(outputs['attentions'][student_layer_idx], teacher_outputs['attentions'][teacher_layer_idx]) # TODO: FIX
                # ...and we distill the hidden_states, where layer indices are offset by 1 due to the embedding layer
                total_loss += mse_loss_fn(outputs['hidden_states'][student_layer_idx+1], teacher_outputs['hidden_states'][teacher_layer_idx+1]) # TODO: FIX

        # Distillation: First one complete training only on hidden_states and attentions... in the decoder
        for student_layer_idx, teacher_layer_idx in enumerate(decoder_layer_map): # TODO: Fix layer mapping
            # We distill the attention scores...
            total_loss += mse_loss_fn(outputs['attentions'][student_layer_idx], teacher_outputs['attentions'][teacher_layer_idx]) # TODO: FIX
            # ...and we distill the hidden_states, where layer indices are offset by 1 due to the embedding layer
            total_loss += mse_loss_fn(outputs['hidden_states'][student_layer_idx+1], teacher_outputs['hidden_states'][teacher_layer_idx+1]) # TODO: FIX

        # Finally we also distill the embedding layer
        total_loss += mse_loss_fn(outputs['hidden_states'][0], teacher_outputs['hidden_states'][0])

    if (not is_first_distillation_phase) or is_1phase_distillation:
        # Flatten the tokens and labels for CrossEntropyLoss
        logits: torch.Tensor = outputs['logits']

        # Distillation: ...afterwards one complete training only on logits
        # Have to use log_softmax for predictions here, see: https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
        loss_val = kl_loss_fn(F.log_softmax(logits, dim=-1), F.softmax(teacher_outputs['logits'], dim=-1))
        total_loss += loss_val

    return total_loss

