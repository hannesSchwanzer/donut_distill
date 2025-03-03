import torch
from torch import nn
import torch.nn.functional as F
from typing import List
from transformers.modeling_outputs import Seq2SeqLMOutput

# Define loss functions
mse_loss_fn = nn.MSELoss()
ce_loss_fn = nn.CrossEntropyLoss()
kl_loss_fn = nn.KLDivLoss(reduction='batchmean')

def calculate_loss_and_accuracy_distillation(
    outputs: Seq2SeqLMOutput,
    teacher_outputs: Seq2SeqLMOutput,
    is_first_distillation_phase: bool,
    is_1phase_distillation: bool,
    decoder_layer_map: List[int],
    device: torch.device,
    alpha: float = 1,
    beta: float = 1,
    gamma: float = 1,
    delta: float = 1
) -> torch.Tensor:
    """
    Calculate the distillation loss between teacher and student models.

    Args:
        outputs (Seq2SeqLMOutput): The student model outputs.
        teacher_outputs (Seq2SeqLMOutput): The teacher model outputs.
        is_first_distillation_phase (bool): Whether this is the first distillation phase.
        is_1phase_distillation (bool): If true, uses a combined phase for distillation.
        decoder_layer_map (List[int]): Mapping between teacher and student decoder layers.
        device (torch.device): Device (CPU or GPU).
        alpha (float): Weight for self-attention loss.
        beta (float): Weight for hidden state loss.
        gamma (float): Weight for logit-based loss.

    Returns:
        torch.Tensor: The total computed loss for backpropagation.
    """

    # Normalize alpha, beta, gamma for balanced loss contribution
    if is_1phase_distillation:
        normalize_factor = 1 / (alpha + beta + gamma + delta)
        alpha *= normalize_factor
        beta *= normalize_factor
        gamma *= normalize_factor
        delta *= normalize_factor
    elif is_first_distillation_phase:
        normalize_factor = 1 / (alpha + beta + delta)
        alpha *= normalize_factor
        beta *= normalize_factor
        delta *= normalize_factor
    else:
        gamma = 1

    total_loss = 0.0

    # Phase 1: Hidden states & attentions distillation
    if is_first_distillation_phase or is_1phase_distillation:
        for student_layer_idx, teacher_layer_idx in enumerate(decoder_layer_map):
            # Self-attention
            total_loss += safe_mse_loss(
                outputs.decoder_attentions[student_layer_idx],
                teacher_outputs.decoder_attentions[teacher_layer_idx],
                device,
                weight=(1 / len(decoder_layer_map)) * alpha
            )

            # Cross-attention
            total_loss += safe_mse_loss(
                outputs.cross_attentions[student_layer_idx],
                teacher_outputs.cross_attentions[teacher_layer_idx],
                device,
                weight=(1 / len(decoder_layer_map)) * delta
            )

            # Hidden States
            total_loss += safe_mse_loss(
                outputs.decoder_hidden_states[student_layer_idx+1],
                teacher_outputs.decoder_hidden_states[teacher_layer_idx+1],
                device,
                weight=(1 / (len(decoder_layer_map) + 1)) * beta
            )

        # Distill embedding layer
        total_loss += safe_mse_loss(
            outputs.decoder_hidden_states[0],
            teacher_outputs.decoder_hidden_states[0],
            device,
            weight=(1 / (len(decoder_layer_map) + 1)) * beta
        )

    # Phase 2: Logit-based distillation (KL divergence)
    if (not is_first_distillation_phase) or is_1phase_distillation:
        epsilon = 1e-10
        logits: torch.Tensor = outputs.logits
        loss_val = gamma * kl_loss_fn(
            F.log_softmax(logits + epsilon, dim=-1),
            F.softmax(teacher_outputs.logits + epsilon, dim=-1)
        ).to(device)
        total_loss += loss_val

    return total_loss

def safe_mse_loss(output: torch.Tensor, target: torch.Tensor, device, weight: float = 1.0) -> torch.Tensor:
    """
    Computes MSE loss and handles NaN or Inf values.
    
    Args:
        output (torch.Tensor): The output tensor from the student model.
        target (torch.Tensor): The corresponding target tensor from the teacher model.
        device: The device (CPU or CUDA) to run the computation on.
        weight (float): Scaling factor for the loss.

    Returns:
        torch.Tensor: The safe MSE loss.
    """
    loss = weight * mse_loss_fn(output, target)

    # Check for NaN or Inf
    if torch.isnan(loss).any() or torch.isinf(loss).any():
        print("NaN or Inf detected in MSE loss. Replacing with default values.")
        loss = torch.nan_to_num(loss, nan=0.0, posinf=1e5, neginf=-1e5).to(device)

    return loss
