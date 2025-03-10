import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict, List
from transformers.modeling_outputs import Seq2SeqLMOutput

# Define loss functions
mse_loss_fn = nn.MSELoss()
ce_loss_fn = nn.CrossEntropyLoss()
kl_loss_fn = nn.KLDivLoss(reduction="batchmean")

# Doesn't change for student, because only blocks from stages gets removed (not whole stages)
ENCODER_STAGES = 4


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
    delta: float = 1,
    encoder_weight: float = 1,
    decoder_weight: float = 1,
) -> Dict[str, torch.Tensor | float]:
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

    # Normalize encoder and decoder weights
    encoder_weight *= 1 / (encoder_weight + decoder_weight)
    decoder_weight *= 1 / (encoder_weight + decoder_weight)

    total_loss = 0.0
    self_attention_loss = 0.0
    cross_attention_loss = 0.0
    hidden_state_loss = 0.0
    logit_loss = 0.0

    # Phase 1: Hidden states & attentions distillation
    if is_first_distillation_phase or is_1phase_distillation:
        # Distill Encoder
        if encoder_weight != 0:
            encoder_self_attention_weight = encoder_weight * (1 / ENCODER_STAGES) * alpha
            encoder_hidden_state_weight = encoder_weight * (1 / (ENCODER_STAGES + 1)) * beta
            for stage_idx in range(ENCODER_STAGES):
                # Self-attention
                if encoder_self_attention_weight != 0:
                    loss = safe_mse_loss(
                        outputs.encoder_attentions[stage_idx],
                        teacher_outputs.encoder_attentions[stage_idx],
                        device,
                        weight=encoder_self_attention_weight,
                    )
                    self_attention_loss += loss.item()
                    total_loss += loss

                # Hidden States
                if encoder_hidden_state_weight != 0:
                    loss = safe_mse_loss(
                        outputs.encoder_hidden_states[stage_idx + 1],
                        teacher_outputs.encoder_hidden_states[stage_idx + 1],
                        device,
                        weight=encoder_hidden_state_weight,
                    )
                    hidden_state_loss += loss.item()
                    total_loss += loss

            # Distill embedding layer
            if encoder_hidden_state_weight != 0:
                loss = safe_mse_loss(
                    outputs.decoder_hidden_states[0],
                    teacher_outputs.decoder_hidden_states[0],
                    device,
                    weight=encoder_hidden_state_weight,
                )
                hidden_state_loss += loss.item()
                total_loss += loss

        # Distill Decoder
        if decoder_weight != 0:
            decoder_self_attention_weight = decoder_weight * (1 / len(decoder_layer_map)) * alpha
            decoder_cross_attention_weight = decoder_weight * (1 / len(decoder_layer_map)) * delta
            decoder_hidden_state_weight = decoder_weight * (1 / (len(decoder_layer_map) + 1)) * beta
            for student_layer_idx, teacher_layer_idx in enumerate(decoder_layer_map):
                # Self-attention
                if decoder_self_attention_weight != 0:
                    loss = safe_mse_loss(
                        outputs.decoder_attentions[student_layer_idx],
                        teacher_outputs.decoder_attentions[teacher_layer_idx],
                        device,
                        weight=decoder_self_attention_weight,
                    )
                    self_attention_loss += loss.item()
                    total_loss += loss

                # Cross-attention
                if decoder_cross_attention_weight != 0:
                    loss = safe_mse_loss(
                        outputs.cross_attentions[student_layer_idx],
                        teacher_outputs.cross_attentions[teacher_layer_idx],
                        device,
                        weight=decoder_cross_attention_weight,
                    )
                    cross_attention_loss += loss.item()
                    total_loss += loss

                # Hidden States
                if decoder_hidden_state_weight != 0:
                    loss = safe_mse_loss(
                        outputs.decoder_hidden_states[student_layer_idx + 1],
                        teacher_outputs.decoder_hidden_states[teacher_layer_idx + 1],
                        device,
                        weight=decoder_hidden_state_weight,
                    )
                    hidden_state_loss += loss.item()
                    total_loss += loss

            # Distill embedding layer
            if decoder_hidden_state_weight != 0:
                loss = safe_mse_loss(
                    outputs.decoder_hidden_states[0],
                    teacher_outputs.decoder_hidden_states[0],
                    device,
                    weight=decoder_self_attention_weight,
                )
                hidden_state_loss += loss.item()
                total_loss += loss

    # Phase 2: Logit-based distillation (KL divergence)
    if (not is_first_distillation_phase) or is_1phase_distillation:
        epsilon = 1e-10 # in case one logit is 0
        logits: torch.Tensor = outputs.logits
        loss = gamma * kl_loss_fn(
            F.log_softmax(logits + epsilon, dim=-1),
            F.softmax(teacher_outputs.logits + epsilon, dim=-1),
        ).to(device)
        logit_loss += loss.item()
        total_loss += loss
    
    return {
        "total_loss": total_loss,
        "losses/self_attention_loss": self_attention_loss,
        "losses/cross_attention_loss": cross_attention_loss,
        "losses/hidden_state_loss": hidden_state_loss,
        "losses/logit_loss": logit_loss,
    }


def safe_mse_loss(
    output: torch.Tensor, target: torch.Tensor, device, weight: float = 1.0
) -> torch.Tensor:
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
