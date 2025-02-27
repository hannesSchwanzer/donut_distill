import torch
import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import DonutProcessor, VisionEncoderDecoderModel

from donut_distill.data.donut_dataset import DonutDataset
import donut_distill.config.config as CONFIG

def prepare_dataloader(model: VisionEncoderDecoderModel, processor: DonutProcessor):
    """
    Prepare the training and validation dataloaders for model training.

    Args:
        model (VisionEncoderDecoderModel): The Donut model to be trained.
        processor (DonutProcessor): The processor for tokenizing and processing input data.

    Returns:
        tuple: A tuple containing:
            - train_dataloader (DataLoader): Dataloader for the training dataset.
            - val_dataloader (DataLoader): Dataloader for the validation dataset.
    """

    train_dataset = DonutDataset(
        dataset_name_or_path=CONFIG.DATASET,
        processor=processor,
        model=model,
        max_length=CONFIG.MAX_LENGTH,
        split=CONFIG.DATASET_NAME_TRAINING,
        task_start_token="<s_docvqa>",
        prompt_end_token="<s_answer>",
        sort_json_key=CONFIG.SORT_JSON_KEY,
        task="docvqa",
    )

    val_dataset = DonutDataset(
        dataset_name_or_path=CONFIG.DATASET,
        processor=processor,
        model=model,
        max_length=CONFIG.MAX_LENGTH,
        split=CONFIG.DATASET_NAME_VALIDATE,
        task_start_token="<s_docvqa>",
        prompt_end_token="<s_answer>",
        sort_json_key=CONFIG.SORT_JSON_KEY,
        task="docvqa",
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=CONFIG.TRAIN_BATCH_SIZES,
        shuffle=True,
        num_workers=CONFIG.NUM_WORKERS,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=CONFIG.VAL_BATCH_SIZES,
        shuffle=True,
        num_workers=CONFIG.NUM_WORKERS,
    )

    return train_dataloader, val_dataloader


def cosine_scheduler(optimizer: Optimizer, training_steps: int, warmup_steps: int) -> LambdaLR:
    """
    Creates a cosine learning rate scheduler with a linear warmup phase.

    Args:
        optimizer (Optimizer): The optimizer for which to schedule the learning rate.
        training_steps (int): Total number of training steps.
        warmup_steps (int): Number of warmup steps before cosine decay starts.

    Returns:
        LambdaLR: A learning rate scheduler following the cosine decay schedule.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            # Linearly increase the learning rate during warmup
            return current_step / max(1, warmup_steps)
        
        # Compute progress after warmup as a fraction of total training steps
        progress = (current_step - warmup_steps) / max(1, training_steps - warmup_steps)
        
        # Apply cosine decay to learning rate
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def prepare_optimizer_and_scheduler(model: VisionEncoderDecoderModel, len_trainingsdata: int) -> tuple[Optimizer, LambdaLR]:
    """
    Prepares the optimizer and learning rate scheduler for training.

    Args:
        model (VisionEncoderDecoderModel): The Donut model to be optimized.
        len_trainingsdata (int): The number of training samples.

    Returns:
        tuple: A tuple containing:
            - optimizer (Optimizer): Adam optimizer for model training.
            - scheduler (LambdaLR): Cosine learning rate scheduler.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG.LR)

    # Compute the total number of iterations based on epochs and dataset size
    if int(CONFIG.MAX_EPOCHS) > 0:
        max_iter = (CONFIG.MAX_EPOCHS * len_trainingsdata) / (
            CONFIG.TRAIN_BATCH_SIZES * torch.cuda.device_count() * CONFIG.NUM_NODES
        )

    # If max training steps are defined, ensure we do not exceed them
    if int(CONFIG.MAX_STEPS) > 0:
        max_iter = (
            min(CONFIG.MAX_STEPS, max_iter)
            if max_iter is not None
            else CONFIG.MAX_STEPS
        )

    assert max_iter is not None, "max_iter must be defined before creating the scheduler."

    scheduler = cosine_scheduler(optimizer, max_iter, CONFIG.WARMUP_STEPS)
    
    return optimizer, scheduler
