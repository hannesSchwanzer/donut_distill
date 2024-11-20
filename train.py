from donut import DonutDataset, DonutModel
from sconf import Config
from torch.utils.data import DataLoader
import torch
import math
from torch.optim.lr_scheduler import LambdaLR
from pathlib import Path
from torch.utils.tensorboard.writer import SummaryWriter
from torch.nn.utils.rnn import pad_sequence
import re
import numpy as np
from editdistance import eval as edit_distance
import wandb


def prepare_dataloader(config, model):
    training_data = DonutDataset(
        dataset_name_or_path="preprocessed_dataset/training_data",
        donut_model=model,
        max_length=config.max_length,
        split="train",
        task_start_token="<s_funsd>",  # TODO: Check if necessary
        # prompt_end_token="</s_funsd>",
    )
    test_data = DonutDataset(
        dataset_name_or_path="preprocessed_dataset/testing_data",
        donut_model=model,
        max_length=config.max_length,
        split="test",  # TODO: Check output of dataset (changes if it is train or not)
        task_start_token="<s_funsd>",
        # prompt_end_token="</s_funsd>",
    )
    train_dataloader = DataLoader(
        training_data,
        batch_size=config.train_batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        test_data,
        batch_size=config.val_batch_size,
        pin_memory=True,
        shuffle=False,
    )

    return train_dataloader, val_dataloader


def train():
    config = Config("./train_funsd.yaml")
    verbose = config.verbose

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if verbose:
        print("Using device", device)


    model = DonutModel.from_pretrained(
        config.pretrained_model_name_or_path,
        input_size=config.input_size,
        max_length=config.max_length,
        align_long_axis=config.align_long_axis,
        ignore_mismatched_sizes=True,
    )
    model = model.to(device)

    train_dataloader, val_dataloader = prepare_dataloader(config, model)

    # Setup Optimizer and Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    warmup_steps = config.warmup_steps

    max_iter = (config.max_epochs * config.num_training_samples_per_epoch) / (
        config.train_batch_sizes[0]
        * torch.cuda.device_count()
        * config.get("num_nodes", 1)
    )
    if config.max_steps > 0:
        max_iter = min(config.max_steps, max_iter) if max_iter else config.max_steps

    def cosine_scheduler(optimizer, training_steps, warmup_steps):
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return current_step / max(1, warmup_steps)
            progress = current_step - warmup_steps
            progress /= max(1, training_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return LambdaLR(optimizer, lr_lambda)

    scheduler = cosine_scheduler(optimizer, max_iter, warmup_steps)

    # Setup Logging and Checkpointing
    log_dir = (
        Path(config.result_path) / "donut_funsd"
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = SummaryWriter(log_dir=log_dir)

    wandb.init(
        project="donut-funsd",
        config={
            "learning_rate": config.lr,
            "architecture": "Donut",
            "dataset": "funsd",
            "epochs": config.max_epochs,
            "gradient_clip_val": config.gradient_clip_val,
        },
    )

    best_val_metric = float("inf")
    best_checkpoint_path = log_dir / "best_model.pth"

    # Trainingsloop
    for epoch in range(config.max_epochs):
        if verbose:
            print(f"Epoch {epoch + 1}/{config.max_epochs}")

        # Train
        model.train()
        train_loss = 0.0

        for batch_idx, batch in enumerate(train_dataloader):
            image_tensors, decoder_input_ids, decoder_labels = batch

            image_tensors = torch.cat(image_tensors).to(device)
            decoder_input_ids = torch.cat(decoder_input_ids).to(device)
            decoder_labels = torch.cat(decoder_labels).to(device)

            # Get loss (could also get logits, hidden_states, decoder_attentions, cross_attentions)
            loss = model(image_tensors, decoder_input_ids, decoder_labels)[0]

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_val)

            optimizer.step()

            # Accumulate loss
            train_loss += loss.item()

            # Log training metrics
            if batch_idx % 10 == 0:
                step = epoch * len(train_dataloader) + batch_idx
                logger.add_scalar("train/loss", loss.item(), step)
                wandb.log({"train/loss": loss.item()}, step=step)
                if verbose:
                    print(f"Batch {batch_idx}, Loss: {loss.item()}")

        train_loss /= len(train_dataloader)
        if verbose:
            print(f"Epoch {epoch + 1} Train Loss: {train_loss}")

        # VALIDATION
        model.eval()
        val_loss = 0.0
        val_metric = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in val_dataloader:
                image_tensors, decoder_input_ids, prompt_end_idxs, answers = batch
                image_tensors = image_tensors.to(device)
                decoder_input_ids = decoder_input_ids.to(device)

                # Prepare decoder prompts
                decoder_prompts = pad_sequence(
                    [
                        input_id[: end_idx + 1]
                        for input_id, end_idx in zip(decoder_input_ids, prompt_end_idxs)
                    ],
                    batch_first=True,
                ).to(device)

                # Model inference
                preds = model.inference(
                    image_tensors=image_tensors,
                    prompt_tensors=decoder_prompts,
                    return_json=False,
                    return_attentions=False,
                )["predictions"]

                # Compute scores (e.g., normalized edit distance)
                scores = []
                for pred, answer in zip(preds, answers):
                    pred = re.sub(r"(?:(?<=>) | (?=</s_))", "", pred)
                    answer = re.sub(r"<.*?>", "", answer, count=1)
                    answer = answer.replace(model.decoder.tokenizer.eos_token, "")
                    score = edit_distance(pred, answer) / max(len(pred), len(answer))
                    scores.append(score)

                val_loss += np.sum(scores)
                total_samples += len(scores)

        val_metric = val_loss / total_samples
        if verbose:
            print(f"Epoch {epoch + 1} Validation Metric: {val_metric}")
        logger.add_scalar("val/metric", val_metric, epoch)
        wandb.log({"val/metric": val_metric}, step=epoch*len(train_dataloader))

        # Save the best model
        if val_metric < best_val_metric:
            best_val_metric = val_metric
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metric": val_metric,
                },
                best_checkpoint_path,
            )
            if verbose:
                print(f"Best model saved at epoch {epoch + 1} with metric {val_metric}")

        # Step the scheduler
        scheduler.step()

    # Close logger
    logger.close()
    if verbose:
        print(f"Training complete. Best validation metric: {best_val_metric}")


if __name__ == "__main__":
    train()
