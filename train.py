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
from transformers import (
    DonutProcessor,
    VisionEncoderDecoderModel,
    VisionEncoderDecoderConfig,
)
from donut_dataset import DonutDataset
from tqdm import tqdm

TOKENIZERS_PARALLELISM = False


def prepare_dataloader(config, processor, model):
    training_data = DonutDataset(
        dataset_name_or_path="preprocessed_dataset",
        processor=processor,
        model=model,
        max_length=config.max_length,
        split="train",
        # task_start_token="<s_funsd>",  # TODO: Check if necessary
        sort_json_key=config.sort_json_key,
        # prompt_end_token="</s_funsd>",
    )
    test_data = DonutDataset(
        dataset_name_or_path="preprocessed_dataset",
        processor=processor,
        model=model,
        max_length=config.max_length,
        split="test",  # TODO: Check output of dataset (changes if it is train or not)
        # task_start_token="<s_funsd>",
        sort_json_key=config.sort_json_key,
        # prompt_end_token="</s_funsd>",
    )
    train_dataloader = DataLoader(
        training_data,
        batch_size=config.train_batch_sizes,
        num_workers=config.num_workers,
        pin_memory=True,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        test_data,
        batch_size=config.val_batch_sizes,
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

    donut_config = VisionEncoderDecoderConfig.from_pretrained(
        "naver-clova-ix/donut-base"
    )  # TODO: Check config
    donut_config.encoder.image_size = config.input_size
    donut_config.decoder.max_length = config.max_length

    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
    model = VisionEncoderDecoderModel.from_pretrained(
        "naver-clova-ix/donut-base", config=donut_config
    )
    model = model.to(device)

    processor.image_processor.size = config.input_size[::-1]
    processor.image_processor.do_align_long_axis = config.align_long_axis

    train_dataloader, val_dataloader = prepare_dataloader(config, processor, model)

    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(
        [""]
    )[0]

    # Setup Optimizer and Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    warmup_steps = config.warmup_steps

    max_iter = (config.max_epochs * len(train_dataloader.dataset)) / (
        config.train_batch_sizes
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
    log_dir = Path(config.result_path) / "donut_funsd"
    model_dir = log_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    wandb.init(
        project="donut-funsd",
        name="train-torch",
        config={
            "learning_rate": config.lr,
            "architecture": "Donut",
            "dataset": "funsd",
            "epochs": config.max_epochs,
            "gradient_clip_val": config.gradient_clip_val,
        },
    )

    best_val_metric = float("inf")
    # best_checkpoint_path = log_dir / "best_model.pth"
    # best_processor_path = log_dir / "best_processor.pth"

    steps = 0
    # Trainingsloop
    for epoch in range(config.max_epochs):
        if verbose:
            print(f"Epoch {epoch + 1}/{config.max_epochs}")

        # Train
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}"):
            pixel_values, labels, _ = batch
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)

            # Get loss (could also get logits, hidden_states, decoder_attentions, cross_attentions)
            optimizer.zero_grad()
            outputs = model(pixel_values, labels=labels)
            loss = outputs.loss
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_val)

            optimizer.step()

            # Accumulate loss
            train_loss += loss.item()

            # Log training metrics
            # if steps % 10 == 0:
            #     wandb.log({"train/loss": loss.item()}, step=steps)
            #     if verbose:
            #         print(f"Batch {batch_idx}, Loss: {loss.item()}")

            steps += 1

        avg_train_loss = train_loss / len(train_dataloader)
        if verbose:
            print(f"Epoch {epoch + 1} Train Loss: {avg_train_loss}")

        # VALIDATION
        model.eval()
        val_edit_distance = 0.0
        avg_val_edit_distance = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Validation Epoch {epoch+1}"):
                pixel_values, labels, answers = batch
                pixel_values = pixel_values.to(device)
                # labels = labels.to(device)
                # labels = torch.full(
                #     (config.val_batch_sizes, 1),
                #     model.config.decoder_start_token_id,
                #     device=device,
                # )
                decoder_input_ids = torch.full(
                    (config.val_batch_size, 1),
                    model.config.decoder_start_token_id,
                    device=device
                )

                outputs = model.generate(
                    pixel_values,
                    decoder_input_ids=decoder_input_ids,
                    max_length=config.max_length,
                    early_stopping=True,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    use_cache=True,
                    num_beams=1,
                    bad_words_ids=[[processor.tokenizer.unk_token_id]],
                    return_dict_in_generate=True,
                )

                # Model inference
                predictions = []
                for seq in processor.tokenizer.batch_decode(outputs.sequences):
                    seq = seq.replace(processor.tokenizer.eos_token, "").replace(
                        processor.tokenizer.pad_token, ""
                    )
                    seq = re.sub(
                        r"<.*?>", "", seq, count=1
                    ).strip()  # remove first task start token
                    predictions.append(seq)

                # Compute scores (e.g., normalized edit distance)
                scores = []
                for pred, answer in zip(predictions, answers):
                    pred = re.sub(r"(?:(?<=>) | (?=</s_))", "", pred)
                    answer = re.sub(r"<.*?>", "", answer, count=1)
                    answer = answer.replace(processor.tokenizer.eos_token, "")
                    score = edit_distance(pred, answer) / max(
                        len(pred), len(answer)
                    )
                    scores.append(score)

                    if config.get("verbose", False):
                        print(f"Prediction: {pred}")
                        print(f"    Answer: {answer}")
                        print(f" Normed ED: {scores[0]}")

                val_edit_distance += np.sum(scores)
                total_samples += len(scores)

        avg_val_edit_distance = val_edit_distance / total_samples
        if verbose:
            print(f"Epoch {epoch + 1} Validation edit distance: {avg_val_edit_distance}")
        wandb.log({
            "train/avg_loss": avg_train_loss,
            "validate/avg_edit_distance": avg_val_edit_distance,
        })

        # Save the best model
        if avg_val_edit_distance < best_val_metric:
            best_val_metric = avg_val_edit_distance
            # torch.save(
            #     {
            #         "epoch": epoch,
            #         "model_state_dict": model.state_dict(),
            #         "optimizer_state_dict": optimizer.state_dict(),
            #         "val_metric": val_metric,
            #     },
            #     best_checkpoint_path,
            # )
            model.save_pretrained(model_dir)
            processor.save_pretrained(model_dir)
            if verbose:
                print(f"Best model saved at epoch {epoch + 1} with metric {avg_val_edit_distance}")

        # Step the scheduler
        scheduler.step()

    if verbose:
        print(f"Training complete. Best validation metric: {best_val_metric}")


if __name__ == "__main__":
    train()
