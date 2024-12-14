from transformers import (
    DonutProcessor,
    VisionEncoderDecoderModel,
    VisionEncoderDecoderConfig,
)
from donut_dataset import DonutDataset
from torch.utils.data import DataLoader
import re
from nltk import edit_distance
import numpy as np
from sconf import Config
import torch
import wandb
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence

TOKENIZERS_PARALLELISM = False

def prepare_dataloader(config, model, processor):
    train_dataset = DonutDataset(
        dataset_name_or_path="preprocessed_dataset",
        processor=processor,
        model=model,
        max_length=config.max_length,
        split="train",
        task_start_token="<s_funsd>",
        sort_json_key=False,  # cord dataset is preprocessed, so no need for this
    )

    val_dataset = DonutDataset(
        dataset_name_or_path="preprocessed_dataset",
        processor=processor,
        model=model,
        max_length=config.max_length,
        split="test",
        task_start_token="<s_funsd>",
        sort_json_key=False,  # cord dataset is preprocessed, so no need for this
    )

    train_dataloader = DataLoader(train_dataset, batch_size=config.train_batch_sizes, shuffle=True, num_workers=config.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=config.val_batch_sizes, shuffle=False, num_workers=config.num_workers)
    
    return train_dataloader, val_dataloader

def train():
    config = Config("./train_funsd.yaml")

    donut_config = VisionEncoderDecoderConfig.from_pretrained("naver-clova-ix/donut-base")
    donut_config.encoder.image_size = config.input_size
    donut_config.decoder.max_length = config.max_length

    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
    model = VisionEncoderDecoderModel.from_pretrained(
        "naver-clova-ix/donut-base", config=donut_config
    )

    processor.image_processor.size = config.input_size[::-1]
    processor.image_processor.do_align_long_axis = False

    train_dataloader, val_dataloader = prepare_dataloader(config, model, processor)

    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids([""])[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("lr"))
    scaler = torch.amp.GradScaler('cuda')

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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = Path(config.result_path) / f"donut_{timestamp}" / "model"
    processor_dir = Path(config.result_path) / f"donut_{timestamp}" / "processor"
    
    best_val_metric = float("inf")
    steps = 0
    for epoch in range(config["max_epochs"]):
        # Training phase
        model.train()
        losses = []
        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}"):
            pixel_values, labels, answers = batch
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)

            with torch.autocast(device_type="cuda"):
                outputs = model(pixel_values, labels=labels)
                loss = outputs.loss
            optimizer.zero_grad()
            # loss.backward()
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_val"])
            scaler.step(optimizer)
            scaler.update()
            # optimizer.step()
            losses.append(loss.item())

            # Log training metrics
            if steps % 10 == 0:
                wandb.log({"train/loss": loss.item()}, step=steps)
            steps += 1

        avg_train_loss = np.mean(losses)

        val_scores = []
        model.eval()
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validate"):
                pixel_values, decoder_input_ids, prompt_end_idxs, answers = batch
                pixel_values = pixel_values.to(device)
                batch_size = pixel_values.shape[0]
                # we feed the prompt to the model
                # decoder_input_ids = torch.full(
                #     (batch_size, 1), model.config.decoder_start_token_id, device=device
                # )
                with torch.autocast(device_type="cuda"):
                    decoder_prompts = pad_sequence(
                        [input_id[: end_idx + 1] for input_id, end_idx in zip(decoder_input_ids, prompt_end_idxs)],
                        batch_first=True,
                    ).to(device)

                    outputs = model.generate(
                        pixel_values,
                        decoder_input_ids=decoder_prompts,
                        max_length=config.max_length,
                        early_stopping=True,
                        pad_token_id=processor.tokenizer.pad_token_id,
                        eos_token_id=processor.tokenizer.eos_token_id,
                        use_cache=True,
                        num_beams=1,
                        bad_words_ids=[[processor.tokenizer.unk_token_id]],
                        return_dict_in_generate=True,
                    )

                    predictions = []
                    for seq in processor.tokenizer.batch_decode(outputs.sequences):
                        seq = seq.replace(processor.tokenizer.eos_token, "").replace(
                            processor.tokenizer.pad_token, ""
                        )
                        seq = re.sub(
                            r"<.*?>", "", seq, count=1
                        ).strip()  # remove first task start token
                        predictions.append(seq)

                    scores = []
                    for pred, answer in zip(predictions, answers):
                        pred = re.sub(r"(?:(?<=>) | (?=</s_))", "", pred, count=1)
                        answer = re.sub(r"<.*?>", "", answer, count=1)
                        answer = answer.replace(processor.tokenizer.eos_token, "")
                        scores.append(
                            edit_distance(pred, answer) / max(len(pred), len(answer))
                        )

                        if config.get("verbose", False) and len(scores) == 1:
                            print("\n----------------------------------------\n")
                            print(f"\nPrediction: {pred}")
                            print(f"\n\tAnswer: {answer}")
                            print(f"\n\tNormed ED: {scores[0]}")

                val_scores.extend(scores)

        avg_val_score = np.mean(val_scores)
        wandb.log(
            {
                "train/avg_loss": avg_train_loss,
                "validate/avg_edit_distance": avg_val_score,
            }, step=steps
        )


        if avg_val_score < best_val_metric:
            print("Saving Model!")
            best_val_metric = avg_val_score
            model.save_pretrained(model_dir)
            processor.save_pretrained(processor_dir)

if __name__ == "__main__":
    train()
