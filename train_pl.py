from transformers import (
    DonutProcessor,
    VisionEncoderDecoderModel,
    VisionEncoderDecoderConfig,
)
from donut_dataset import DonutDataset, added_tokens
from torch.utils.data import DataLoader
from pathlib import Path
import re
from nltk import edit_distance
import numpy as np
import math
from sconf import Config
import torch

from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback, EarlyStopping


TOKENIZERS_PARALLELISM = False

config = Config("./train_funsd.yaml")
donut_config = VisionEncoderDecoderConfig.from_pretrained("naver-clova-ix/donut-base")
donut_config.encoder.image_size = config.input_size
donut_config.decoder.max_length = config.max_length


processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
model = VisionEncoderDecoderModel.from_pretrained(
    "naver-clova-ix/donut-base", config=donut_config
)

train_dataset = DonutDataset(
    dataset_name_or_path="preprocessed_dataset",
    processor=processor,
    model=model,
    max_length=config.max_length,
    split="train",
    task_start_token="",
    prompt_end_token="",
    sort_json_key=False,  # cord dataset is preprocessed, so no need for this
)

val_dataset = DonutDataset(
    dataset_name_or_path="preprocessed_dataset",
    processor=processor,
    model=model,
    max_length=config.max_length,
    split="test",
    task_start_token="",
    prompt_end_token="",
    sort_json_key=False,  # cord dataset is preprocessed, so no need for this
)

print("Added tokens:", added_tokens)

pixel_values, labels, target_sequence = train_dataset[0]
print("Pixel_values", pixel_values.shape)
for id in labels.tolist()[:30]:
  if id != -100:
    print(processor.decode([id]))
  else:
    print(id)

print(target_sequence)

model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids([''])[0]

# sanity check
print("Pad token ID:", processor.decode([model.config.pad_token_id]))
print("Decoder start token ID:", processor.decode([model.config.decoder_start_token_id]))


train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

batch = next(iter(train_dataloader))
pixel_values, labels, target_sequences = batch
print("train_dataloader")
print("Pixel_values", pixel_values.shape)

for id in labels.squeeze().tolist()[:30]:
  if id != -100:
    print(processor.decode([id]))
  else:
    print(id)

batch = next(iter(val_dataloader))
pixel_values, labels, target_sequences = batch
print("val_dataloader")
print("Pixel_values", pixel_values.shape)
print(target_sequences[0])

class DonutModelPLModule(pl.LightningModule):
    def __init__(self, config, processor, model):
        super().__init__()
        self.config = config
        self.processor = processor
        self.model = model

    def training_step(self, batch, batch_idx):
        pixel_values, labels, _ = batch
        
        outputs = self.model(pixel_values, labels=labels)
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        pixel_values, labels, answers = batch
        batch_size = pixel_values.shape[0]
        # we feed the prompt to the model
        decoder_input_ids = torch.full((batch_size, 1), self.model.config.decoder_start_token_id, device=self.device)
        
        outputs = self.model.generate(pixel_values,
                                   decoder_input_ids=decoder_input_ids,
                                   max_length=config.max_length,
                                   early_stopping=True,
                                   pad_token_id=self.processor.tokenizer.pad_token_id,
                                   eos_token_id=self.processor.tokenizer.eos_token_id,
                                   use_cache=True,
                                   num_beams=1,
                                   bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                                   return_dict_in_generate=True,)
    
        predictions = []
        for seq in self.processor.tokenizer.batch_decode(outputs.sequences):
            seq = seq.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "")
            seq = re.sub(r"<.*?>", "", seq, count=1).strip()  # remove first task start token
            predictions.append(seq)

        scores = []
        for pred, answer in zip(predictions, answers):
            pred = re.sub(r"(?:(?<=>) | (?=", "", answer, count=1)
            answer = answer.replace(self.processor.tokenizer.eos_token, "")
            scores.append(edit_distance(pred, answer) / max(len(pred), len(answer)))

            if self.config.get("verbose", False) and len(scores) == 1:
                print(f"Prediction: {pred}")
                print(f"    Answer: {answer}")
                print(f" Normed ED: {scores[0]}")

        self.log("val_edit_distance", np.mean(scores))
        
        return scores

    def configure_optimizers(self):
        # you could also add a learning rate scheduler if you want
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.get("lr"))
    
        return optimizer

    def train_dataloader(self):
        return train_dataloader

    def val_dataloader(self):
        return val_dataloader


train_config = {"max_epochs":config.max_epochs,
          "val_check_interval":0.2, # how many times we want to validate during an epoch
          "check_val_every_n_epoch":1,
          "gradient_clip_val":config.gradient_clip_val,
          "num_training_samples_per_epoch": len(train_dataset),
          "lr":config.lr,
          "train_batch_sizes": [config.train_batch_sizes],
          "val_batch_sizes": [config.val_batch_sizes],
          # "seed":2022,
          "num_nodes": 1,
          "warmup_steps": config.warmup_steps, # 800/8*30/10, 10%
          "result_path": "./result",
          "verbose": True,
          }

model_module = DonutModelPLModule(train_config, processor, model)

wandb_logger = WandbLogger(project="donut-funsd", name="pl-lightning")

early_stop_callback = EarlyStopping(monitor="val_edit_distance", patience=3, verbose=False, mode="min")

trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=train_config.get("max_epochs"),
        val_check_interval=train_config.get("val_check_interval"),
        check_val_every_n_epoch=train_config.get("check_val_every_n_epoch"),
        gradient_clip_val=train_config.get("gradient_clip_val"),
        precision=16, # we'll use mixed precision
        num_sanity_val_steps=0,
        logger=wandb_logger,
        callbacks=[early_stop_callback],
)

trainer.fit(model_module)
