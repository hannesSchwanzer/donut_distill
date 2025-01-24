from transformers import (
    DonutProcessor,
    VisionEncoderDecoderModel,
    VisionEncoderDecoderConfig,
)
import numpy as np
import torch

from transformers import GenerationConfig

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor

import donut_distill.config as CONFIG
from donut_distill.evaluate import evaluate_step_funsd
from donut_distill.train_teacher import prepare_dataloader

TOKENIZERS_PARALLELISM = False

donut_config = VisionEncoderDecoderConfig.from_pretrained(CONFIG.MODEL_ID)
donut_config.encoder.image_size = CONFIG.INPUT_SIZE
donut_config.decoder.max_length = CONFIG.MAX_LENGTH

processor = DonutProcessor.from_pretrained(CONFIG.MODEL_ID)
model = VisionEncoderDecoderModel.from_pretrained(
    CONFIG.MODEL_ID, config=donut_config
)

model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids([''])[0]

train_dataloader, val_dataloader = prepare_dataloader(model, processor)


class DonutModelPLModule(pl.LightningModule):
    def __init__(self, config, processor, model, generation_configs):
        super().__init__()
        self.processor = processor
        self.model = model
        self.config = config
        self.generation_configs = generation_configs

    def training_step(self, batch, batch_idx):
        pixel_values, labels, _ = batch
        
        outputs = self.model(pixel_values, labels=labels)
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        
        scores = dict()
        for i, (name, generation_config) in enumerate(self.generation_configs):
            f1_scores = evaluate_step_funsd(
                batch=batch,
                batch_idx=batch_idx,
                processor=self.processor,
                model=self.model,
                generation_config=generation_config
            )
            if name not in scores:
                scores[name] = []
            scores[name].append(f1_scores)
            self.log(f"f1/{name}", np.mean(f1_scores), on_epoch=True)


    # def validation_epoch_end(self, outputs):
    #     all_results = {}
    #     for output in outputs:
    #         for config_name, f1_scores in output.items():
    #             if config_name not in all_results:
    #                 all_results[config_name] = []
    #             all_results[config_name].extend(f1_scores)
    #
    #
    #     mean_f1_scores = {}
    #     for config_name, f1_list in all_results.items():
    #         mean_f1 = np.mean(f1_list) if f1_list else 0.0
    #         mean_f1_scores[config_name] = mean_f1
    #         self.log(f"{config_name}_mean_f1", mean_f1)

    def configure_optimizers(self):
        # you could also add a learning rate scheduler if you want
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.get("lr"))
    
        return optimizer

    def train_dataloader(self):
        return train_dataloader

    def val_dataloader(self):
        return val_dataloader


train_config = {"max_epochs":CONFIG.MAX_EPOCHS,
          "val_check_interval":1, # how many times we want to validate during an epoch
          "check_val_every_n_epoch":CONFIG.VALIDATE_EVERY_N_EPOCH,
          "gradient_clip_val":CONFIG.GRADIENT_CLIP_VAL,
          "num_training_samples_per_epoch": len(train_dataloader.dataset),
          "lr":CONFIG.LR,
          "train_batch_sizes": [CONFIG.TRAIN_BATCH_SIZES],
          "val_batch_sizes": [CONFIG.VAL_BATCH_SIZES],
          # "seed":2022,
          "num_nodes": CONFIG.NUM_NODES,
          "warmup_steps": CONFIG.WARMUP_STEPS, # 800/8*30/10, 10%
          "result_path": "./result",
          "verbose": True,
          }

generation_configs =  [
                ("Beam ngrams, num=5", GenerationConfig(
                    num_beams=5,
                    early_stopping=True,
                )),
                ("Beam ngrams, num=5 ngrams=8", GenerationConfig(
                    num_beams=5,
                    no_repeat_ngram_size=8,
                    early_stopping=True,
                )),
                ("Greedy", GenerationConfig(
                )),
]
model_module = DonutModelPLModule(train_config, processor, model, generation_configs)

wandb_logger = WandbLogger(project="donut-funsd", name="pl-lightning")

# early_stop_callback = EarlyStopping(monitor="val_edit_distance", patience=3, verbose=False, mode="min")
lr_monitor = LearningRateMonitor(logging_interval='step')

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
        # callbacks=[early_stop_callback],
        callbacks=[lr_monitor],
)

trainer.fit(model_module)
