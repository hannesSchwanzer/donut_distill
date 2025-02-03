#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/Donut/DocVQA/Fine_tune_Donut_on_DocVQA.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Fine-tune Donut 🍩 on DocVQA
# 
# In this notebook, we'll fine-tune Donut (which is an instance of [`VisionEncoderDecoderModel`](https://huggingface.co/docs/transformers/model_doc/vision-encoder-decoder)) on a DocVQA dataset, which is a dataset consisting of (document, question, answer(s)) triplets. This way, the model will learn to look at an image, and answer a question related to the document. Pretty cool, isn't it?
# 
# ## Set-up environment
# 
# First, let's install the relevant libraries:
# * 🤗 Transformers, for the model
# * 🤗 Datasets, for loading + processing the data
# * PyTorch Lightning, for training the model
# * Weights and Biases, for logging metrics during training
# * Sentencepiece, used for tokenization.

# In[1]:




# In[2]:




# In[3]:




# ## Load dataset
# 
# Next, let's load the dataset from the [hub](https://huggingface.co/datasets/naver-clova-ix/cord-v2). We're prepared a minimal dataset for DocVQA, the notebook for that can be found [here](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Donut/DocVQA/Creating_a_toy_DocVQA_dataset_for_Donut.ipynb).
# 
# Important here is that we've added a "ground_truth" column, containing the ground truth JSON which the model will learn to generate.

# In[4]:


from datasets import load_dataset

# dataset = load_dataset("nielsr/docvqa_1200_examples_donut")


# As can be seen, the dataset contains a training and test split, and each example consists of an image, a question ("query"), and one or more answers.

# In[5]:


# dataset


# ## Load model and processor
# 
# Next, we load the model (which is an instance of [VisionEncoderDecoderModel](https://huggingface.co/docs/transformers/model_doc/vision-encoder-decoder), and the processor, which is the object that can be used to prepare inputs for the model.

# In[6]:


from transformers import VisionEncoderDecoderConfig

max_length = 128
image_size = [1280, 960]

# update image_size of the encoder
# during pre-training, a larger image size was used
config = VisionEncoderDecoderConfig.from_pretrained("naver-clova-ix/donut-base")
config.encoder.image_size = image_size # (height, width)
# update max_length of the decoder (for generation)
config.decoder.max_length = max_length
# TODO we should actually update max_position_embeddings and interpolate the pre-trained ones:
# https://github.com/clovaai/donut/blob/0acc65a85d140852b8d9928565f0f6b2d98dc088/donut/model.py#L602


# In[7]:


from transformers import DonutProcessor, VisionEncoderDecoderModel, BartConfig

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base", config=config)


# ## Add special tokens
# 
# For DocVQA, we add special tokens for \<yes> and \<no/>, to make sure that the model (actually the decoder) learns embedding vectors for those explicitly.

# In[28]:


from typing import List

def add_tokens(list_of_tokens: List[str]):
    """
    Add tokens to tokenizer and resize the token embeddings
    """
    newly_added_num = processor.tokenizer.add_tokens(list_of_tokens)
    if newly_added_num > 0:
        model.decoder.resize_token_embeddings(len(processor.tokenizer))


# In[29]:


additional_tokens = ["<yes/>", "<no/>"]

add_tokens(additional_tokens)


# ## Create PyTorch dataset
# 
# Here we create a regular PyTorch dataset.
# 
# The model doesn't directly take the (image, JSON) pairs as input and labels. Rather, we create `pixel_values`, `decoder_input_ids` and `labels`. These are all PyTorch tensors. The `pixel_values` are the input images (resized, padded and normalized), the `decoder_input_ids` are the decoder inputs, and the `labels` are the decoder targets.
# 
# The reason we create the `decoder_input_ids` explicitly here is because otherwise, the model would create them automatically based on the `labels` (by prepending the decoder start token ID, replacing -100 tokens by padding tokens). The reason for that is that we don't want the model to learn to generate the entire prompt, which includes the question. Rather, we only want it to learn to generate the answer. Hence, we'll set the labels of the prompt tokens to -100.
# 
# 

# In[30]:


import json
import random
from typing import Any, List, Tuple

import torch
from torch.utils.data import Dataset

added_tokens = []

class DonutDataset(Dataset):
    """
    DonutDataset which is saved in huggingface datasets format. (see details in https://huggingface.co/docs/datasets)
    Each row, consists of image path(png/jpg/jpeg) and gt data (json/jsonl/txt),
    and it will be converted into input_tensor(vectorized image) and input_ids(tokenized string).
    Args:
        dataset_name_or_path: name of dataset (available at huggingface.co/datasets) or the path containing image files and metadata.jsonl
        max_length: the max number of tokens for the target sequences
        split: whether to load "train", "validation" or "test" split
        ignore_id: ignore_index for torch.nn.CrossEntropyLoss
        task_start_token: the special token to be fed to the decoder to conduct the target task
        prompt_end_token: the special token at the end of the sequences
        sort_json_key: whether or not to sort the JSON keys
    """

    def __init__(
        self,
        dataset_name_or_path: str,
        max_length: int,
        split: str = "train",
        ignore_id: int = -100,
        task_start_token: str = "<s>",
        prompt_end_token: str = None,
        sort_json_key: bool = True,
        task: str = "",
    ):
        super().__init__()

        self.max_length = max_length
        self.split = split
        self.ignore_id = ignore_id
        self.task_start_token = task_start_token
        self.prompt_end_token = prompt_end_token if prompt_end_token else task_start_token
        self.sort_json_key = sort_json_key
        self.task = task

        self.dataset = load_dataset(dataset_name_or_path, split=self.split)
        self.dataset_length = len(self.dataset)

        self.gt_token_sequences = []
        for sample in self.dataset:
            ground_truth = json.loads(sample["ground_truth"])
            if "gt_parses" in ground_truth:  # when multiple ground truths are available, e.g., docvqa
                assert isinstance(ground_truth["gt_parses"], list)
                gt_jsons = ground_truth["gt_parses"]
            else:
                assert "gt_parse" in ground_truth and isinstance(ground_truth["gt_parse"], dict)
                gt_jsons = [ground_truth["gt_parse"]]

            self.gt_token_sequences.append(
                [
                    self.json2token(
                        gt_json,
                        update_special_tokens_for_json_key=self.split == "train",
                        sort_json_key=self.sort_json_key,
                    )
                    + processor.tokenizer.eos_token
                    for gt_json in gt_jsons  # load json from list of json
                ]
            )

        self.add_tokens([self.task_start_token, self.prompt_end_token])
        self.prompt_end_token_id = processor.tokenizer.convert_tokens_to_ids(self.prompt_end_token)

    def json2token(self, obj: Any, update_special_tokens_for_json_key: bool = True, sort_json_key: bool = True):
        """
        Convert an ordered JSON object into a token sequence
        """
        if type(obj) == dict:
            if len(obj) == 1 and "text_sequence" in obj:
                return obj["text_sequence"]
            else:
                output = ""
                if sort_json_key:
                    keys = sorted(obj.keys(), reverse=True)
                else:
                    keys = obj.keys()
                for k in keys:
                    if update_special_tokens_for_json_key:
                        self.add_tokens([fr"<s_{k}>", fr"</s_{k}>"])
                    output += (
                        fr"<s_{k}>"
                        + self.json2token(obj[k], update_special_tokens_for_json_key, sort_json_key)
                        + fr"</s_{k}>"
                    )
                return output
        elif type(obj) == list:
            return r"<sep/>".join(
                [self.json2token(item, update_special_tokens_for_json_key, sort_json_key) for item in obj]
            )
        else:
            obj = str(obj)
            if f"<{obj}/>" in added_tokens:
                obj = f"<{obj}/>"  # for categorical special tokens
            return obj
    
    def add_tokens(self, list_of_tokens: List[str]):
        """
        Add special tokens to tokenizer and resize the token embeddings of the decoder
        """
        newly_added_num = processor.tokenizer.add_tokens(list_of_tokens)
        if newly_added_num > 0:
            model.decoder.resize_token_embeddings(len(processor.tokenizer))
            added_tokens.extend(list_of_tokens)
    
    def __len__(self) -> int:
        return self.dataset_length - 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load image from image_path of given dataset_path and convert into input_tensor and labels
        Convert gt data into input_ids (tokenized string)
        Returns:
            input_tensor : preprocessed image
            input_ids : tokenized gt_data
            labels : masked labels (model doesn't need to predict prompt and pad token)
        """
        sample = self.dataset[idx]

        # input_tensor
        pixel_values = processor(sample["image"].convert("RGB"), random_padding=self.split == "train", return_tensors="pt").pixel_values
        input_tensor = pixel_values.squeeze()

        # input_ids
        processed_parse = random.choice(self.gt_token_sequences[idx])  # can be more than one, e.g., DocVQA Task 1
        input_ids = processor.tokenizer(
            processed_parse,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        if self.split == "train":
            labels = input_ids.clone()
            labels[
                labels == processor.tokenizer.pad_token_id
            ] = self.ignore_id  # model doesn't need to predict pad token
            labels[
                : torch.nonzero(labels == self.prompt_end_token_id).sum() + 1
            ] = self.ignore_id  # model doesn't need to predict prompt (for VQA)
            return input_tensor, input_ids, labels
        else:
            prompt_end_index = torch.nonzero(
                input_ids == self.prompt_end_token_id
            ).sum()  # return prompt end index instead of target output labels

            if self.task == 'docvqa':
                return input_tensor, input_ids, prompt_end_index, "\n".join(self.gt_token_sequences[idx])
            else:
                return input_tensor, input_ids, prompt_end_index, processed_parse


# In[31]:


dataset


# In[32]:


# we update some settings which differ from pretraining; namely the size of the images + no rotation required
# source: https://github.com/clovaai/donut/blob/master/config/train_cord.yaml
processor.feature_extractor.size = image_size[::-1] # should be (width, height)
processor.feature_extractor.do_align_long_axis = False

train_dataset = DonutDataset("./preprocessed_dataset_docvqa/", max_length=max_length,
                             split="train", task_start_token="<s_docvqa>", prompt_end_token="<s_answer>",
                             sort_json_key=False, # cord dataset is preprocessed, so no need for this
                             )

val_dataset = DonutDataset("./preprocessed_dataset_docvqa/", max_length=max_length,
                             split="validation", task_start_token="<s_docvqa>", prompt_end_token="<s_answer>",
                             sort_json_key=False, # cord dataset is preprocessed, so no need for this
                            task='docvqa',
                             )

# Limit dataset size
train_dataset = torch.utils.data.Subset(train_dataset, range(1000))  # First 1000 samples
val_dataset = torch.utils.data.Subset(val_dataset, range(200))  # First 200 samples


# In[33]:


pixel_values, decoder_input_ids, labels = train_dataset[0]


# In[34]:


pixel_values.shape


# In[35]:


print(labels)


# In[36]:


for decoder_input_id, label in zip(decoder_input_ids.tolist()[:-1], labels.tolist()[1:]):
  if label != -100:
    print(processor.decode([decoder_input_id]), processor.decode([label]))
  else:
    print(processor.decode([decoder_input_id]), label)


# In[37]:


pixel_values, decoder_input_ids, prompt_end_index, answer = val_dataset[0]


# In[38]:


pixel_values.shape


# In[39]:


prompt_end_index


# In[40]:


answer


# ## Create PyTorch DataLoaders
# 
# Next, we create corresponding PyTorch DataLoaders.

# In[41]:


from torch.utils.data import DataLoader

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)


# Let's verify a batch:

# In[42]:


batch = next(iter(train_dataloader))
pixel_values, decoder_input_ids, labels = batch
print(pixel_values.shape)


# In[43]:


decoder_input_ids.shape


# We can clearly see that we have set the labels of all prompt tokens (which includes the question) to -100, to make sure the model doesn't learn to generate them. We only start to have labels starting from the \<s_answer> decoder input token.

# In[44]:


for decoder_input_id, label in zip(decoder_input_ids[0].tolist()[:-1][:30], labels[0].tolist()[1:][:30]):
  if label != -100:
    print(processor.decode([decoder_input_id]), processor.decode([label]))
  else:
    print(processor.decode([decoder_input_id]), label)


# ## Define LightningModule
# 
# We'll fine-tune the model using [PyTorch Lightning](https://www.pytorchlightning.ai/) here, but note that you can of course also just fine-tune with regular PyTorch, HuggingFace [Accelerate](https://github.com/huggingface/accelerate), the HuggingFace [Trainer](https://huggingface.co/docs/transformers/main_classes/trainer), etc.
# 
# PyTorch Lightning is pretty convenient to handle things like device placement, mixed precision and logging for you.

# In[45]:


from pathlib import Path
import re
from nltk import edit_distance
import numpy as np
import math

from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only

from donut_distill.other import postprocess_donut_docvqa, postprocess_donut_funsd
from donut_distill.metrics import calculate_metrics_docvqa, calculate_metrics_funsd


import os

class DonutModelPLModule(pl.LightningModule):
    def __init__(self, config, processor, model):
        super().__init__()
        self.config = config
        self.processor = processor
        self.model = model
        self.best_anls = 0  # Track the best ANLS score

    def training_step(self, batch, batch_idx):
        pixel_values, decoder_input_ids, labels = batch
        
        outputs = self.model(pixel_values,
                             decoder_input_ids=decoder_input_ids[:, :-1],
                             labels=labels[:, 1:])
        loss = outputs.loss
        self.log_dict({"train_loss": loss}, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        pixel_values, decoder_input_ids, prompt_end_idxs, answers_lists = batch
        decoder_prompts = pad_sequence(
            [input_id[: end_idx + 1] for input_id, end_idx in zip(decoder_input_ids, prompt_end_idxs)],
            batch_first=True,
        )

        decoded_prompts = self.processor.tokenizer.batch_decode(decoder_prompts)[0]
        
        outputs = self.model.generate(pixel_values,
                                   decoder_input_ids=decoder_prompts,
                                   max_length=max_length,
                                   early_stopping=True,
                                   pad_token_id=self.processor.tokenizer.pad_token_id,
                                   eos_token_id=self.processor.tokenizer.eos_token_id,
                                   use_cache=True,
                                   num_beams=1,
                                   bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                                   return_dict_in_generate=True,)
    
        predictions = []
        for seq in self.processor.tokenizer.batch_decode(outputs.sequences):
            # seq = seq.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "")
            # seq = re.sub(r"<.*?>", "", seq, count=1).strip()  # remove first task start token
            predictions.append(seq)

        batch_metrics = {"anls": [], "exact_match": []}
        for pred, answers, prompt in zip(predictions, answers_lists, decoded_prompts):
            answer_list = answers.split("\n")
            answer_list = [postprocess_donut_docvqa(ans, processor) for ans in answer_list]
            pred = postprocess_donut_docvqa(pred, processor, verbose=config.verbose)
            metric = calculate_metrics_docvqa(answer_list, pred)
            batch_metrics["anls"].append(metric["anls"])
            batch_metrics["exact_match"].append(float(metric["exact_match"]))  # Convert bool to float for averaging

            if self.config.get("verbose", False):
                print(f"Prompt: {prompt}")
                print(f"\tPrediction: {pred}")
                print(f"\tAnswer: {answer}")
                print(metric)

        batch_metrics["anls"] = torch.tensor(batch_metrics["anls"], dtype=torch.float32)
        batch_metrics["exact_match"] = torch.tensor(batch_metrics["exact_match"], dtype=torch.float32)
        self.log("val_anls", batch_metrics["anls"].mean(), prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val_accuracy", batch_metrics["exact_match"].mean(), prog_bar=True, on_epoch=True, sync_dist=True)

        return batch_metrics

    def on_validation_epoch_end(self):
        avg_anls = self.trainer.callback_metrics["val_anls"].item()  # Get current ANLS score

        # Save model & processor if it's the best ANLS so far
        if avg_anls > self.best_anls:
            self.best_anls = avg_anls
            save_path = os.path.join(self.config["result_path"], "best_model")
            os.makedirs(save_path, exist_ok=True)

            # Save model state_dict
            torch.save(self.model.state_dict(), os.path.join(save_path, "best_model.pth"))
            # Save processor
            self.processor.save_pretrained(os.path.join(save_path, "processor"))

            print(f"Best model saved with ANLS: {avg_anls:.4f}")

    def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=self.config.get("lr"))
    # 
    #     return optimizer
        max_iter = None

        if int(self.config.get("max_epochs", -1)) > 0:
            assert len(self.config.train_batch_sizes) == 1, "Set max_epochs only if the number of datasets is 1"
            max_iter = (self.config.max_epochs * self.config.num_training_samples_per_epoch) / (
                self.config.train_batch_sizes[0] * torch.cuda.device_count() * self.config.get("num_nodes", 1)
            )

        if int(self.config.get("max_steps", -1)) > 0:
            max_iter = min(self.config.max_steps, max_iter) if max_iter is not None else self.config.max_steps

        assert max_iter is not None
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.get("lr", 3e-5))
        scheduler = {
            "scheduler": self.cosine_scheduler(optimizer, max_iter, self.config.warmup_steps),
            "name": "learning_rate",
            "interval": "step",
        }
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return train_dataloader

    def val_dataloader(self):
        return val_dataloader


# Next, we instantiate the module:

# In[46]:


config = {"max_epochs":30,
          "val_check_interval":0.2, # how many times we want to validate during an epoch
          "check_val_every_n_epoch":1,
          "gradient_clip_val":0.25,
          "num_training_samples_per_epoch": len(train_dataset),
          "lr":3e-5,
          "train_batch_sizes": [4],
          "val_batch_sizes": [4],
          # "seed":2022,
          "num_nodes": 1,
          "warmup_steps": 10000, # 800/8*30/10, 10%
          "result_path": "./result/docvqa",
          "verbose": True,
          }

model_module = DonutModelPLModule(config, processor, model)


# ## Train!

# In[27]:


from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

wandb_logger = WandbLogger(project="Donut-DocVQA")
# Define the checkpoint callback
checkpoint_callback = ModelCheckpoint(
    monitor="val_anls",  # Monitor the ANLS metric
    mode="max",  # Save the model with the highest ANLS
    save_top_k=1,  # Keep only the best model
    dirpath=config.get("result_path", "./result"),
    filename="best-checkpoint-{epoch:02d}-{val_anls:.4f}",
    verbose=True,
)
lr_callback = LearningRateMonitor(logging_interval="step")

trainer = pl.Trainer(
        limit_val_batches=0.2,
        accelerator="gpu",
        devices=1,
        max_epochs=config.get("max_epochs"),
        val_check_interval=config.get("val_check_interval"),
        check_val_every_n_epoch=config.get("check_val_every_n_epoch"),
        gradient_clip_val=config.get("gradient_clip_val"),
        precision=16, # we'll use mixed precision
        num_sanity_val_steps=0,
        logger=wandb_logger,
        callbacks=[lr_callback, checkpoint_callback],
)

trainer.fit(model_module)


# # Push to hub and reuse
# 
# HuggingFace's [hub](https://huggingface.co/) is a nice place to host, version and share machine learning models (and datasets, and demos in the form of [Spaces](https://huggingface.co/spaces)).
# 
# We first provide our authentication token.

# In[47]:




# Pushing to the hub after training is as easy as:

# In[49]:


# repo_name = "nielsr/donut-docvqa-demo"

# here we push the processor and model to the hub
# note that you can add `private=True` in case you're using the private hub
# which makes sure the model is only shared with your colleagues
# model_module.processor.push_to_hub(repo_name)
# model_module.model.push_to_hub(repo_name)


# Reloading can then be done as:

# In[50]:


# from transformers import DonutProcessor, VisionEncoderDecoderModel
#
# processor = DonutProcessor.from_pretrained("nielsr/donut-docvqa-demo")
# model = VisionEncoderDecoderModel.from_pretrained("nielsr/donut-docvqa-demo")


# ## Inference
# 
# For inference, we refer to the [docs](https://huggingface.co/docs/transformers/main/en/model_doc/donut#inference) of Donut, or the corresponding [notebook](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Donut/DocVQA/Quick_inference_with_DONUT_for_DocVQA.ipynb).

# In[ ]:




