import json
import random
from typing import Any, List
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import DonutProcessor, VisionEncoderDecoderModel

# https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Donut/CORD/Fine_tune_Donut_on_a_custom_dataset_(CORD)_with_PyTorch_Lightning.ipynb

added_tokens = []


class DonutDataset(Dataset):
    """
    PyTorch Dataset for Donut. This class takes a HuggingFace Dataset as input.

    Each row consists of an image path (png/jpg/jpeg) and ground truth (JSON/JSONL/TXT).
    It is converted into pixel_values (vectorized image) and labels (tokenized input_ids).

    Modified  from original:
     - https://github.com/clovaai/donut/blob/4cfcf972560e1a0f26eb3e294c8fc88a0d336626/donut/util.py#L31
     - https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Donut/DocVQA/Fine_tune_Donut_on_DocVQA.ipynb

    Args:
        processor (DonutProcessor): Processor to handle images and tokenization.
        model (VisionEncoderDecoderModel): Model whose tokenizer may be extended.
        dataset_name_or_path (str): Dataset name (on HuggingFace) or local path containing images and metadata.jsonl.
        max_length (int): Maximum number of tokens for target sequences.
        split (str): Dataset split - "train", "validation", or "test".
        ignore_id (int): Token ID to be ignored in loss computation (default: -100).
        task_start_token (str): Special token indicating the start of the task.
        prompt_end_token (str, optional): Token marking the end of the prompt (defaults to task_start_token).
        sort_json_key (bool): Whether to sort JSON keys before tokenization.
        task (str): Task name (used for specific behavior like DocVQA).
    """

    def __init__(
        self,
        processor: DonutProcessor,
        model: VisionEncoderDecoderModel,
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

        self.processor = processor
        self.model = model
        self.max_length = max_length
        self.split = split
        self.ignore_id = ignore_id
        self.task_start_token = task_start_token
        self.prompt_end_token = (
            prompt_end_token if prompt_end_token else task_start_token
        )
        self.sort_json_key = sort_json_key
        self.task = task

        # Load dataset from HuggingFace or local path
        self.dataset = load_dataset(dataset_name_or_path, split=self.split)
        self.dataset_length = len(self.dataset)

        # Process ground truth token sequences
        self.gt_token_sequences = []
        for sample in self.dataset:
            ground_truth = json.loads(sample["ground_truth"])
            if (
                "gt_parses" in ground_truth
            ):  # Multiple ground truth examples (e.g., DocVQA)
                assert isinstance(ground_truth["gt_parses"], list)
                gt_jsons = ground_truth["gt_parses"]
            else:
                assert "gt_parse" in ground_truth and isinstance(
                    ground_truth["gt_parse"], dict
                )
                gt_jsons = [ground_truth["gt_parse"]]

            self.gt_token_sequences.append(
                [
                    self.json2token(
                        gt_json,
                        update_special_tokens_for_json_key=self.split == "train",
                        sort_json_key=self.sort_json_key,
                    )
                    + self.processor.tokenizer.eos_token
                    for gt_json in gt_jsons  # Convert each JSON object into tokens
                ]
            )

        # Add special tokens for start/prompt end
        self.add_tokens([self.task_start_token, self.prompt_end_token])
        self.prompt_end_token_id = self.processor.tokenizer.convert_tokens_to_ids(
            self.prompt_end_token
        )

    def json2token(
        self,
        obj: Any,
        update_special_tokens_for_json_key: bool = True,
        sort_json_key: bool = True,
    ):
        """
        Convert a JSON object into a token sequence.

        Handles different data structures:
        - Dict: Recursively converts keys/values into a structured token format.
        - List: Joins elements with a separator token.
        - Other types: Converts to string and applies special token formatting.
        """
        if type(obj) is dict:
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
                        self.add_tokens([rf"<s_{k}>", rf"</s_{k}>"])
                    output += (
                        rf"<s_{k}>"
                        + self.json2token(
                            obj[k], update_special_tokens_for_json_key, sort_json_key
                        )
                        + rf"</s_{k}>"
                    )
                return output
        elif type(obj) is list:
            return r"<sep/>".join(
                [
                    self.json2token(
                        item, update_special_tokens_for_json_key, sort_json_key
                    )
                    for item in obj
                ]
            )
        else:
            obj = str(obj)
            if f"<{obj}/>" in added_tokens:
                obj = f"<{obj}/>"  # for categorical special tokens
            return obj

    def add_tokens(self, list_of_tokens: List[str]):
        """
        Add special tokens to the tokenizer and resize the model's embedding layer accordingly.
        """
        newly_added_num = self.processor.tokenizer.add_tokens(list_of_tokens)
        if newly_added_num > 0:
            self.model.decoder.resize_token_embeddings(len(self.processor.tokenizer))
            self.model.config.vocab_size = len(self.processor.tokenizer)
            added_tokens.extend(list_of_tokens)

    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset.
        """
        return self.dataset_length

    def __getitem__(self, idx: int):
        """
        Retrieve and preprocess a dataset sample.

        This function:
        - Loads an image and converts it into a tensor.
        - Selects a target sequence and tokenizes it.
        - Masks certain labels during training.
        - Handles variations based on the task.

        Returns:
            If training:
                - pixel_values (Tensor): Preprocessed image.
                - input_ids (Tensor): Tokenized ground truth sequence.
                - labels (Tensor): Masked labels for loss computation.
            Otherwise:
                - pixel_values (Tensor): Preprocessed image.
                - input_ids (Tensor): Tokenized ground truth sequence.
                - prompt_end_index (int): Index marking the end of the prompt.
                - target_sequence (str) or full ground truth sequence (str) for DocVQA.
        """
        sample = self.dataset[idx]

        # Process image input
        image = sample["image"].convert("RGB")
        pixel_values = self.processor(
            image, random_padding=self.split == "train", return_tensors="pt"
        ).pixel_values.squeeze()

        # Select a ground truth token sequence (can be multiple for DocVQA)
        target_sequence = random.choice(self.gt_token_sequences[idx])

        # Tokenize the target sequence
        input_ids = self.processor.tokenizer(
            target_sequence,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        if self.split == "train":
            # Mask labels: ignore padding and prompt tokens
            labels = input_ids.clone()
            labels[labels == self.processor.tokenizer.pad_token_id] = (
                self.ignore_id
            )
            labels[: torch.nonzero(labels == self.prompt_end_token_id).sum() + 1] = (
                self.ignore_id
            )
            return pixel_values, input_ids, labels
        else:
            # Return prompt_end_index instead of masked labels for inference
            prompt_end_index = torch.nonzero(
                input_ids == self.prompt_end_token_id
            ).sum()
            if self.task == 'docvqa':
                return pixel_values, input_ids, prompt_end_index, "\n".join(self.gt_token_sequences[idx])
            else:
                return pixel_values, input_ids, prompt_end_index, target_sequence
