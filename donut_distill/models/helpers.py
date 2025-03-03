from transformers import (
    DonutProcessor,
    VisionEncoderDecoderModel,
    VisionEncoderDecoderConfig,
)
import donut_distill.config.config as CONFIG
from typing import List, Optional, Tuple
from typing import Dict, List, Optional
import torch
from PIL import Image
import re

def prepare_model_and_processor(
    special_tokens: Optional[List[str]] = None,
    return_config: bool = False,
    load_teacher: bool = False,
) -> Tuple[VisionEncoderDecoderModel, DonutProcessor] | Tuple[VisionEncoderDecoderModel, DonutProcessor, VisionEncoderDecoderConfig]:
    """
    Loads and configures the Donut model and processor.

    Args:
        special_tokens (Optional[List[str]]): Additional special tokens to add to the tokenizer.
        return_config (bool): Whether to return the model configuration along with the model and processor.
        load_teacher (bool): If True, loads the teacher model instead of the default model.

    Returns:
        Tuple: A tuple containing:
            - model (VisionEncoderDecoderModel): The Donut model.
            - processor (DonutProcessor): The processor for tokenizing and image processing.
            - donut_config (VisionEncoderDecoderConfig), optional: Returned if `return_config` is True.
    """

    # Determine which model to load (teacher or student)
    model_dir = CONFIG.TEACHER_MODEL_PATH if load_teacher else CONFIG.MODEL_ID

    # Load model configuration and update relevant parameters
    donut_config: VisionEncoderDecoderConfig = VisionEncoderDecoderConfig.from_pretrained(model_dir)
    donut_config.encoder.image_size = CONFIG.INPUT_SIZE
    donut_config.decoder.max_length = CONFIG.MAX_LENGTH

    # Load processor and model
    processor: DonutProcessor = DonutProcessor.from_pretrained(model_dir)
    model: VisionEncoderDecoderModel = VisionEncoderDecoderModel.from_pretrained(
        model_dir, config=donut_config
    )

    # Add special tokens if provided
    if special_tokens:
        add_tokens(model, processor, special_tokens)
        donut_config.decoder.vocab_size = len(processor.tokenizer)  # Update vocabulary size

    # Update image processing settings
    processor.image_processor.size = CONFIG.INPUT_SIZE[::-1]  # Reverse (height, width) format
    processor.image_processor.do_align_long_axis = False

    # Return model, processor, and optionally the configuration
    return (model, processor, donut_config) if return_config else (model, processor)


# https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Donut/DocVQA/Fine_tune_Donut_on_DocVQA.ipynb
def add_tokens(model: VisionEncoderDecoderModel, processor: DonutProcessor, list_of_tokens: List[str]):
    """
    Add new tokens to the tokenizer and resize token embeddings.

    Args:
        model (VisionEncoderDecoderModel): The Donut model.
        processor (DonutProcessor): The processor for handling input/output data.
        list_of_tokens (List[str]): List of special tokens to add.
    """
    processor.tokenizer.add_tokens(list_of_tokens)
    model.decoder.resize_token_embeddings(len(processor.tokenizer))
    model.config.vocab_size = len(processor.tokenizer)


# TODO: Update function
def inference(
    model: VisionEncoderDecoderModel,
    processor: DonutProcessor,
    device: torch.device | str,
    pixel_values: Optional[torch.Tensor] = None,
    image: Optional[Image.Image] = None,
    task_prompt_ids: Optional[torch.Tensor] = None,
    task_prompt: str = "",
):
    if pixel_values is None:
        assert Image is not None, "pixel_values or image has to be set"
        pixel_values = processor(image.convert("RGB"), return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    if task_prompt_ids is None:
        task_prompt_ids = processor.tokenizer(
            task_prompt, add_special_tokens=False, return_tensors="pt"
        )["input_ids"]
    task_prompt_ids = task_prompt_ids.to(device)

    outputs = model.generate(
        pixel_values.to(device),
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=model.decoder.config.max_position_embeddings,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=1,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
        output_scores=True,
    )

    decoded_output = processor.batch_decode(outputs.sequences)[0]
    decoded_output = decoded_output.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    decoded_output = re.sub(r"<.*?>", "", seq, count=1).strip()  # remove first task start token
    decoded_output = processor.token2json(decoded_output)
    # decoded_output = postprocess_donut_funsd(decoded_output)
    return decoded_output


