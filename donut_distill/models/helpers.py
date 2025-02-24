from transformers import (
    DonutProcessor,
    VisionEncoderDecoderModel,
    VisionEncoderDecoderConfig,
)
import donut_distill.config.config as CONFIG
from typing import List, Optional, Tuple
from typing import Dict, List, Optional
from transformers import (
    DonutProcessor,
    VisionEncoderDecoderModel,
)
import torch
from PIL import Image
import re

def prepare_model_and_processor(
    special_tokens: Optional[List[str]] = None,
    return_config: bool = False,
    load_teacher: bool = False,
) -> Tuple[VisionEncoderDecoderModel, DonutProcessor] | Tuple[VisionEncoderDecoderModel, DonutProcessor, VisionEncoderDecoderConfig]:

    if load_teacher:
        model_dir = CONFIG.TEACHER_MODEL_PATH
    else:
        model_dir = CONFIG.MODEL_ID
    donut_config: VisionEncoderDecoderConfig = VisionEncoderDecoderConfig.from_pretrained(model_dir)
    donut_config.encoder.image_size = CONFIG.INPUT_SIZE
    donut_config.decoder.max_length = CONFIG.MAX_LENGTH

    processor: DonutProcessor = DonutProcessor.from_pretrained(model_dir)
    model: VisionEncoderDecoderModel = VisionEncoderDecoderModel.from_pretrained(
        model_dir, config=donut_config
    )

    if special_tokens:
        add_tokens(model, processor, special_tokens)

    processor.image_processor.size = CONFIG.INPUT_SIZE[::-1]
    processor.image_processor.do_align_long_axis = False

    if return_config:
        return model, processor, donut_config
    else:
        return model, processor


# https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Donut/DocVQA/Fine_tune_Donut_on_DocVQA.ipynb
def add_tokens(model: VisionEncoderDecoderModel, processor: DonutProcessor, list_of_tokens: List[str]):
    """
    Add new tokens to the tokenizer and resize token embeddings.

    Args:
        model (VisionEncoderDecoderModel): The Donut model.
        processor (DonutProcessor): The processor for handling input/output data.
        list_of_tokens (List[str]): List of special tokens to add.
    """
    newly_added_num = processor.tokenizer.add_tokens(list_of_tokens)
    if newly_added_num > 0:
        model.decoder.resize_token_embeddings(len(processor.tokenizer))


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


