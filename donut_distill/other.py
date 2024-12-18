from typing import Dict, List, Optional
from transformers import (
    DonutProcessor,
    VisionEncoderDecoderModel,
)
import torch
from PIL import Image
import re


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
    decoded_output = postprocess_donut_funsd(decoded_output)
    return decoded_output


def postprocess_donut_funsd(predictions: List[dict]):
    result = []
    if not isinstance(predictions, list):
        return result

    for prediction in predictions:
        if ("text" not in prediction or "label" not in prediction
            or not isinstance(prediction["text"], str) or not isinstance(prediction["label"], str)
            or not prediction["text"] or not prediction["label"]):
            continue

        result.append({
            "text": prediction["text"],
            "label": prediction["label"]
        })

    return result

