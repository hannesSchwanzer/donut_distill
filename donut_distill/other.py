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


def postprocess_donut_funsd(outputs: str | dict | list, processor: DonutProcessor, verbose: bool=False) -> List[dict]:
    result = []

    if isinstance(outputs, str):
        outputs = outputs.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
        # outputs = re.sub(r"<.*?>", "", outputs, count=1).strip()  # remove first task start token
        outputs = processor.token2json(outputs)
        if verbose:
            print(outputs)

    if isinstance(outputs, dict):  # Check if it's a dictionary
        for key, value in outputs.items():
            if isinstance(value, (dict, list)):
                result.extend(postprocess_donut_funsd(value, processor))

        if ("text" in outputs and "label" in outputs
            and isinstance(outputs["text"], str) and isinstance(outputs["label"], str)
            and outputs["text"] and outputs["label"]):
            result.append({
                "text": outputs["text"].strip(),
                "label": outputs["label"].strip()
            })
    elif isinstance(outputs, list):  # Check if it's a list
        for output in outputs:
            result.extend(postprocess_donut_funsd(output, processor))

    return result


def postprocess_donut_docvqa(outputs: str, processor: DonutProcessor, verbose: bool=False) -> str:

    outputs = outputs.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    # outputs = re.sub(r"<.*?>", "", outputs, count=1).strip()  # remove first task start token
    outputs_json: Dict[str, str] = processor.token2json(outputs)
    if verbose:
        print("Json", outputs_json)


    return outputs_json.get("answer", "").lower()



