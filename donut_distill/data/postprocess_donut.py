from typing import Dict, List, Optional
from transformers import (
    DonutProcessor,
)

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
        print("Unprocessed", outputs)
        print("Json", outputs_json)


    return outputs_json.get("answer", "").lower()



