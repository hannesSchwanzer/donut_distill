from typing import Dict, List, Union
from transformers import (
    DonutProcessor,
)


def postprocess_donut_funsd(
    outputs: Union[str, dict, list], processor: DonutProcessor, verbose: bool = False
) -> List[dict]:
    """
    Postprocess the output of the Donut model for FUNSD (Form Understanding in Noisy Scanned Documents).

    This function:
    - Removes special tokens from string outputs and converts them into JSON format.
    - Recursively processes dictionaries and lists to extract relevant text-label pairs.
    - Returns a structured list of extracted entities.

    Args:
        outputs (str | dict | list): Model output, which can be a string, dictionary, or list.
        processor (DonutProcessor): Tokenizer processor used to decode model outputs.
        verbose (bool, optional): If True, prints intermediate JSON outputs. Default is False.

    Returns:
        List[dict]: A list of extracted text-label pairs in the format:
            [{"text": "example text", "label": "example label"}, ...]
    """
    result = []

    if isinstance(outputs, str):
        # Remove special tokens (EOS and PAD)
        outputs = outputs.replace(processor.tokenizer.eos_token, "").replace(
            processor.tokenizer.pad_token, ""
        )
        # Convert tokenized output into JSON format
        outputs = processor.token2json(outputs)

        if verbose:
            print(outputs)

    if isinstance(outputs, dict):  # If the output is a dictionary, process recursively
        for key, value in outputs.items():
            if isinstance(value, (dict, list)):
                result.extend(postprocess_donut_funsd(value, processor))

        # Extract relevant text-label pairs
        if (
            "text" in outputs
            and "label" in outputs
            and isinstance(outputs["text"], str)
            and isinstance(outputs["label"], str)
            and outputs["text"]
            and outputs["label"]
        ):
            result.append(
                {"text": outputs["text"].strip(), "label": outputs["label"].strip()}
            )

    elif isinstance(outputs, list):  # If the output is a list, process each item
        for output in outputs:
            result.extend(postprocess_donut_funsd(output, processor))

    return result


def postprocess_donut_docvqa(
    outputs: str, processor: DonutProcessor, verbose: bool = False
) -> str:
    """
    Postprocess the output of the Donut model for the DocVQA (Document Visual Question Answering) task.

    This function:
    - Removes special tokens (EOS and PAD) from the model output.
    - Converts the processed string into a JSON format using the processor.
    - Extracts and returns the answer field in lowercase.

    Args:
        outputs (str): The raw model output as a tokenized string.
        processor (DonutProcessor): Tokenizer processor used for decoding.
        verbose (bool, optional): If True, prints intermediate processing steps. Default is False.

    Returns:
        str: The extracted answer in lowercase, or an empty string if not found.
    """
    # Remove special tokens (EOS and PAD)
    outputs = outputs.replace(processor.tokenizer.eos_token, "").replace(
        processor.tokenizer.pad_token, ""
    )

    # Convert tokenized output to JSON format
    outputs_json: Dict[str, str] = processor.token2json(outputs)
    
    if verbose:
        print("Unprocessed:", outputs)
        print("Json:", outputs_json)

    # Extract and return the answer, converting it to lowercase
    return outputs_json.get("answer", "").lower()


