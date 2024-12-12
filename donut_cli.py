from transformers import (
    DonutProcessor,
    VisionEncoderDecoderModel,
)
import os
from PIL import Image
import torch
import re
import json

if __name__ == "__main__":
    donut_path = "result/donut_20241208_143816"
    model_path = os.path.join(donut_path, "model")
    processor_path = os.path.join(donut_path, "processor")
    # if not os.path.isdir(model_path):
    #     print("Path is not a directory")
    #     exit(1)

    # donut_config = VisionEncoderDecoderConfig.from_pretrained(model_path)
    processor = DonutProcessor.from_pretrained(processor_path)
    model = VisionEncoderDecoderModel.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    task_prompt = "<s_funsd>"

    while True:
        print("Enter path to document image:")
        document_path = input()
        try:
            image = Image.open(document_path)
        except:
            print("Couldn't open image")
            continue

        rgb_image = Image.merge("RGB", (image, image, image))
        pixel_values = processor(rgb_image, return_tensors="pt").pixel_values
        print(pixel_values.shape)
        decoder_input_ids = processor.tokenizer(
            task_prompt, add_special_tokens=False, return_tensors="pt"
        )["input_ids"]
        print(decoder_input_ids.shape)
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
        output = {"predictions": list()}
        for seq in processor.batch_decode(outputs.sequences):
            seq = seq.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
            seq = re.sub(r"<.*?>", "", seq, count=1).strip()  # remove first task start token
            output["predictions"].append(processor.token2json(seq))

        print(output)
