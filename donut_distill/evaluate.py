from typing import List, Optional, Tuple
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    DonutProcessor,
    VisionEncoderDecoderModel,
)
from torch.nn.utils.rnn import pad_sequence
import config as CONFIG
from donut_distill.donut_dataset import DonutDataset
from donut_distill.metrics import calculate_metrics
from transformers import GenerationConfig
from donut_distill.other import postprocess_donut_funsd
import numpy as np

def evaluate(
    model: VisionEncoderDecoderModel,
    processor: DonutProcessor,
    device: torch.device,
    val_dataloader: DataLoader,
    generation_config: Optional[GenerationConfig] = None
    ):

    if generation_config == None:
        # Default generation config TODO:
        generation_config = GenerationConfig(early_stopping=True, num_beams=1)

    val_metrics = {
        "f1_score": [],
        "recall": [],
        "precision": []
    }

    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validate"):
            pixel_values, decoder_input_ids, prompt_end_idxs, answers = batch
            pixel_values = pixel_values.to(device)

            decoder_prompts = pad_sequence(
                [
                    input_id[: end_idx + 1]
                    for input_id, end_idx in zip(
                        decoder_input_ids, prompt_end_idxs
                    )
                ],
                batch_first=True,
            ).to(device)

            outputs = model.generate(
                pixel_values,
                decoder_input_ids=decoder_prompts,
                max_length=CONFIG.MAX_LENGTH,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                use_cache=True,
                bad_words_ids=[[processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
                generation_config=generation_config
            )

            predictions = []
            for seq in processor.tokenizer.batch_decode(outputs.sequences):
                postprocess_donut_funsd(seq, processor)
                predictions.append(seq)

            scores = []
            for pred, answer in zip(predictions, answers):
                answer = postprocess_donut_funsd(answer, processor)

                f1_score, recall, precision = calculate_metrics(answer, pred)
                val_metrics["f1_score"].append(f1_score)
                val_metrics["recall"].append(recall)
                val_metrics["precision"].append(precision)

                if CONFIG.VERBOSE and len(scores) == 1:
                    print("\n----------------------------------------\n")
                    print(f"\nPrediction: {pred}")
                    print(f"\n\tAnswer: {answer}")
                    print(f"\n\tF1-Score: {f1_score}")

    return np.mean(val_metrics["f1_score"]), np.mean(val_metrics["recall"]), np.mean(val_metrics["precision"])


def test_generation_configs(model, processor, device, generationsconfigs: List[Tuple[str, GenerationConfig]]):
    val_dataset = DonutDataset(
        dataset_name_or_path="preprocessed_dataset",
        processor=processor,
        model=model,
        max_length=CONFIG.MAX_LENGTH,
        split="test",
        task_start_token="<s_funsd>",
        sort_json_key=False,  # cord dataset is preprocessed, so no need for this
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=CONFIG.VAL_BATCH_SIZES,
        shuffle=False,
        num_workers=CONFIG.NUM_WORKERS,
    )

    for description, generation_config in generationsconfigs:
        f1_score, recall, precision = evaluate(
            model=model,
            processor=processor,
            device = device,
            val_dataloader=val_dataloader,
            generation_config=generation_config,
        )

        print(100*'-')
        print(description)
        print("\tF1-score:", f1_score)


if __name__ == "__main__":
    generation_configs = [
        ("Top k - 50", GenerationConfig(
            do_sample=True,
            top_k=50
        )),
        ("Top k - 35", GenerationConfig(
            do_sample=True,
            top_k=35
        )),
        ("Top k - 20", GenerationConfig(
            do_sample=True,
            top_k=20
        )),
        ("Nucleus, p=0.9", GenerationConfig(
            do_sample=True,
            top_p=0.9,
            top_k=0
        )),
        ("Nucleus, p=0.95", GenerationConfig(
            do_sample=True,
            top_p=0.95,
            top_k=0
        )),
        ("Nucleus, p=0.92", GenerationConfig(
            do_sample=True,
            top_p=0.92,
            top_k=0
        )),
        ("Nucleus, p=0.94", GenerationConfig(
            do_sample=True,
            top_p=0.94,
            top_k=0
        )),
        ("Nucleus K, p=0.95 k=50", GenerationConfig(
            do_sample=True,
            top_k=50,
            top_p=0.95,
        )),
    ]
    import os

    donut_path = "result/donut_20241208_143816"
    model_path = os.path.join(donut_path, "model")
    processor_path = os.path.join(donut_path, "processor")
    processor = DonutProcessor.from_pretrained(processor_path)
    model = VisionEncoderDecoderModel.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    test_generation_configs(model, processor, device, generation_configs)
