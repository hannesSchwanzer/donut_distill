from typing import List, Optional, Tuple
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    DonutProcessor,
    VisionEncoderDecoderModel,
)
from torch.nn.utils.rnn import pad_sequence
import donut_distill.config as CONFIG
from donut_distill.donut_dataset import DonutDataset
from donut_distill.metrics import calculate_metrics_docvqa, calculate_metrics_funsd
from transformers import GenerationConfig
from donut_distill.other import postprocess_donut_docvqa, postprocess_donut_funsd
import numpy as np


def evaluate_docvqa(
    model: VisionEncoderDecoderModel,
    processor: DonutProcessor,
    device: torch.device,
    val_dataloader: DataLoader,
    generation_config: Optional[GenerationConfig],
):
    val_metrics = {"exact_match": [], "anls": []}

    model.eval()
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validate"):
            pixel_values, decoder_input_ids_list, prompt_end_idxs_list, answers_list = batch
            pixel_values = pixel_values.to(device)  # (batch_size, channels, height, width)

            batch_size = pixel_values.shape[0]

            # **Decoder prompts remain the same across all answers per item**
            decoder_prompts = pad_sequence(
                [
                    input_ids[0][: prompt_end_idx[0] + 1]
                    for input_ids, prompt_end_idx in zip(decoder_input_ids_list, prompt_end_idxs_list)
                ],
                batch_first=True,
            ).to(device)  # Shape: (batch_size, seq_length)

            # **Generate all predictions at once**
            outputs = model.generate(
                pixel_values,
                decoder_input_ids=decoder_prompts,
                max_length=CONFIG.MAX_LENGTH,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                use_cache=True,
                bad_words_ids=[[processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
                generation_config=generation_config,
            )

            # **Decode predictions**
            predictions = processor.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)

            # **Compare each prediction with all possible answers**
            for pred, answers in zip(predictions, answers_list):
                if CONFIG.VERBOSE:
                    print("\n----------------------------------------")
                    print("Answers unverarbeitet:", answers)
                    print("Prediction unverarbeitet:", pred)

                pred = postprocess_donut_docvqa(pred, processor, verbose=CONFIG.VERBOSE)
                answers = [postprocess_donut_docvqa(ans, processor) for ans in answers]

                # **Pass all answers at once to calculate_metrics**
                metric = calculate_metrics_docvqa(answers, pred)  # Now takes a list of answers

                # **Store metrics**
                val_metrics["anls"].append(metric["anls"])
                val_metrics["exact_match"].append(metric["exact_match"])

                if CONFIG.VERBOSE:
                    print(f"\tPrediction: {pred}")
                    print(f"\tAnswers: {answers}")
                    print(f"\texact_match: {metric['exact_match']}")
                    print(f"\tanls: {metric['anls']}")

    return {
        "eval/accuracy": np.mean(val_metrics["exact_match"]),
        "eval/anls": np.mean(val_metrics["anls"]),
    }


def evaluate_funsd(
    model: VisionEncoderDecoderModel,
    processor: DonutProcessor,
    device: torch.device,
    val_dataloader: DataLoader,
    generation_config: Optional[GenerationConfig] = None,
):
    if generation_config == None:
        # Default generation config TODO:
        generation_config = GenerationConfig(early_stopping=True, num_beams=1)

    val_metrics = {"f1": [], "recall": [], "precision": []}
    # num_samples = len(val_dataloader.dataset) // 3

    model.eval()
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validate"):
            # for batch in tqdm(itertools.islice(val_dataloader, num_samples // val_dataloader.batch_size), desc="Validate"): #TODO: REMOVE
            pixel_values, decoder_input_ids, prompt_end_idxs, answers = batch
            pixel_values = pixel_values.to(device)

            decoder_prompts = pad_sequence(
                [
                    input_id[: end_idx + 1]
                    for input_id, end_idx in zip(decoder_input_ids, prompt_end_idxs)
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
                generation_config=generation_config,
            )

            predictions = processor.tokenizer.batch_decode(outputs.sequences)

            for pred, answer in zip(predictions, answers):
                answer = postprocess_donut_funsd(answer, processor)
                if CONFIG.VERBOSE:
                    print("\n----------------------------------------\n")
                    print("Prediction unverarbeitet:")
                pred = postprocess_donut_funsd(pred, processor, verbose=CONFIG.VERBOSE)

                f1_score, recall, precision = calculate_metrics_funsd(answer, pred)
                val_metrics["f1"].append(f1_score)
                val_metrics["recall"].append(recall)
                val_metrics["precision"].append(precision)

                if CONFIG.VERBOSE:
                    print(f"\nPrediction: {pred}")
                    print(f"\n\tAnswer: {answer}")
                    print(f"\n\tF1-Score: {f1_score}")
                    print(f"\n\tRecall: {recall}")
                    print(f"\n\tPrecsion: {precision}")

    val_metrics["f1"] = np.mean(val_metrics["f1"])
    val_metrics["recall"] = np.mean(val_metrics["recall"])
    val_metrics["precision"] = np.mean(val_metrics["precision"])

    return val_metrics


def evaluate_step_funsd(batch, batch_idx, processor, model, generation_config):
    pixel_values, decoder_input_ids, prompt_end_idxs, answers = batch

    decoder_prompts = pad_sequence(
        [
            input_id[: end_idx + 1]
            for input_id, end_idx in zip(decoder_input_ids, prompt_end_idxs)
        ],
    )

    outputs = model.generate(
        pixel_values,
        decoder_input_ids=decoder_prompts,
        max_length=CONFIG.MAX_LENGTH,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
        generation_config=generation_config,
    )

    predictions = processor.tokenizer.batch_decode(outputs.sequences)

    f1_scores = []
    for pred, answer in zip(predictions, answers):
        answer = postprocess_donut_funsd(answer, processor)
        pred = postprocess_donut_funsd(pred, processor)

        f1_score, recall, precision = calculate_metrics_funsd(answer, pred)
        f1_scores.append(f1_score)
        if CONFIG.VERBOSE:
            print("\n----------------------------------------\n")
            print(f"\nPrediction: {pred}")
            print(f"\n\tAnswer: {answer}")
            print(f"\n\tF1-Score: {f1_score}")

    return f1_scores


def evaluate_generation_configs_funsd(
    model,
    processor,
    device,
    val_dataloader,
    generationsconfigs: List[Tuple[str, GenerationConfig]],
):
    results = list()
    for description, generation_config in generationsconfigs:
        if CONFIG.VERBOSE:
            print(
                f"------------------------{description}--------------------------------"
            )

        result = evaluate_funsd(
            model=model,
            processor=processor,
            device=device,
            val_dataloader=val_dataloader,
            generation_config=generation_config,
        )

        results.append(
            {
                f"f1/{description}": result["f1"],
                f"recall/{description}": result["precision"],
                f"recall/{description}": result["recall"],
            }
        )

        if CONFIG.VERBOSE:
            print(100 * "-")
            print(description)
            print("\tF1-score:", result["f1"])
            print("\tRecall:", result["recall"])
            print("\tPrecision:", result["precision"])

    return results


def evaluate_generation_configs_docvqa(
    model,
    processor,
    device,
    val_dataloader,
    generationsconfigs: List[Tuple[str, GenerationConfig]],
):
    results = list()
    for description, generation_config in generationsconfigs:
        if CONFIG.VERBOSE:
            print(
                f"------------------------{description}--------------------------------"
            )

        result = evaluate_docvqa(
            model=model,
            processor=processor,
            device=device,
            val_dataloader=val_dataloader,
            generation_config=generation_config,
        )

        results.append(
            {
                f"accuracy/{description}": result["accuracy"],
                f"avg_normed_edit_distance/{description}": result["avg_normed_edit_distance"],
            }
        )

        if CONFIG.VERBOSE:
            print(100 * "-")
            print(description)
            print(f"accuracy", result["accuracy"])
            print("avg_normed_edit_distance", result["avg_normed_edit_distance"])

    return results


if __name__ == "__main__":
    generation_configs = [
        ("Top k - 50", GenerationConfig(do_sample=True, top_k=50)),
        ("Top k - 35", GenerationConfig(do_sample=True, top_k=35)),
        ("Top k - 20", GenerationConfig(do_sample=True, top_k=20)),
        ("Nucleus, p=0.9", GenerationConfig(do_sample=True, top_p=0.9, top_k=0)),
        ("Nucleus, p=0.95", GenerationConfig(do_sample=True, top_p=0.95, top_k=0)),
        ("Nucleus, p=0.92", GenerationConfig(do_sample=True, top_p=0.92, top_k=0)),
        ("Nucleus, p=0.94", GenerationConfig(do_sample=True, top_p=0.94, top_k=0)),
        (
            "Nucleus K, p=0.95 k=50",
            GenerationConfig(
                do_sample=True,
                top_k=50,
                top_p=0.95,
            ),
        ),
        (
            "Contrastive search, alpha=0.6, k=4",
            GenerationConfig(
                penalty_alpha=0.6,
                top_k=4,
            ),
        ),
        (
            "Contrastive search, alpha=0.8, k=4",
            GenerationConfig(
                penalty_alpha=0.8,
                top_k=4,
            ),
        ),
        (
            "Contrastive search, alpha=0.6, k=8",
            GenerationConfig(
                penalty_alpha=0.6,
                top_k=8,
            ),
        ),
        (
            "Contrastive search, alpha=0.6, k=10",
            GenerationConfig(
                penalty_alpha=0.6,
                top_k=10,
            ),
        ),
        (
            "Contrastive search, alpha=0.6, k=4",
            GenerationConfig(
                penalty_alpha=0.7,
                top_k=4,
            ),
        ),
        (
            "Nucleus K, p=0.95 k=40",
            GenerationConfig(
                do_sample=True,
                top_k=40,
                top_p=0.95,
            ),
        ),
        (
            "Nucleus K, p=0.94 k=50",
            GenerationConfig(
                do_sample=True,
                top_k=50,
                top_p=0.94,
            ),
        ),
        (
            "Nucleus K, p=0.93 k=40",
            GenerationConfig(
                do_sample=True,
                top_k=40,
                top_p=0.93,
            ),
        ),
        (
            "Nucleus K, p=0.92 k=30",
            GenerationConfig(
                do_sample=True,
                top_k=30,
                top_p=0.92,
            ),
        ),
        ("Greedy", GenerationConfig()),
        ("Beam, num=5", GenerationConfig(num_beams=5, early_stopping=True)),
        ("Beam, num=3", GenerationConfig(num_beams=3, early_stopping=True)),
        ("Beam, num=7", GenerationConfig(num_beams=7, early_stopping=True)),
        (
            "Beam ngrams, num=5 ngrams=2",
            GenerationConfig(
                num_beams=5,
                no_repeat_ngram_size=2,
                early_stopping=True,
            ),
        ),
        (
            "Beam ngrams, num=5 ngrams=4",
            GenerationConfig(
                num_beams=5,
                no_repeat_ngram_size=4,
                early_stopping=True,
            ),
        ),
        (
            "Beam ngrams, num=5 ngrams=8",
            GenerationConfig(
                num_beams=5,
                no_repeat_ngram_size=8,
                early_stopping=True,
            ),
        ),
    ]
    import os

    donut_path = "result/donut_149"
    model_path = os.path.join(donut_path, "model")
    processor_path = os.path.join(donut_path, "processor")
    processor = DonutProcessor.from_pretrained(processor_path)
    model = VisionEncoderDecoderModel.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    evaluate_generation_configs_funsd(model, processor, device, generation_configs)
