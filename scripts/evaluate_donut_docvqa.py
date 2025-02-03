from torch.utils.data import DataLoader
from transformers import (
    DonutProcessor,
    VisionEncoderDecoderModel,
    GenerationConfig,
)
import torch
from donut_distill.evaluate import evaluate_docvqa
from donut_distill.donut_dataset import DonutDataset,collate_fn_docvqa_eval 
import donut_distill.config as CONFIG

MODEL_ID = "naver-clova-ix/donut-base-finetuned-docvqa"
def validate_finedtuned_donut_on_docvqa():
    processor = DonutProcessor.from_pretrained(MODEL_ID)
    model = VisionEncoderDecoderModel.from_pretrained(
        MODEL_ID
    )

    device = torch.device("cuda")
    model.to(device)


    val_dataset = DonutDataset(
        dataset_name_or_path=CONFIG.DATASET,
        processor=processor,
        model=model,
        max_length=CONFIG.MAX_LENGTH,
        split=CONFIG.DATASET_NAME_VALIDATE,
        task_start_token="<s_docvqa>",
        prompt_end_token="<s_answer>",
        sort_json_key=CONFIG.SORT_JSON_KEY,  # cord dataset is preprocessed, so no need for this
        task="docvqa",
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=CONFIG.NUM_WORKERS,
        collate_fn=collate_fn_docvqa_eval,
    )

    with torch.autocast(device_type="cuda"):
        eval_results = evaluate_docvqa(
            model=model,
            processor=processor,
            device=device,
            val_dataloader=val_dataloader,
            generation_config=GenerationConfig(
                early_stopping=True,
                num_beams=1,
            ),
        )

    print(eval_results)


if __name__ == "__main__":
    validate_finedtuned_donut_on_docvqa()
