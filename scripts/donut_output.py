import torch
from donut_distill import config as CONFIG
from donut_distill.train_teacher import add_tokens, prepare_dataloader, prepare_model_and_processor
from donut_distill.donut_dataset import DonutDataset
from torch.utils.data import DataLoader


model, processor = prepare_model_and_processor()

add_tokens(model, processor, ["<yes/>", "<no/>"])

val_dataset = DonutDataset(
    dataset_name_or_path=CONFIG.DATASET,
    processor=processor,
    model=model,
    max_length=CONFIG.MAX_LENGTH,
    split=CONFIG.DATASET_NAME_VALIDATE,
    task_start_token="<s_docvqa>",
    prompt_end_token="<s_answer>",
    sort_json_key=False,  # cord dataset is preprocessed, so no need for this
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=CONFIG.NUM_WORKERS,
)

model.config.pad_token_id = processor.tokenizer.pad_token_id

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

model.train()
losses = []
for batch in val_dataloader:
    pixel_values, decoder_input_ids, prompt_end_index, target_sequence = batch

    labels = decoder_input_ids.clone()
    labels[labels == processor.tokenizer.pad_token_id] = (
        -100
    )  # model doesn't need to predict pad token
    labels[: torch.nonzero(labels == processor.tokenizer.convert_tokens_to_ids("<s_answer>")).sum() + 1] = -100 # model doesn't need to predict prompt (for VQA)
    pixel_values = pixel_values.to(device)
    decoder_input_ids = decoder_input_ids[:, :-1].to(device)
    labels = labels[:, 1:].to(device)

    with torch.autocast(device_type="cuda"):
        outputs = model(pixel_values,
                     decoder_input_ids=decoder_input_ids,
                     labels=labels,
                        output_attentions=True,
                        output_hidden_states=True)

        print(outputs)
        print("Loss",outputs.loss)
        print("logits",outputs.logits)
        print("decoder hidden states",outputs.decoder_hidden_states)
        print("encoder hidden states",outputs.encoder_hidden_states)
        print("decoder attentions",outputs.decoder_attentions)
        print("encoder attention",outputs.encoder_attentions)

    break
