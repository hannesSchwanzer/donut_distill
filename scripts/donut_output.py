import torch
from donut_distill import config as CONFIG
from donut_distill.train_teacher import add_tokens, prepare_dataloader, prepare_model_and_processor
from donut_distill.donut_dataset import DonutDataset
from torch.utils.data import DataLoader
from transformers import (
    DonutProcessor,
    VisionEncoderDecoderModel,
    VisionEncoderDecoderConfig,
)
import gc



donut_config = VisionEncoderDecoderConfig.from_pretrained(CONFIG.MODEL_ID)
donut_config.encoder.image_size = [1280, 960]
donut_config.decoder.max_length = CONFIG.MAX_LENGTH

processor = DonutProcessor.from_pretrained(CONFIG.MODEL_ID)
model = VisionEncoderDecoderModel.from_pretrained(
    CONFIG.MODEL_ID, config=donut_config
)

processor.image_processor.size = ([1280, 960])[::-1]
processor.image_processor.do_align_long_axis = False

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
val_dataset_small = torch.utils.data.Subset(val_dataset, range(5))  # First 200 samples


del val_dataset
gc.collect()

val_dataloader = DataLoader(
    val_dataset_small,
    batch_size=1,
    shuffle=False,
    num_workers=CONFIG.NUM_WORKERS,
)

model.config.pad_token_id = processor.tokenizer.pad_token_id

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.half()

model.eval()
losses = []
with torch.no_grad():  # Disables unnecessary gradient tracking
    for batch in val_dataloader:
        pixel_values, input_ids, prompt_end_index, target_sequence = batch

        labels = input_ids.clone()
        labels[labels == processor.tokenizer.pad_token_id] = (
            -100
        )  # model doesn't need to predict pad token
        labels[: torch.nonzero(labels == processor.tokenizer.convert_tokens_to_ids("<s_answer>")).sum() + 1] = -100

        # Move tensors to GPU (keep them in full precision)
        pixel_values = pixel_values.to(device).half()
        decoder_input_ids = input_ids.to(device)
        labels = labels.to(device)

        # Forward pass (keep output_attentions & output_hidden_states)
        outputs = model(
            pixel_values,
            decoder_input_ids=decoder_input_ids[:, :-1],
            labels=labels[:, 1:],
            output_attentions=True,
            output_hidden_states=True
        )

        # print("Loss:", outputs.loss.shape, type(outputs.loss))
        # print("Logits:", outputs.logits.shape, type(outputs.logits))
        # print("Decoder Hidden States:", outputs.decoder_hidden_states.shape, type(outputs.decoder_hidden_states))
        # print("Encoder Hidden States:", outputs.encoder_hidden_states.shape, type(outputs.encoder_hidden_states))
        # print("Decoder Attentions:", outputs.decoder_attentions.shape, type(outputs.decoder_attentions))
        # print("Encoder Attentions:", outputs.encoder_attentions.shape, type(outputs.encoder_attentions))
        #
        # print("Loss:", outputs.loss.shape, type(outputs.loss))
        print("Logits:", type(outputs.logits), outputs.logits.shape)
        print("Decoder Hidden States:", type(outputs.decoder_hidden_states), len(outputs.decoder_hidden_states))
        for item in outputs.decoder_hidden_states:
            print(type(item), item.shape)
        print("Encoder Hidden States:", type(outputs.encoder_hidden_states), len(outputs.encoder_hidden_states))
        for item in outputs.encoder_hidden_states:
            print(type(item), item.shape)
        print("Decoder Attentions:", type(outputs.decoder_attentions), len(outputs.decoder_attentions))
        for item in outputs.decoder_attentions:
            print(type(item), item.shape)
        print("Encoder Attentions:", type(outputs.encoder_attentions), len(outputs.encoder_attentions))
        for item in outputs.encoder_attentions:
            print(type(item), item.shape)

        break  # Stop after first batch
