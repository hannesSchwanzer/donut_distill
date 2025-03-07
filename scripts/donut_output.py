import torch
import donut_distill.config.config as CONFIG
from donut_distill.data.donut_dataset import DonutDataset
from donut_distill.models.student import create_student_small_with_encoder
from torch.utils.data import DataLoader
from transformers import (
    DonutProcessor,
    VisionEncoderDecoderModel,
    VisionEncoderDecoderConfig,
)

donut_config = VisionEncoderDecoderConfig.from_pretrained(CONFIG.MODEL_ID)
donut_config.encoder.image_size = [1280, 960]
donut_config.decoder.max_length = CONFIG.MAX_LENGTH

processor = DonutProcessor.from_pretrained(CONFIG.MODEL_ID)
model = VisionEncoderDecoderModel.from_pretrained(
    CONFIG.MODEL_ID, config=donut_config
)

processor.image_processor.size = ([1280, 960])[::-1]
processor.image_processor.do_align_long_axis = False

# add_tokens(model, processor, ["<yes/>", "<no/>"])

model = create_student_small_with_encoder(
    teacher=model,
    teacher_config=donut_config,
    encoder_layer_map=[[0,1], [0,1], [0,1,6,7,12,13], [0,1]],
    decoder_layer_map=[0,1,2,3])

train_dataset = DonutDataset(
    dataset_name_or_path="./preprocessed_dataset_docvqa_small/",
    processor=processor,
    model=model,
    max_length=CONFIG.MAX_LENGTH,
    split=CONFIG.DATASET_NAME_TRAINING,
    task_start_token="<s_docvqa>",
    prompt_end_token="<s_answer>",
    sort_json_key=False,  # cord dataset is preprocessed, so no need for this
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=CONFIG.NUM_WORKERS,
)

model.config.pad_token_id = processor.tokenizer.pad_token_id

for key in model.encoder.state_dict().keys():
    print(key)
# print(list(set(model.encoder.state_dict().keys())-set(student_model.encoder.state_dict().keys())))
# print(donut_config.encoder)

# exit()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.half()

model.eval()
losses = []
with torch.no_grad():  # Disables unnecessary gradient tracking
    for batch in train_dataloader:
        pixel_values, decoder_input_ids, labels = batch
        pixel_values = pixel_values.to(device).half()
        decoder_input_ids = decoder_input_ids[:, :-1].to(device)
        labels = labels[:, 1:].to(device)

        # Forward pass (keep output_attentions & output_hidden_states)
        outputs = model(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
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
        # print("Logits:", type(outputs.logits), outputs.logits.shape)
        # print("Decoder Hidden States:", type(outputs.decoder_hidden_states), len(outputs.decoder_hidden_states))
        # for item in outputs.decoder_hidden_states:
        #     print(type(item), item.shape)
        # print("Decoder Attentions:", type(outputs.decoder_attentions), len(outputs.decoder_attentions))
        # for item in outputs.decoder_attentions:
        #     print(type(item), item.shape)
        print("Encoder Hidden States:", type(outputs.encoder_hidden_states), len(outputs.encoder_hidden_states))
        for item in outputs.encoder_hidden_states:
            print(type(item), item.shape)
        print("Encoder Attentions:", type(outputs.encoder_attentions), len(outputs.encoder_attentions))
        for item in outputs.encoder_attentions:
            print(type(item), item.shape)

        break  # Stop after first batch
