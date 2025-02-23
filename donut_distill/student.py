from copy import deepcopy
from typing import List
from transformers import (
    VisionEncoderDecoderModel,
    VisionEncoderDecoderConfig,
)
import re
import donut_distill.config as CONFIG

def copy_decoder_layers(student_decoder_state_dict, teacher_decoder_state_dict, decoder_layer_map):
    # This regex matches keys starting with "decoder.layers.<number>."
    pattern = re.compile(r"^(decoder\.layers\.)(\d+)(\..+)$")
    for s_key in list(student_decoder_state_dict.keys()):
        m = pattern.match(s_key)
        if m:
            prefix, s_layer_str, suffix = m.groups()
            s_layer_idx = int(s_layer_str)
            # Look up the corresponding teacher layer index from the mapping
            t_layer_idx = decoder_layer_map[s_layer_idx]
            # Construct the teacher key exactly
            t_key = f"{prefix}{t_layer_idx}{suffix}"
            if t_key in teacher_decoder_state_dict:
                if student_decoder_state_dict[s_key].shape == teacher_decoder_state_dict[t_key].shape:
                    student_decoder_state_dict[s_key] = teacher_decoder_state_dict[t_key]
                else:
                    raise ValueError(
                        f"Shape mismatch: Student layer {s_key} {student_decoder_state_dict[s_key].shape} "
                        f"and Teacher layer {t_key} {teacher_decoder_state_dict[t_key].shape} don't match."
                    )
            else:
                raise KeyError(f"Teacher key {t_key} not found in teacher state dict")


def create_student(
    teacher: VisionEncoderDecoderModel,
    teacher_config: VisionEncoderDecoderConfig,
    encoder_layer_map: List[List[int]],     # e.g. [[1,2], [1,2], [1,2,4,5,7,8,10,11,13,14], [1,2]] Teacher has form [2,2,14,2]
    decoder_layer_map: List[int], # Teacher has 4 Layers
    vocab_map=None,  # TODO: Ignore for now
):
    # TODO: Replace
    # Name of the layer that has the word embeddings: lilt.embeddings.word_embeddings.weight
    embed_key = "lilt.embeddings.word_embeddings.weight"
    # embed_key = 'embeddings.word_embeddings.weight'
    vocab_size = len(vocab_map) if vocab_map is not None else None

    # initialize student
    config = deepcopy(teacher_config)
    config.encoder.depths = [len(x) for x in encoder_layer_map]
    print("Encoder depths:", [len(x) for x in encoder_layer_map])
    config.decoder.decoder_layers = len(decoder_layer_map)  # initially 12
    # TODO:
    # config.vocab_size = vocab_size or config.vocab_size  # initially 50265

    config.decoder.hidden_size = teacher_config.decoder.hidden_size
    config.decoder.encoder_hidden_size = config.encoder.hidden_size

    student = VisionEncoderDecoderModel(config=config)

    t_state_dict = teacher.state_dict()
    t_encoder_state_dict = teacher.encoder.state_dict()
    t_decoder_state_dict = teacher.decoder.state_dict()

    # Temporarly delete embedding layer, otherwise we get size missmatch error
    if vocab_size is not None:
	# TODO:
        temp_state_dict = deepcopy(t_state_dict)
        del temp_state_dict[embed_key]
        student.load_state_dict(temp_state_dict, strict=False)
    else:
        student.load_state_dict(t_state_dict, strict=False)
    s_encoder_state_dict = student.encoder.state_dict()
    s_decoder_state_dict = student.decoder.state_dict()

    # copy each layer weights (check for state_dict with numbers)
    # encoder
    # for stage_no in range(len(encoder_layer_map)):
    #     for s_encoder_layer_no, t_encoder_layer_no in enumerate(encoder_layer_map[stage_no]):
    #         if t_encoder_layer_no is None:
    #             continue
    #         s_encoder_layer_no = str(s_encoder_layer_no)
    #         t_encoder_layer_no = str(t_encoder_layer_no)
    #         for s_k in s_encoder_state_dict.keys():
    #             t_k = s_k.replace(f"layers.{stage_no}.blocks.{s_encoder_layer_no}", f"layers.{stage_no}.blocks.{t_encoder_layer_no}")
    #             # print(f's_k: {s_k}, t_k:{t_k}')
    #             if f"layers.{stage_no}.blocks.{s_encoder_layer_no}" in s_k:
    #                 s_encoder_state_dict[s_k] = t_encoder_state_dict[t_k]

    # decoder
#     for s_decoder_layer_no, t_decoder_layer_no in enumerate(decoder_layer_map):
#         if t_decoder_layer_no is None:
#             continue
# # decoder.layers.1
#         s_decoder_layer_no = str(s_decoder_layer_no)
#         t_decoder_layer_no = str(t_decoder_layer_no)
#         for s_k in s_decoder_state_dict.keys():
#             t_k = s_k.replace(f"decoder.layers.{s_decoder_layer_no}", f"decoder.layers.{t_decoder_layer_no}")
#             # print(f's_k: {s_k}, t_k:{t_k}')
#             if f"decoder.layers.{s_decoder_layer_no}" in s_k:
#                 s_decoder_state_dict[s_k] = t_decoder_state_dict[t_k]
    # Copy decoder weights using the regex-based mapping
    copy_decoder_layers(s_decoder_state_dict, t_decoder_state_dict, decoder_layer_map)
    # TODO:
    # copy embedding weights
    # if vocab_size is not None:
    #     s_embed = s_state_dict[embed_key]
    #     t_embed = t_state_dict[embed_key]
    #     s_embed[: len(vocab_map)] = t_embed[vocab_map]
    #     s_state_dict[embed_key] = s_embed

    student.encoder.load_state_dict(s_encoder_state_dict, strict=True)
    student.decoder.load_state_dict(s_decoder_state_dict, strict=True)
    # for k in student.state_dict():
    # 	print(k)


    print("Teacher encoder hidden size:", teacher_config.encoder.hidden_size)
    print("Student encoder hidden size:", student.encoder.config.hidden_size)
    print("Teacher decoder hidden size:", teacher_config.decoder.hidden_size)
    print("Student decoder hidden size:", student.decoder.config.hidden_size)


    for s_key in s_decoder_state_dict:
        if 'weight' in s_key:
            print(f"Student {s_key}: {s_decoder_state_dict[s_key].shape}")
            print(f"Teacher {s_key}: {t_decoder_state_dict[s_key].shape}")

    return student



def distill_decoder(teacher_model: VisionEncoderDecoderModel, remove_layer_idx: int = 1) -> VisionEncoderDecoderModel:
    """
    Creates a distilled student model by removing one decoder layer from the teacher model.
    
    This function assumes:
      - The teacher model is a VisionEncoderDecoderModel with an attribute `decoder.layers`
        that is a list of identical layers.
      - The teacher config contains an attribute (e.g., `decoder_layers`) for the number of decoder layers.
      
    The function removes the layer at index `remove_layer_idx` (default is 1, i.e. the second layer)
    and copies the parameters from the remaining teacher layers to the corresponding student layers.
    
    Args:
        teacher_model (VisionEncoderDecoderModel): The teacher model loaded with its weights.
        remove_layer_idx (int, optional): The index of the decoder layer to remove. Default is 1.
        
    Returns:
        VisionEncoderDecoderModel: A new student model with one fewer decoder layer.
    """
    # Copy the teacher's config and adjust the number of decoder layers.
    student_config = deepcopy(teacher_model.config)
    
    # Here we assume the config uses an attribute 'decoder_layers' to record the number of layers.
    num_teacher_layers = teacher_model.config.decoder_layers
    student_config.decoder_layers = num_teacher_layers - 1
    
    # Create a new model (student) using the modified config.
    student_model = VisionEncoderDecoderModel(student_config)
    
    # Copy the encoder parameters directly from the teacher.
    student_model.encoder.load_state_dict(teacher_model.encoder.state_dict())
    
    # Now, handle the decoder.
    # This assumes that the decoder has an attribute `layers` (a list of layers).
    teacher_decoder_layers = teacher_model.decoder.layers
    student_decoder_layers = student_model.decoder.layers
    
    # Ensure the student decoder has one less layer than the teacher.
    assert len(teacher_decoder_layers) - 1 == len(student_decoder_layers), "Mismatch in expected decoder layers count."
    
    student_layer_idx = 0
    for teacher_layer_idx, teacher_layer in enumerate(teacher_decoder_layers):
        if teacher_layer_idx == remove_layer_idx:
            # Skip the layer that we want to remove.
            continue
        # Copy the full state (parameters, including attention and hidden state parameters) from teacher to student.
        student_decoder_layers[student_layer_idx].load_state_dict(teacher_layer.state_dict())
        student_layer_idx += 1
    
    return student_model

def print_config(config, indent=0):
    """Recursively print the VisionEncoderDecoderConfig."""
    spacing = "  " * indent
    if isinstance(config, dict):
        for key, value in config.items():
            print(f"{spacing}{key}:")
            print_config(value, indent + 1)
    elif hasattr(config, '__dict__'):
        for key, value in vars(config).items():
            print(f"{spacing}{key}:")
            print_config(value, indent + 1)
    else:
        print(f"{spacing}{config}")


def create_student_small(
    teacher: VisionEncoderDecoderModel,
    teacher_config: VisionEncoderDecoderConfig,
    encoder_layer_map: List[List[int]],     # e.g. [[1,2], [1,2], [1,2,4,5,7,8,10,11,13,14], [1,2]] Teacher has form [2,2,14,2]
    decoder_layer_map: List[int], # Teacher has 4 Layers
):
    # initialize student
    config = deepcopy(teacher_config)
    config.decoder = deepcopy(teacher_config.decoder)
    config.encoder = deepcopy(teacher_config.encoder)
    config.decoder.decoder_layers = len(decoder_layer_map)

    student = VisionEncoderDecoderModel.from_config(config)

    t_state_dict = teacher.state_dict()
    t_encoder_state_dict = teacher.encoder.state_dict()
    t_decoder_state_dict = teacher.decoder.state_dict()

    # Temporarly delete embedding layer, otherwise we get size missmatch error
    student.load_state_dict(t_state_dict, strict=False)
    s_decoder_state_dict = student.decoder.state_dict()

    # Copy decoder weights using the regex-based mapping
    copy_decoder_layers(s_decoder_state_dict, t_decoder_state_dict, decoder_layer_map)

    student.encoder.load_state_dict(t_encoder_state_dict, strict=True)
    student.decoder.load_state_dict(s_decoder_state_dict, strict=True)

    return student

if __name__ == "__main__":


    from transformers import (
        DonutProcessor,
        VisionEncoderDecoderModel,
        VisionEncoderDecoderConfig,
    )
    model_dir = CONFIG.MODEL_ID
    donut_config: VisionEncoderDecoderConfig = VisionEncoderDecoderConfig.from_pretrained(model_dir)
    donut_config.encoder.image_size = CONFIG.INPUT_SIZE
    donut_config.decoder.max_length = CONFIG.MAX_LENGTH

    processor: DonutProcessor = DonutProcessor.from_pretrained(model_dir)
    model: VisionEncoderDecoderModel = VisionEncoderDecoderModel.from_pretrained(
        model_dir, config=donut_config
    )

    processor.image_processor.size = CONFIG.INPUT_SIZE[::-1]
    processor.image_processor.do_align_long_axis = False

    student_model = create_student(
        teacher=model,
        teacher_config=donut_config,
        encoder_layer_map=CONFIG.ENCODER_LAYER_MAP,
        decoder_layer_map=CONFIG.DECODER_LAYER_MAP,
    )

    for key in student_model.decoder.state_dict().keys():
        print(key)


