from copy import deepcopy
from typing import List
from transformers import (
    VisionEncoderDecoderModel,
    VisionEncoderDecoderConfig,
)

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
    for s_decoder_layer_no, t_decoder_layer_no in enumerate(decoder_layer_map):
        if t_decoder_layer_no is None:
            continue
# decoder.layers.1
        s_decoder_layer_no = str(s_decoder_layer_no)
        t_decoder_layer_no = str(t_decoder_layer_no)
        for s_k in s_decoder_state_dict.keys():
            t_k = s_k.replace(f"decoder.layers.{s_decoder_layer_no}", f"decoder.layers.{t_decoder_layer_no}")
            # print(f's_k: {s_k}, t_k:{t_k}')
            if f"decoder.layers.{s_decoder_layer_no}" in s_k:
                s_decoder_state_dict[s_k] = t_decoder_state_dict[t_k]
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

    return student
