from copy import deepcopy
from typing import List, Dict
from transformers import (
    VisionEncoderDecoderModel,
    VisionEncoderDecoderConfig,
)
import re
import donut_distill.config.config as CONFIG
import torch

def copy_encoder_layers(
    student_encoder_state_dict: Dict[str, torch.Tensor],
    teacher_encoder_state_dict: Dict[str, torch.Tensor],
    encoder_layer_map: List[List[int]],  # e.g. [[1,2], [1,2], [1,2,4,5,7,8,10,11,13,14], [1,2]]
):
    """
    Copies the encoder block weights from the teacher model to the student model
    based on the provided mapping. This function assumes that the teacher encoder
    consists of 4 stages (e.g., with [2,2,14,2] blocks) and that encoder_layer_map is a list
    of 4 lists, each mapping student block indices (0-indexed order in the student)
    to teacher block indices (provided here in 1-indexing, so a subtraction by 1 is applied).

    For keys that follow the pattern:
        encoder.layers.<stage_idx>.blocks.<block_idx>.<...>
    the student key is replaced by the teacher key using:
        teacher_block_idx = encoder_layer_map[stage_idx][student_block_idx] - 1

    Args:
        student_encoder_state_dict (Dict[str, torch.Tensor]): The state dictionary of the student's encoder.
        teacher_encoder_state_dict (Dict[str, torch.Tensor]): The state dictionary of the teacher's encoder.
        encoder_layer_map (List[List[int]]): A nested list where each inner list provides the teacher block indices
                                             (1-indexed) to use for the corresponding encoder stage.
    Raises:
        ValueError: If there is a shape mismatch between the student and teacher layers.
        KeyError: If a teacher key is not found in the teacher state dict.
    """
    # This regex matches keys like: "encoder.layers.<stage>.blocks.<block>.<...>"
    block_pattern = re.compile(r"^(encoder\.layers\.)(\d+)(\.blocks\.)(\d+)(\..+)$")
    
    for s_key in list(student_encoder_state_dict.keys()):
        m = block_pattern.match(s_key)
        if m:
            prefix, stage_str, block_prefix, s_block_str, suffix = m.groups()
            stage_idx = int(stage_str)
            s_block_idx = int(s_block_str)
            # Use the mapping for this stage. The mapping is assumed to be 1-indexed.
            teacher_block_idx = encoder_layer_map[stage_idx][s_block_idx] - 1

            # Construct the corresponding teacher key.
            t_key = f"{prefix}{stage_idx}{block_prefix}{teacher_block_idx}{suffix}"
            if t_key in teacher_encoder_state_dict:
                if student_encoder_state_dict[s_key].shape == teacher_encoder_state_dict[t_key].shape:
                    student_encoder_state_dict[s_key] = teacher_encoder_state_dict[t_key]
                else:
                    raise ValueError(
                        f"Shape mismatch for encoder key {s_key}: student shape {student_encoder_state_dict[s_key].shape} vs teacher key {t_key} shape {teacher_encoder_state_dict[t_key].shape}"
                    )
            else:
                raise KeyError(f"Teacher key {t_key} not found in teacher encoder state dict")

def copy_decoder_layers(
    student_decoder_state_dict: Dict[str, torch.Tensor], 
    teacher_decoder_state_dict: Dict[str, torch.Tensor], 
    decoder_layer_map: List[int]
):
    """
    Copies the decoder layer weights from the teacher model to the student model based on the provided mapping.

    Args:
        student_decoder_state_dict (Dict[str, torch.Tensor]): The state dictionary of the student's decoder layers.
        teacher_decoder_state_dict (Dict[str, torch.Tensor]): The state dictionary of the teacher's decoder layers.
        decoder_layer_map (List[int]): A list mapping student decoder layers to the corresponding teacher layers.

    Raises:
        ValueError: If there is a shape mismatch between the student and teacher layers.
        KeyError: If a teacher layer key is not found in the teacher's state dictionary.
    """
    # This regex matches keys starting with "decoder.layers.<number>."
    pattern = re.compile(r"^(decoder\.layers\.)(\d+)(\..+)$")
    
    # Iterate through each key in the student decoder's state dict
    for s_key in list(student_decoder_state_dict.keys()):
        m = pattern.match(s_key)  # Match keys following the decoder layer pattern
        
        if m:
            # Extract layer number from the student key
            prefix, s_layer_str, suffix = m.groups()
            s_layer_idx = int(s_layer_str)
            
            # Look up the corresponding teacher layer index from the decoder_layer_map
            t_layer_idx = decoder_layer_map[s_layer_idx]
            
            # Construct the corresponding teacher key
            t_key = f"{prefix}{t_layer_idx}{suffix}"
            
            if t_key in teacher_decoder_state_dict:
                # Check if the shapes match before copying weights
                if student_decoder_state_dict[s_key].shape == teacher_decoder_state_dict[t_key].shape:
                    student_decoder_state_dict[s_key] = teacher_decoder_state_dict[t_key]
                else:
                    raise ValueError(
                        f"Shape mismatch: Student layer {s_key} {student_decoder_state_dict[s_key].shape} "
                        f"and Teacher layer {t_key} {teacher_decoder_state_dict[t_key].shape} don't match."
                    )
            else:
                # Raise an error if the teacher layer key is missing
                raise KeyError(f"Teacher key {t_key} not found in teacher state dict")


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


def create_student_small_with_encoder(
    teacher: VisionEncoderDecoderModel,
    teacher_config: VisionEncoderDecoderConfig,
    encoder_layer_map: List[List[int]],  # e.g. [[1,2], [1,2], [1,2,4,5,7,8,10,11,13,14], [1,2]] (Teacher: [2,2,14,2])
    decoder_layer_map: List[int],  # Maps teacher's decoder layers to the student (Teacher has 4 layers)
) -> VisionEncoderDecoderModel:
    """
    Creates a smaller student model by selecting and copying layers from the teacher model.
    This function does NOT perform distillation, it only initializes a reduced version of the model.

    Args:
        teacher (VisionEncoderDecoderModel): The pre-trained teacher model.
        teacher_config (VisionEncoderDecoderConfig): The configuration of the teacher model.
        encoder_layer_map (List[List[int]]): Mapping of teacher encoder layers to student layers.
        decoder_layer_map (List[int]): Mapping of teacher decoder layers to student layers.

    Returns:
        VisionEncoderDecoderModel: A smaller student model with selected layers from the teacher.
    """

    # Initialize student model with a deep copy of the teacher's configuration
    config = deepcopy(teacher_config)
    config.decoder = deepcopy(teacher_config.decoder)
    config.encoder = deepcopy(teacher_config.encoder)

    # Reduce the number of decoder layers according to the provided mapping
    config.decoder.decoder_layers = len(decoder_layer_map)
    config.encoder.depths = [len(mapping) for mapping in encoder_layer_map]

    student = VisionEncoderDecoderModel(config=config)

    # Get state dictionaries for weight transfer
    t_state_dict = teacher.state_dict()
    t_encoder_state_dict = teacher.encoder.state_dict()
    t_decoder_state_dict = teacher.decoder.state_dict()

    # Load all available weights from teacher into student (some layers will be missing)
    student.load_state_dict(t_state_dict, strict=False)

    # Copy selected encoder block weights based on the provided mapping
    s_encoder_state_dict = student.encoder.state_dict()
    copy_encoder_layers(s_encoder_state_dict, t_encoder_state_dict, encoder_layer_map)
    student.encoder.load_state_dict(s_encoder_state_dict, strict=True)

    # Copy selected decoder weights based on the provided mapping
    s_decoder_state_dict = student.decoder.state_dict()
    copy_decoder_layers(s_decoder_state_dict, t_decoder_state_dict, decoder_layer_map)
    student.decoder.load_state_dict(s_decoder_state_dict, strict=True)

    return student


def create_student_small(
    teacher: VisionEncoderDecoderModel,
    teacher_config: VisionEncoderDecoderConfig,
    encoder_layer_map: List[List[int]],  # e.g. [[1,2], [1,2], [1,2,4,5,7,8,10,11,13,14], [1,2]] (Teacher: [2,2,14,2])
    decoder_layer_map: List[int],  # Maps teacher's decoder layers to the student (Teacher has 4 layers)
) -> VisionEncoderDecoderModel:
    """
    Creates a smaller student model by selecting and copying layers from the teacher model.
    This function does NOT perform distillation, it only initializes a reduced version of the model.

    Args:
        teacher (VisionEncoderDecoderModel): The pre-trained teacher model.
        teacher_config (VisionEncoderDecoderConfig): The configuration of the teacher model.
        encoder_layer_map (List[List[int]]): Mapping of teacher encoder layers to student layers.
        decoder_layer_map (List[int]): Mapping of teacher decoder layers to student layers.

    Returns:
        VisionEncoderDecoderModel: A smaller student model with selected layers from the teacher.
    """

    # Initialize student model with a deep copy of the teacher's configuration
    config = deepcopy(teacher_config)
    config.decoder = deepcopy(teacher_config.decoder)
    config.encoder = deepcopy(teacher_config.encoder)

    # Reduce the number of decoder layers according to the provided mapping
    config.decoder.decoder_layers = len(decoder_layer_map)

    student = VisionEncoderDecoderModel(config=config)

    # Get state dictionaries for weight transfer
    t_state_dict = teacher.state_dict()
    t_encoder_state_dict = teacher.encoder.state_dict()
    t_decoder_state_dict = teacher.decoder.state_dict()

    # Load all available weights from teacher into student (some layers will be missing)
    student.load_state_dict(t_state_dict, strict=False)

    # Get the newly initialized student decoder state dict
    s_decoder_state_dict = student.decoder.state_dict()

    # Copy selected decoder weights based on the provided mapping
    copy_decoder_layers(s_decoder_state_dict, t_decoder_state_dict, decoder_layer_map)

    # Load the exact encoder weights from the teacher
    student.encoder.load_state_dict(t_encoder_state_dict, strict=True)

    # Load the modified decoder weights into the student model
    student.decoder.load_state_dict(s_decoder_state_dict, strict=True)

    return student

if __name__ == "__main__":

    from donut_distill.models.helpers import prepare_model_and_processor

    model, processor, donut_config = prepare_model_and_processor(
        special_tokens=["<yes/>", "<no/>"], return_config=True, load_teacher=CONFIG.DISTILL
    )

    student_model = create_student_small(
        teacher=model,
        teacher_config=donut_config,
        encoder_layer_map=CONFIG.ENCODER_LAYER_MAP,
        decoder_layer_map=CONFIG.DECODER_LAYER_MAP,
    )

    from pathlib import Path
    model_dir = Path(CONFIG.RESULT_PATH) / "student_untrained"

    student_model.save_pretrained(model_dir)
    processor.save_pretrained(model_dir)
    donut_config.save_pretrained(model_dir)
