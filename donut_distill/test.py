from transformers import VisionEncoderDecoderConfig, VisionEncoderDecoderModel
from student import create_student

# donut_config = VisionEncoderDecoderConfig.from_pretrained("naver-clova-ix/donut-base")
# # processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
# model = VisionEncoderDecoderModel(config=donut_config)
#
#
# t_state_dict = model.encoder.state_dict()
# t_encoder_state_dict = model.encoder.state_dict()
# t_decoder_state_dict = model.decoder.state_dict()
#
# print("Ganzes state dict:")
# for k in t_state_dict.keys():
#     print(k)
#
# print("\n----------------------------")
#
# print("Encoder state dict:")
# for k in t_encoder_state_dict.keys():
#     print(k)
#
# print("\n----------------------------")
#
# print("Decoder state dict:")
# for k in t_decoder_state_dict.keys():
#     print(k)
#
# print("\n----------------------------")


# config = AutoConfig.from_pretrained('SCUT-DLVCLab/lilt-roberta-en-base')
# model = AutoModel.from_pretrained('SCUT-DLVCLab/lilt-roberta-en-base', config=config)  # Load the LiLT backbone
# t_state_dict = model.state_dict()
# for k in t_state_dict.keys():
#     print(k)

donut_config = VisionEncoderDecoderConfig.from_pretrained("naver-clova-ix/donut-base")

teacher = VisionEncoderDecoderModel(config=donut_config)
teacher_config = donut_config
encoder_layer_map = [[0, 1], [0, 1], [0, 1, 2, 4, 5, 7, 8, 10, 11, 13], [0, 1]]
decoder_layer_map = [0, 2, 3]
vocab_map = (None,)  # TODO: Ignore for now
student_model = create_student(
    teacher=teacher,
    teacher_config=teacher_config,
    encoder_layer_map=encoder_layer_map,
    decoder_layer_map=decoder_layer_map,
)

state_dict = student_model.state_dict()
for k in state_dict.keys():
    print(k)
