''' General '''
DISTILL = False
RESULT_PATH= "./result/docvqa"
VERBOSE= True
LOG_INTERVAL = 10 # After how many steps the logger should log
WANDB_NAME = "Finetune decoder"

''' transformer parameters '''
MODEL_ID            = 'naver-clova-ix/donut-base'
MAX_LENGTH          = 128
INPUT_SIZE= [1280, 960] # when the input resolution differs from the pre-training setting, some weights will be newly initialized (but the model training would be okay)

''' Dataset parameters '''
DATASET= "./preprocessed_dataset_docvqa/" # loading datasets (from moldehub or path)
DATASET_NAME_TRAINING="train"
DATASET_NAME_VALIDATE="validation"
SORT_JSON_KEY= False
ALIGN_LONG_AXIS= False

''' Train parameters '''
TRAIN_BATCH_SIZES=4
ACCUMULATION_STEPS = 1
LR= 3e-5
GRADIENT_CLIP_VAL= 0.25

NUM_NODES= 1
NUM_WORKERS= 0

WARMUP_STEPS= 10000 # 800/8*30/10, 10%
MAX_EPOCHS= 30
MAX_STEPS= -1

''' Validation parameters '''
VAL_BATCH_SIZES=1
VAL_CHECK_INTERVAL = 0.2
LIMIT_VAL_BATCHES = 1

''' Distillation parameters '''
TEACHER_MODEL_PATH = 'result/docvqa/best_model'
DECODER_LAYER_MAP = [0, 2, 3]
ENCODER_LAYER_MAP = [[0,1], [0,1], [0,1,2,3,4,5,6,7,8,9,10,11,12,13], [0,1]]
ALPHA = 1 # Weight for self-attention loss.
BETA = 1 # Weight for hidden states loss.
GAMMA = 1 # Weight for logit-based loss.
DELTA = 1 # Weight for cross-attention loss.
ENCODER_WEIGHT = 1
DECODER_WEIGHT = 1

