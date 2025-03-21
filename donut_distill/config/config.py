''' General '''
DISTILL = False # True if you want to distill student
RESULT_PATH= "./result/docvqa" # where model should be saved
VERBOSE= True 
LOG_INTERVAL = 10 # After how many steps the logger should log
WANDB_NAME = "Finetune decoder" # Name for wandb rund
USE_STUDENT_WITHOUT_DISTILLING = False # When finetuning student (Creates student model and trains it like teacher)

''' transformer parameters '''
MODEL_ID            = 'naver-clova-ix/donut-base'
MAX_LENGTH          = 128 # Decoder Token length
INPUT_SIZE= [1280, 960] # image input size

''' Dataset parameters '''
DATASET= "./preprocessed_dataset_docvqa/" # path to dataset
DATASET_NAME_TRAINING="train" 
DATASET_NAME_VALIDATE="validation"
SORT_JSON_KEY= False
ALIGN_LONG_AXIS= False

''' Train parameters '''
TRAIN_BATCH_SIZES=4
ACCUMULATION_STEPS = 1 # >1 to enable gradient accumulation
LR= 3e-5
GRADIENT_CLIP_VAL= 0.25

NUM_NODES= 1
NUM_WORKERS= 5

WARMUP_STEPS= 10000 # For cosine scheduler
MAX_EPOCHS= 30
MAX_STEPS= -1 # -1 to disable

''' Validation parameters '''
VAL_BATCH_SIZES=1
VAL_CHECK_INTERVAL = 0.2 # After how many trainingssamples it should validate in percent (max 1)
LIMIT_VAL_BATCHES = 1 # Limit how many samples from validation set should be used in percent

''' Distillation parameters '''
TEACHER_MODEL_PATH = 'result/docvqa/best_model' # Path to teacher model
DECODER_LAYER_MAP = [0, 2, 3] # Which decoder layers should be removed
ENCODER_LAYER_MAP = [[0,1], [0,1], [0,1,2,3,4,5,6,7,8,9,10,11,12,13], [0,1]] # Which encoder layers should be removed
ALPHA = 1 # Weight for self-attention loss.
BETA = 1 # Weight for hidden states loss.
GAMMA = 1 # Weight for logit-based loss.
DELTA = 1 # Weight for cross-attention loss.
ENCODER_WEIGHT = 1 
DECODER_WEIGHT = 1

