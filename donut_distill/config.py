
''' transformer parameters '''
MODEL_ID            = 'naver-clova-ix/donut-base'
MAX_LENGTH          = 768
RESUME_FROM_CHECKPOINT_PATH= None # only used for resume_from_checkpoint option in PL
RESULT_PATH= "./result"
PRETRAINED_MODEL_NAME_OR_PATH= "naver-clova-ix/donut-base" # loading a pre-trained model (from moldehub or path)
# DATASET_TRAINING= "./preprocessed_dataset/train" # loading datasets (from moldehub or path)
# DATASET_VALIDATE= "./preprocessed_dataset/test" # loading datasets (from moldehub or path)
DATASET= "./preprocessed_dataset" # loading datasets (from moldehub or path)
DATASET_NAME_TRAINING="train"
DATASET_NAME_VALIDATE="test"
SORT_JSON_KEY= False
TRAIN_BATCH_SIZES= 4
VAL_BATCH_SIZES= 2
INPUT_SIZE= [800, 1000] # when the input resolution differs from the pre-training setting, some weights will be newly initialized (but the model training would be okay)
ALIGN_LONG_AXIS= False
NUM_NODES= 1
LR= 3e-5
WARMUP_STEPS= 60 # 800/8*30/10, 10%
MAX_EPOCHS= 50
MAX_STEPS= -1
NUM_WORKERS= 1
# val_check_interval: 1.0
# check_val_every_n_epoch: 3
GRADIENT_CLIP_VAL= 1.0
VERBOSE= True



''' training parameters'''
BATCH_SIZE          = 5
EPOCHS              = 10
LR                  = 0.000006
NUM_WORKERS         = 1                             # how many processes in parallel are preparing batches
