MAX_WHOLE_MODEL_EPOCHS = 12
EARLY_STOPPING_PATIENCE_WHOLE_MODEL = 3
DATA_PATH = "Artifacts/Data"
CACHE_PATH = "Artifacts/Cache"
MODEL_DEBERTA = "deberta_large"
MODEL_DIR = 'Artifacts/Model'
T5_variant = "t5-large"
OFA_PATH = 'Artifacts/Model/OFA-base'
UPLOADS_PATH = 'static/uploads'
SAMPLE_PATH = 'static/EXCLAIM_Samples'
# './OFA-large'
# './OFA-huge'
# './OFA-tiny'
DEVICE = "cuda:1"
MAX_LEN = 64
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 8
LR = 1e-5
ACCUMULATION_STEPS = 2
SIM_THRESH = 71 #Textual similarity with the standard reference entitites (70 is decent)