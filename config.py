# config.py
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_DIR_NAME = "runs"; LOG_DIR = os.path.join(PROJECT_ROOT, LOG_DIR_NAME)
BASE_MODELS_DIR_NAME = "trained_models"; BASE_MODELS_DIR = os.path.join(PROJECT_ROOT, BASE_MODELS_DIR_NAME)

# Checkpoint names (CLM_PRETRAIN_CHECKPOINT_FILENAME will now store the PrefixLM trained model)
SUPERVISED_CHECKPOINT_FILENAME = "model_supervised_checkpoint.pth"
BEST_MODEL_SUPERVISED_CHECKPOINT_FILENAME = "model_best_supervised_checkpoint.pth"
CONFIG_SNAPSHOT_FILENAME = "config_snapshot.json"
CLM_PRETRAIN_CHECKPOINT_FILENAME = "model_clm_pretrained_checkpoint.pth" # Reusing this for PrefixLM stage

TOKENIZER_MODEL_DIR_NAME = "tokenizer_model"
TOKENIZER_MODEL_PATH = os.path.join(PROJECT_ROOT, TOKENIZER_MODEL_DIR_NAME, "custom_tokenizer.model")
UNK_TOKEN_ID = 0; BOS_TOKEN_ID = 1; EOS_TOKEN_ID = 2; PAD_TOKEN_ID = 3
FALLBACK_VOCAB_SIZE = 8000

D_MODEL = 256 # Keep your corrected D_MODEL
N_HEADS = 8   # Keep your corrected N_HEADS
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
D_FF = 1024 # CRITICAL: 4 * D_MODEL
TRANSFORMER_DROPOUT = 0.1

NN_LEARNING_RATE = 5e-4 # Peak LR
OPTIMIZER_WEIGHT_DECAY = 0.01
LR_SCHEDULER_WARMUP_STEPS = 4000

# Sequence Lengths
MAX_PROBLEM_STATEMENT_TOKENS = 256 # For supervised code encoder
MAX_GOLD_SOLUTION_TOKENS = 384   # For supervised code decoder
MAX_TEXT_PRETRAIN_SEQ_LEN = 256  # Max length for EITHER prefix OR suffix in PrefixLM

# Supervised Code Training (Fine-tuning after PrefixLM)
CODE_DATASET_DIR_NAME = "dataset/data"
CODE_DATASET_DIR = os.path.join(PROJECT_ROOT, CODE_DATASET_DIR_NAME)
NUM_CODE_DATASET_FILES_TO_LOAD = 11
SUPERVISED_EPOCHS = 5 # Epochs for code fine-tuning
SUPERVISED_BATCH_SIZE = 16 # Adjust based on memory after PrefixLM model size
SUPERVISED_VALIDATION_SPLIT_RATIO = 0.1
SUPERVISED_SAVE_EVERY_N_EPOCHS = 1
SUPERVISED_VALIDATE_EVERY_N_BATCHES = 200

# PrefixLM Pre-training (replaces simple CLM)
FINEWEB_DATA_DIR_NAME = "fineweb_data"
FINEWEB_DATA_DIR = os.path.join(PROJECT_ROOT, FINEWEB_DATA_DIR_NAME)
NUM_FINEWEB_FILES_TO_CYCLE = 5 # Use all your 5 FineWeb files
DOCS_CHUNK_SIZE_PER_PREFIXLM_EPOCH = 250000 # Docs from a file portion to make one dataset for a "sub-epoch"
CLM_PRETRAIN_TOTAL_PASSES = 2 # Used by train_orchestrator and training_logic for total passes
PREFIXLM_TOTAL_PASSES = 2 # How many full passes over the (NUM_FINEWEB_FILES_TO_CYCLE) files
PREFIXLM_BATCH_SIZE = 16  # Batch size for PrefixLM
PREFIXLM_VALIDATE_EVERY_N_BATCHES = 5000 # How often to run a (simpler) validation during PrefixLM

# Cache directory for PrefixLM should be distinct if format changes
CACHE_DIR_PREFIXLM_BASE = os.path.join(PROJECT_ROOT, "dataset_cache", "{model_run_name}", "fineweb_prefix_lm")


EXECUTION_TIMEOUT_SECONDS = 2.0
FORBIDDEN_MODULES = ['os', 'sys', 'shutil', 'subprocess', 'socket', 'requests', 'threading', 'multiprocessing', 'ctypes', 'pickle', 'glob', 'pathlib']