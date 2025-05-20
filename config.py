# --- START OF FILE config.py ---

# config.py
import os

# --- Core Project Setup ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_DIR_NAME = "runs"  # For TensorBoard and other logs
LOG_DIR = os.path.join(PROJECT_ROOT, LOG_DIR_NAME)
BASE_MODELS_DIR_NAME = "trained_models" # Where model checkpoints and run artifacts are saved
BASE_MODELS_DIR = os.path.join(PROJECT_ROOT, BASE_MODELS_DIR_NAME)
CONFIG_SNAPSHOT_FILENAME = "config_snapshot.json" # Saves a snapshot of config for each run

# --- Tokenizer Configuration ---
TOKENIZER_MODEL_DIR_NAME = "tokenizer_model" # Subdirectory for the tokenizer model
TOKENIZER_MODEL_PATH = os.path.join(PROJECT_ROOT, TOKENIZER_MODEL_DIR_NAME, "custom_tokenizer.model")
# Special Token IDs (ensure these match your SentencePiece training and TokenizerWrapper logic)
UNK_TOKEN_ID = 0
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2
PAD_TOKEN_ID = 3 # Important: Used for padding sequences, nn.Embedding, CrossEntropyLoss ignore_index
FALLBACK_VOCAB_SIZE = 8000 # Used if tokenizer fails or as a default estimate for new tokenizers

# --- Model Architecture (Default for NEW runs, existing runs use their snapshot) ---
D_MODEL = 256 
N_HEADS = 8   
NUM_ENCODER_LAYERS = 4 # Default for new models
NUM_DECODER_LAYERS = 4 # Default for new models
D_FF = D_MODEL * 4 
TRANSFORMER_DROPOUT = 0.1 
POSITIONAL_ENCODING_MAX_LEN = 1024 # Default for new models

# --- Training - General Parameters (Default for NEW runs) ---
NN_LEARNING_RATE = 5e-4 
OPTIMIZER_WEIGHT_DECAY = 0.01 
LR_SCHEDULER_WARMUP_STEPS = 2000 

# --- Sequence Lengths (Defaults for NEW runs) ---
MAX_PROBLEM_STATEMENT_TOKENS = 384 # For supervised code fine-tuning (encoder input)
MAX_GOLD_SOLUTION_TOKENS = 512     # For supervised code fine-tuning (decoder input/output)
MAX_TEXT_PRETRAIN_SEQ_LEN = 384    # For PrefixLM pre-training (max length of prefix OR suffix part)

# --- Dataset Paths and Parameters ---
# Supervised Code Fine-tuning Dataset
CODE_DATASET_DIR_NAME = "dataset/data" # Relative to PROJECT_ROOT
CODE_DATASET_DIR = os.path.join(PROJECT_ROOT, CODE_DATASET_DIR_NAME)
NUM_CODE_DATASET_FILES_TO_LOAD = 0 # 0 or negative: load all found .parquet files in CODE_DATASET_DIR

# PrefixLM Pre-training Dataset (e.g., FineWeb)
FINEWEB_DATA_DIR_NAME = "fineweb_data" # Relative to PROJECT_ROOT
FINEWEB_DATA_DIR = os.path.join(PROJECT_ROOT, FINEWEB_DATA_DIR_NAME)
NUM_FINEWEB_FILES_TO_CYCLE = 0 # 0 or negative: use all found .parquet files in FINEWEB_DATA_DIR
DOCS_CHUNK_SIZE_PER_PREFIXLM_EPOCH = 150000 # Num docs from a Parquet file for one "dataset chunk"
MIN_LEN_FOR_PREFIXLM_SPLIT = 60    # Min total tokens in a document to consider for PrefixLM split
MIN_PART_LEN_FOR_PREFIXLM = 20     # Min tokens for prefix AND suffix parts AFTER split (before truncation to MAX_TEXT_PRETRAIN_SEQ_LEN)

# --- PrefixLM Pre-training Phase ---
PREFIXLM_CHECKPOINT_FILENAME = "model_prefixlm_checkpoint.pth" # Standard name for this phase's checkpoints
PREFIXLM_TOTAL_PASSES = 3      # Total passes over selected FineWeb files for this phase
PREFIXLM_BATCH_SIZE = 12       

# --- Supervised Fine-tuning Phase (Code Training) ---
SUPERVISED_CHECKPOINT_FILENAME = "model_supervised_checkpoint.pth"
BEST_MODEL_SUPERVISED_CHECKPOINT_FILENAME = "model_best_supervised_checkpoint.pth"
SUPERVISED_EPOCHS = 15         
SUPERVISED_BATCH_SIZE = 12     
SUPERVISED_VALIDATION_SPLIT_RATIO = 0.05 
SUPERVISED_SAVE_EVERY_N_EPOCHS = 1
SUPERVISED_VALIDATE_EVERY_N_BATCHES = 200 # Validate N times within an epoch if batches_per_epoch > N

# --- Cache Directories ---
# {model_run_name} is replaced by unique run ID (e.g., my_run_YYYYMMDD-HHMMSS)
CACHE_DIR_BASE_TEMPLATE = os.path.join(PROJECT_ROOT, "dataset_cache", "{model_run_name}")

# --- Evaluation & Chat CLI ---
EXECUTION_TIMEOUT_SECONDS = 3.0 # For code_executor, if used for evaluation
FORBIDDEN_MODULES = [ # For code_executor sandboxing (if used)
    'os', 'sys', 'shutil', 'subprocess', 'socket', 'requests', 'http', 'urllib',
    'threading', 'multiprocessing', 'ctypes', 'pickle', 'shelve', 'dbm',
    'glob', 'fnmatch', 'tempfile', 'pathlib', 'importlib', 'pkgutil',
    'eval', 'exec', 'open', 'compile', 'input', '__import__',
    'exit', 'quit', 'help', 'globals', 'locals', 'vars', 'dir'
]

# --- Web Dashboard Default Training Parameters (Placeholder - Not actively used by CLUI) ---
DEFAULT_UI_TRAINING_PARAMS = {
    "NN_LEARNING_RATE": NN_LEARNING_RATE,
    "SUPERVISED_EPOCHS": SUPERVISED_EPOCHS,
    "PREFIXLM_TOTAL_PASSES": PREFIXLM_TOTAL_PASSES,
    "SUPERVISED_BATCH_SIZE": SUPERVISED_BATCH_SIZE,
    "PREFIXLM_BATCH_SIZE": PREFIXLM_BATCH_SIZE,
    "SUPERVISED_VALIDATE_EVERY_N_BATCHES": SUPERVISED_VALIDATE_EVERY_N_BATCHES,
    "LR_SCHEDULER_WARMUP_STEPS": LR_SCHEDULER_WARMUP_STEPS,
}