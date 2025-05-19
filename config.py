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
# Special Token IDs (ensure these match your SentencePiece training)
UNK_TOKEN_ID = 0
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2
PAD_TOKEN_ID = 3 # Important: Used for padding sequences
FALLBACK_VOCAB_SIZE = 8000 # Used if tokenizer fails or as a default estimate

# --- Model Architecture ---
D_MODEL = 256 # Core dimension of the Transformer model (embeddings, hidden states)
N_HEADS = 8   # Number of attention heads (must be a divisor of D_MODEL)
NUM_ENCODER_LAYERS = 4 # Number of layers in the Transformer encoder stack
NUM_DECODER_LAYERS = 4 # Number of layers in the Transformer decoder stack
D_FF = D_MODEL * 4 # Dimension of the feed-forward sub-layer within Transformer blocks
TRANSFORMER_DROPOUT = 0.1 # Dropout rate applied within Transformer layers
POSITIONAL_ENCODING_MAX_LEN = 1024 # Maximum sequence length supported by positional encodings

# --- Training - General Parameters ---
NN_LEARNING_RATE = 5e-4 # Peak learning rate for the AdamW optimizer
OPTIMIZER_WEIGHT_DECAY = 0.01 # Weight decay for the AdamW optimizer
LR_SCHEDULER_WARMUP_STEPS = 2000 # Number of warmup steps for the learning rate scheduler (reduced for potentially faster warmup)

# --- Sequence Lengths ---
# Max tokens for source (encoder input) in supervised code fine-tuning tasks
MAX_PROBLEM_STATEMENT_TOKENS = 384
# Max tokens for target (decoder input/output) in supervised code fine-tuning tasks
MAX_GOLD_SOLUTION_TOKENS = 512
# Max tokens for EITHER prefix OR suffix segment in PrefixLM pre-training
MAX_TEXT_PRETRAIN_SEQ_LEN = 384

# --- Dataset Paths and Parameters ---
# Supervised Code Fine-tuning Dataset
CODE_DATASET_DIR_NAME = "dataset/data"
CODE_DATASET_DIR = os.path.join(PROJECT_ROOT, CODE_DATASET_DIR_NAME)
NUM_CODE_DATASET_FILES_TO_LOAD = 0 # Set to 0 or negative to load all found .parquet files

# PrefixLM Pre-training Dataset (e.g., FineWeb)
FINEWEB_DATA_DIR_NAME = "fineweb_data"
FINEWEB_DATA_DIR = os.path.join(PROJECT_ROOT, FINEWEB_DATA_DIR_NAME)
NUM_FINEWEB_FILES_TO_CYCLE = 0 # Set to 0 or negative to use all found .parquet files
# Number of documents to process from a Parquet file to form one "dataset chunk" for PrefixLM
DOCS_CHUNK_SIZE_PER_PREFIXLM_EPOCH = 150000 # Slightly reduced for potentially faster chunk processing
MIN_LEN_FOR_PREFIXLM_SPLIT = 60 # Min total tokens in a document to consider splitting for PrefixLM
MIN_PART_LEN_FOR_PREFIXLM = 20  # Min tokens for prefix and suffix after split (before truncation)

# --- PrefixLM Pre-training Phase ---
PREFIXLM_CHECKPOINT_FILENAME = "model_prefixlm_checkpoint.pth"
PREFIXLM_TOTAL_PASSES = 3 # Total number of passes over the selected FineWeb files
PREFIXLM_BATCH_SIZE = 12 # Adjusted for potentially longer sequences or larger model

# --- Supervised Fine-tuning Phase (Code Training) ---
SUPERVISED_CHECKPOINT_FILENAME = "model_supervised_checkpoint.pth"
BEST_MODEL_SUPERVISED_CHECKPOINT_FILENAME = "model_best_supervised_checkpoint.pth"
SUPERVISED_EPOCHS = 15 # Increased epochs for potentially better fine-tuning
SUPERVISED_BATCH_SIZE = 12 # Adjusted
SUPERVISED_VALIDATION_SPLIT_RATIO = 0.05 # Use 5% of code data for validation
SUPERVISED_SAVE_EVERY_N_EPOCHS = 1
SUPERVISED_VALIDATE_EVERY_N_BATCHES = 200 # Validate more frequently within an epoch

# --- Cache Directories ---
# {model_run_name} will be replaced by the unique run ID (e.g., my_run_YYYYMMDD-HHMMSS)
CACHE_DIR_BASE_TEMPLATE = os.path.join(PROJECT_ROOT, "dataset_cache", "{model_run_name}")

# --- Evaluation & Chat CLI ---
EXECUTION_TIMEOUT_SECONDS = 3.0 # For code_executor, if used
FORBIDDEN_MODULES = [ # For code_executor sandboxing
    'os', 'sys', 'shutil', 'subprocess', 'socket', 'requests',
    'threading', 'multiprocessing', 'ctypes', 'pickle', 'glob', 'pathlib',
    'eval', 'exec', 'open', 'compile', 'input', '__import__',
    'exit', 'quit', 'help', 'globals', 'locals', 'vars'
]

# --- Web Dashboard Default Training Parameters (can be overridden by TrainingState) ---
# These are parameters that the UI might allow modification for,
# and TrainingState would manage their current values for a UI-driven run.
DEFAULT_UI_TRAINING_PARAMS = {
    "NN_LEARNING_RATE": NN_LEARNING_RATE,
    "SUPERVISED_EPOCHS": SUPERVISED_EPOCHS,
    "PREFIXLM_TOTAL_PASSES": PREFIXLM_TOTAL_PASSES,
    "SUPERVISED_BATCH_SIZE": SUPERVISED_BATCH_SIZE,
    "PREFIXLM_BATCH_SIZE": PREFIXLM_BATCH_SIZE,
    "SUPERVISED_VALIDATE_EVERY_N_BATCHES": SUPERVISED_VALIDATE_EVERY_N_BATCHES,
    "LR_SCHEDULER_WARMUP_STEPS": LR_SCHEDULER_WARMUP_STEPS,
    # Model architecture params (D_MODEL, N_HEADS, etc.) are generally fixed after run init
    # but could be displayed.
}