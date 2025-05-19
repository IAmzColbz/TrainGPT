# train.py
import torch
import os
import sys
import time
import datetime
from torch.utils.tensorboard import SummaryWriter
import json
import shutil # For copying config

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from tokenizer_wrapper import global_tokenizer, TOKENIZER_VOCAB_SIZE
from nn_model import SimpleTransformerSeq2Seq
import training_logic

def main_training_pipeline(model_run_name_arg: str, resume_if_exists: bool):
    if global_tokenizer is None:
        print("CRITICAL: Tokenizer not loaded. Exiting.")
        return

    # --- Setup Model Run Directory ---
    # If resuming and directory exists, use it. Otherwise, create with timestamp.
    prospective_run_dir = os.path.join(config.BASE_MODELS_DIR, model_run_name_arg)
    current_model_run_dir = ""
    is_resuming_run = False

    if resume_if_exists and os.path.exists(prospective_run_dir) and os.path.isdir(prospective_run_dir):
        current_model_run_dir = prospective_run_dir
        is_resuming_run = True
        print(f"Attempting to resume existing training run: {current_model_run_dir}")
    else:
        run_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        current_model_run_dir = os.path.join(config.BASE_MODELS_DIR, f"{model_run_name_arg}_{run_timestamp}")
        os.makedirs(current_model_run_dir, exist_ok=True)
        print(f"Starting new training run. Model directory: {current_model_run_dir}")
        is_resuming_run = False

    # TensorBoard Writer (log directory also includes the unique run identifier)
    tensorboard_log_dir = os.path.join(config.LOG_DIR, os.path.basename(current_model_run_dir))
    writer = SummaryWriter(log_dir=tensorboard_log_dir)
    print(f"TensorBoard logs will be saved to: {tensorboard_log_dir}")
    print(f"To view TensorBoard, run: tensorboard --logdir \"{os.path.abspath(config.LOG_DIR)}\"")

    # Save config snapshot for this run (only if not resuming, or if it's missing)
    config_snapshot_path = os.path.join(current_model_run_dir, config.CONFIG_SNAPSHOT_FILENAME)
    if not is_resuming_run or not os.path.exists(config_snapshot_path):
        try:
            # Create a dict of relevant config values (non-module, non-dunder)
            # Convert to string any problematic types if direct dump fails.
            conf_dict_to_save = {}
            for k, v in config.__dict__.items():
                if not k.startswith('__') and not isinstance(v, (type(config), type(os), type(sys))):
                    if isinstance(v, (list, dict, str, int, float, bool, type(None))):
                        conf_dict_to_save[k] = v
                    else:
                        conf_dict_to_save[k] = str(v) # Fallback to string for other types

            conf_dict_to_save["MODEL_RUN_NAME_ARG"] = model_run_name_arg # Actual name used
            conf_dict_to_save["TOKENIZER_VOCAB_SIZE_USED"] = TOKENIZER_VOCAB_SIZE
            with open(config_snapshot_path, 'w') as f:
                json.dump(conf_dict_to_save, f, indent=4, sort_keys=True)
            print(f"Config snapshot saved to {config_snapshot_path}")
        except Exception as e:
            print(f"Warning: Could not save config snapshot: {e}")
    else:
        print(f"Resuming run, config snapshot already exists at {config_snapshot_path}")


    # --- Initialize Model ---
    model = SimpleTransformerSeq2Seq(
        vocab_size=TOKENIZER_VOCAB_SIZE,
        d_model=config.D_MODEL, n_heads=config.N_HEADS,
        num_encoder_layers=config.NUM_ENCODER_LAYERS, num_decoder_layers=config.NUM_DECODER_LAYERS,
        d_ff=config.D_FF, dropout=config.TRANSFORMER_DROPOUT,
        model_run_dir=current_model_run_dir # Crucial: pass the determined run directory
    )
    model.to(model.device)

    # --- Training Phases ---
    training_start_time = time.time()
    
    if config.CLM_PRETRAIN_TOTAL_PASSES > 0:
        print("\n" + "="*10 + " Starting CLM Pre-training Phase " + "="*10)
        training_logic.run_clm_pretraining_phase(model, writer, model_run_name_arg) # Pass model_run_name_arg for consistency
        print("="*10 + " CLM Pre-training Phase Ended " + "="*10 + "\n")
    else:
        print("CLM Pre-training Phase skipped (CLM_PRETRAIN_TOTAL_PASSES is 0).")

    if config.SUPERVISED_EPOCHS > 0:
        print("\n" + "="*10 + " Starting Supervised Code Training Phase " + "="*10)
        training_logic.run_supervised_training_phase(model, writer, model_run_name_arg)
        print("="*10 + " Supervised Code Training Phase Ended " + "="*10 + "\n")
    else:
        print("Supervised Code Training Phase skipped (SUPERVISED_EPOCHS is 0).")

    training_duration = time.time() - training_start_time
    print(f"Total training pipeline finished in {training_duration / 3600:.2f} hours.")
    writer.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Main training script for the model.")
    parser.add_argument(
        "--run_name", 
        type=str, 
        default=f"train_run_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}",
        help="A base name for this training run. A timestamp will be added if not resuming."
    )
    parser.add_argument(
        "--resume",
        action='store_true', # Makes it a flag, true if present
        help="If set, attempts to resume training from a directory matching --run_name (without timestamp)."
    )
    args = parser.parse_args()
    
    print(f"Starting training with run_name base: {args.run_name}, resume: {args.resume}")
    print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    main_training_pipeline(args.run_name, args.resume)