# train_orchestrator.py (Refactored from train.py)
import torch
import os
import sys
import time
import datetime
from torch.utils.tensorboard import SummaryWriter
import json

# Ensure project root is discoverable if this script is called directly
# (though it's intended to be called by clui.py)
if __name__ == '__main__': # Add this block for direct execution testing if needed
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from tokenizer_wrapper import global_tokenizer, TOKENIZER_VOCAB_SIZE
from nn_model import SimpleTransformerSeq2Seq
import training_logic

def start_or_resume_training(model_run_name_base: str, resume_existing: bool):
    """
    Manages the training pipeline, either starting new or resuming.
    Args:
        model_run_name_base: The base name for the run. If resuming, this should be
                             the exact name of the directory to resume (possibly including timestamp).
        resume_existing: Boolean flag, True if attempting to resume.
    Returns:
        True if training started/resumed successfully, False otherwise.
    """
    if global_tokenizer is None:
        print("CRITICAL: Tokenizer not loaded. Cannot start training.")
        return False

    current_model_run_dir = ""
    is_resuming_this_session = False

    potential_resume_dir = os.path.join(config.BASE_MODELS_DIR, model_run_name_base)

    if resume_existing:
        if os.path.exists(potential_resume_dir) and os.path.isdir(potential_resume_dir):
            current_model_run_dir = potential_resume_dir
            is_resuming_this_session = True
            print(f"Attempting to resume training from existing directory: {current_model_run_dir}")
        else:
            print(f"ERROR: Resume requested, but directory '{potential_resume_dir}' not found.")
            print("Please provide the exact name of an existing run directory to resume.")
            return False # Fail if resume is requested but dir not found
    else: # Start a new run
        run_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        current_model_run_dir = os.path.join(config.BASE_MODELS_DIR, f"{model_run_name_base}_{run_timestamp}")
        os.makedirs(current_model_run_dir, exist_ok=True)
        is_resuming_this_session = False
        print(f"Starting new training run. Model directory: {current_model_run_dir}")

    # TensorBoard Writer
    tensorboard_log_dir_name = os.path.basename(current_model_run_dir) # Use the actual dir name
    tensorboard_log_dir = os.path.join(config.LOG_DIR, tensorboard_log_dir_name)
    writer = SummaryWriter(log_dir=tensorboard_log_dir)
    print(f"TensorBoard logs will be saved to: {tensorboard_log_dir}")
    if not is_resuming_this_session: # Only print this for new runs
         print(f"To view TensorBoard, run: tensorboard --logdir \"{os.path.abspath(config.LOG_DIR)}\"")


    config_snapshot_path = os.path.join(current_model_run_dir, config.CONFIG_SNAPSHOT_FILENAME)
    if not is_resuming_this_session or not os.path.exists(config_snapshot_path):
        try:
            conf_dict_to_save = {}
            for k, v in config.__dict__.items():
                if not k.startswith('__') and not isinstance(v, (type(config), type(os), type(sys))):
                    if isinstance(v, (list, dict, str, int, float, bool, type(None))): conf_dict_to_save[k] = v
                    else: conf_dict_to_save[k] = str(v)
            conf_dict_to_save["MODEL_RUN_NAME_BASE_USED"] = model_run_name_base
            conf_dict_to_save["ACTUAL_MODEL_RUN_DIR"] = current_model_run_dir
            conf_dict_to_save["TOKENIZER_VOCAB_SIZE_USED"] = TOKENIZER_VOCAB_SIZE
            with open(config_snapshot_path, 'w') as f: json.dump(conf_dict_to_save, f, indent=4, sort_keys=True)
            print(f"Config snapshot saved to {config_snapshot_path}")
        except Exception as e: print(f"Warning: Could not save config snapshot: {e}")
    else:
        print(f"Resuming run, config snapshot exists: {config_snapshot_path}")

    model = SimpleTransformerSeq2Seq(
        vocab_size=TOKENIZER_VOCAB_SIZE,
        d_model=config.D_MODEL, n_heads=config.N_HEADS,
        num_encoder_layers=config.NUM_ENCODER_LAYERS, num_decoder_layers=config.NUM_DECODER_LAYERS,
        d_ff=config.D_FF, dropout=config.TRANSFORMER_DROPOUT,
        model_run_dir=current_model_run_dir
    )
    model.to(model.device)

    training_start_time = time.time()
    actual_run_dir_basename = os.path.basename(current_model_run_dir) # For cache naming

    try:
        if config.CLM_PRETRAIN_TOTAL_PASSES > 0:
            print("\n" + "="*10 + " Starting/Resuming PrefixLM Pre-training Phase " + "="*10)
            training_logic.run_prefix_lm_pretraining_phase(model, writer, actual_run_dir_basename)
            print("="*10 + " PrefixLM Pre-training Phase Ended/Paused " + "="*10 + "\n")
        else:
            print("PrefixLM Pre-training Phase skipped (CLM_PRETRAIN_TOTAL_PASSES is 0).")

        if config.SUPERVISED_EPOCHS > 0:
            print("\n" + "="*10 + " Starting/Resuming Supervised Code Training Phase " + "="*10)
            training_logic.run_supervised_training_phase(model, writer, actual_run_dir_basename)
            print("="*10 + " Supervised Code Training Phase Ended/Paused " + "="*10 + "\n")
        else:
            print("Supervised Code Training Phase skipped (SUPERVISED_EPOCHS is 0).")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user (Ctrl+C). Saving final state...")
        # Save current state if interrupted
        if config.CLM_PRETRAIN_TOTAL_PASSES > 0 and model.current_phase_epoch > 0 : # Check if in CLM phase
             model.save_checkpoint(model.current_phase_epoch, config.CLM_PRETRAIN_CHECKPOINT_FILENAME)
        elif config.SUPERVISED_EPOCHS > 0 and model.current_phase_epoch > 0 : # Check if in Supervised phase
             model.save_checkpoint(model.current_phase_epoch, config.SUPERVISED_CHECKPOINT_FILENAME)
        print("Model state saved due to interruption.")

    training_duration = time.time() - training_start_time
    print(f"Training session finished/interrupted in {training_duration / 3600:.2f} hours.")
    writer.close()
    return True

# Keep if __main__ block for potential direct testing, but clui.py is new entry
if __name__ == "__main__":
    print("This script is intended to be called by clui.py or with specific arguments for direct training.")
    # Example direct call for testing:
    # test_run_name = f"direct_test_run_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}"
    # print(f"Starting direct test training run: {test_run_name}")
    # start_or_resume_training(test_run_name, resume_existing=False)