# train_orchestrator.py
import torch
import os
import sys
import time
import datetime
from torch.utils.tensorboard import SummaryWriter
import json
import logging

# Ensure project root is discoverable
if __name__ == '__main__': # Add this block for direct execution testing if needed
    # This allows running this script directly for testing, 
    # assuming it's in the project root or one level down.
    project_root = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(project_root) == 'TrainGPT': # Or your project's root folder name
         sys.path.append(project_root)
    else: # If script is in a subdirectory like 'scripts/'
         sys.path.append(os.path.dirname(project_root))


import config # Main project configuration
from tokenizer_wrapper import global_tokenizer, TOKENIZER_VOCAB_SIZE
from nn_model import SimpleTransformerSeq2Seq # Refactored model
import training_logic # Refactored training logic

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
        logger.critical("CRITICAL: Tokenizer not loaded. Cannot start training.")
        return False

    current_model_run_dir = ""
    is_resuming_this_session = False

    potential_resume_dir = os.path.join(config.BASE_MODELS_DIR, model_run_name_base)

    if resume_existing:
        if os.path.exists(potential_resume_dir) and os.path.isdir(potential_resume_dir):
            current_model_run_dir = potential_resume_dir
            is_resuming_this_session = True
            logger.info(f"Attempting to resume training from existing directory: {current_model_run_dir}")
        else:
            logger.error(f"ERROR: Resume requested, but directory '{potential_resume_dir}' not found.")
            logger.info("Please provide the exact name of an existing run directory to resume.")
            return False
    else: # Start a new run
        run_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        current_model_run_dir = os.path.join(config.BASE_MODELS_DIR, f"{model_run_name_base}_{run_timestamp}")
        try:
            os.makedirs(current_model_run_dir, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create model run directory {current_model_run_dir}: {e}")
            return False
        is_resuming_this_session = False
        logger.info(f"Starting new training run. Model directory: {current_model_run_dir}")

    # --- TensorBoard Writer Setup ---
    # Use the unique, timestamped directory name for TensorBoard logs
    actual_run_dir_basename = os.path.basename(current_model_run_dir)
    tensorboard_log_dir = os.path.join(config.LOG_DIR, actual_run_dir_basename)
    try:
        writer = SummaryWriter(log_dir=tensorboard_log_dir)
        logger.info(f"TensorBoard logs will be saved to: {tensorboard_log_dir}")
        if not is_resuming_this_session:
             logger.info(f"To view TensorBoard, run: tensorboard --logdir \"{os.path.abspath(config.LOG_DIR)}\"")
    except Exception as e:
        logger.error(f"Failed to initialize TensorBoard SummaryWriter at {tensorboard_log_dir}: {e}")
        return False


    # --- Save Configuration Snapshot ---
    config_snapshot_path = os.path.join(current_model_run_dir, config.CONFIG_SNAPSHOT_FILENAME)
    if not is_resuming_this_session or not os.path.exists(config_snapshot_path):
        try:
            conf_dict_to_save = {}
            # Iterate over attributes of the config module
            for key in dir(config):
                if not key.startswith('__') and not callable(getattr(config, key)):
                    value = getattr(config, key)
                    # Ensure value is JSON serializable
                    if isinstance(value, (list, dict, str, int, float, bool, type(None))):
                        conf_dict_to_save[key] = value
                    else:
                        conf_dict_to_save[key] = str(value) # Fallback to string for other types
            
            conf_dict_to_save["MODEL_RUN_NAME_BASE_USED"] = model_run_name_base
            conf_dict_to_save["ACTUAL_MODEL_RUN_DIR"] = current_model_run_dir
            conf_dict_to_save["TOKENIZER_VOCAB_SIZE_USED"] = TOKENIZER_VOCAB_SIZE # From tokenizer_wrapper
            
            with open(config_snapshot_path, 'w') as f:
                json.dump(conf_dict_to_save, f, indent=4, sort_keys=True)
            logger.info(f"Configuration snapshot saved to {config_snapshot_path}")
        except Exception as e:
            logger.warning(f"Warning: Could not save config snapshot: {e}")
    else:
        logger.info(f"Resuming run, existing configuration snapshot found: {config_snapshot_path}")

    # --- Initialize Model ---
    # Explicitly pass configuration parameters to the model constructor for clarity and robustness.
    # The model's __init__ was refactored to accept these and also has defaults from config.py.
    try:
        model = SimpleTransformerSeq2Seq(
            vocab_size=TOKENIZER_VOCAB_SIZE, # Get vocab size from the loaded global tokenizer
            d_model=config.D_MODEL,
            n_heads=config.N_HEADS,
            num_encoder_layers=config.NUM_ENCODER_LAYERS,
            num_decoder_layers=config.NUM_DECODER_LAYERS,
            d_ff=config.D_FF,
            dropout=config.TRANSFORMER_DROPOUT,
            model_run_dir=current_model_run_dir, # Crucial for checkpointing
            learning_rate=config.NN_LEARNING_RATE,
            weight_decay=config.OPTIMIZER_WEIGHT_DECAY,
            pad_token_id=config.PAD_TOKEN_ID,
            bos_token_id=config.BOS_TOKEN_ID,
            eos_token_id=config.EOS_TOKEN_ID,
            positional_encoding_max_len=config.POSITIONAL_ENCODING_MAX_LEN
        )
        model.to(model.device) # Ensure model is on the correct device
    except Exception as e:
        logger.error(f"Failed to initialize SimpleTransformerSeq2Seq model: {e}")
        import traceback
        traceback.print_exc()
        if writer: writer.close()
        return False

    # --- Training Lifecycle ---
    training_start_time = time.time()
    
    try:
        # --- PrefixLM Pre-training Phase ---
        if config.PREFIXLM_TOTAL_PASSES > 0:
            logger.info("\n" + "="*15 + " Starting/Resuming PrefixLM Pre-training Phase " + "="*15)
            training_logic.run_prefix_lm_pretraining_phase(
                model=model, 
                writer=writer, 
                model_run_name=actual_run_dir_basename # Pass unique run name for caching
            )
            logger.info("="*15 + " PrefixLM Pre-training Phase Ended/Paused " + "="*15 + "\n")
        else:
            logger.info("PrefixLM Pre-training Phase skipped (PREFIXLM_TOTAL_PASSES is 0 or not set).")

        # --- Supervised Fine-tuning Phase ---
        if config.SUPERVISED_EPOCHS > 0:
            logger.info("\n" + "="*15 + " Starting/Resuming Supervised Fine-tuning Phase " + "="*15)
            training_logic.run_supervised_training_phase(
                model=model, 
                writer=writer, 
                model_run_name=actual_run_dir_basename # Pass unique run name
            )
            logger.info("="*15 + " Supervised Fine-tuning Phase Ended/Paused " + "="*15 + "\n")
        else:
            logger.info("Supervised Fine-tuning Phase skipped (SUPERVISED_EPOCHS is 0 or not set).")

    except KeyboardInterrupt:
        logger.warning("\nTraining interrupted by user (Ctrl+C). Saving final state if applicable...")
        # Attempt to save current state. model.save_checkpoint handles the filename logic based on phase.
        # The `current_phase_step_or_epoch` in the model should reflect the last completed step/epoch.
        # We need to know which phase was active to choose the correct checkpoint name.
        # This is a simplification; a more robust way would be for the model to know its current phase.
        # For now, we assume if PrefixLM ran, it might save a PrefixLM checkpoint.
        # If Supervised ran, it might save a Supervised checkpoint.
        # The model's save_checkpoint takes the epoch/step number and the *specific* checkpoint filename.
        # The training_logic functions are responsible for calling save_checkpoint with correct names.
        # So, an interruption should ideally be caught within those, but a top-level catch is also good.
        # A simple approach here: try to save with a generic "interrupted" name or rely on the last save.
        # For now, we'll rely on the saves within training_logic phases.
        logger.info("Relaying on periodic saves within training phases for interruption recovery.")

    except Exception as e_train:
        logger.error(f"An unexpected error occurred during training: {e_train}")
        import traceback
        traceback.print_exc()
    finally:
        training_duration_seconds = time.time() - training_start_time
        logger.info(f"Training session finished or interrupted in {training_duration_seconds / 3600:.2f} hours.")
        if writer:
            writer.close()
            logger.info("TensorBoard writer closed.")
    
    return True


if __name__ == "__main__":
    # This section is for direct testing of the orchestrator, not typical user flow.
    # User flow is through clui.py.
    logger.info("train_orchestrator.py executed directly (for testing).")
    # Example: Start a new test run
    # test_run_name = f"direct_orchestrator_test_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}"
    # logger.info(f"Starting a direct test training run: {test_run_name}")
    # start_or_resume_training(test_run_name, resume_existing=False)

    # Example: Attempt to resume a specific run (replace with an actual run name)
    # resume_test_run_name = "my_model_v1_20250519-170000" # Replace with an actual existing run name
    # if os.path.exists(os.path.join(config.BASE_MODELS_DIR, resume_test_run_name)):
    #    logger.info(f"Attempting to resume direct test run: {resume_test_run_name}")
    #    start_or_resume_training(resume_test_run_name, resume_existing=True)
    # else:
    #    logger.info(f"Resume test run '{resume_test_run_name}' not found, skipping resume test.")
    pass
