# --- START OF FILE train_orchestrator.py ---

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
if __name__ == '__main__': 
    project_root = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(project_root) == os.path.basename(os.getcwd()): # If script is in project root
         sys.path.append(project_root)
    else: # If script is in a subdirectory relative to where it's run from
         sys.path.append(os.path.dirname(project_root))


import config # Main project configuration
from tokenizer_wrapper import global_tokenizer, TOKENIZER_VOCAB_SIZE
from nn_model import SimpleTransformerSeq2Seq 
import training_logic 

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

def start_or_resume_training(model_run_name_base: str, resume_existing: bool):
    """
    Manages the training pipeline, either starting new or resuming.
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
    else: 
        run_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        current_model_run_dir = os.path.join(config.BASE_MODELS_DIR, f"{model_run_name_base}_{run_timestamp}")
        try:
            os.makedirs(current_model_run_dir, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create model run directory {current_model_run_dir}: {e}")
            return False
        is_resuming_this_session = False
        logger.info(f"Starting new training run. Model directory: {current_model_run_dir}")

    actual_run_dir_basename = os.path.basename(current_model_run_dir) # Use this for TB and cache
    tensorboard_log_dir = os.path.join(config.LOG_DIR, actual_run_dir_basename)
    writer = None
    try:
        writer = SummaryWriter(log_dir=tensorboard_log_dir)
        logger.info(f"TensorBoard logs will be saved to: {tensorboard_log_dir}")
        if not is_resuming_this_session:
             logger.info(f"To view TensorBoard, run: tensorboard --logdir \"{os.path.abspath(config.LOG_DIR)}\"")
    except Exception as e:
        logger.error(f"Failed to initialize TensorBoard SummaryWriter at {tensorboard_log_dir}: {e}")
        return False # Cannot proceed without writer if init fails

    # --- Model Architecture Parameter Determination ---
    model_arch_params = {}
    config_snapshot_path = os.path.join(current_model_run_dir, config.CONFIG_SNAPSHOT_FILENAME)

    if is_resuming_this_session and os.path.exists(config_snapshot_path):
        logger.info(f"Resuming. Loading architecture from snapshot: {config_snapshot_path}")
        try:
            with open(config_snapshot_path, 'r') as f:
                run_config_snapshot = json.load(f)
            
            model_arch_params['vocab_size'] = run_config_snapshot.get("TOKENIZER_VOCAB_SIZE_USED", TOKENIZER_VOCAB_SIZE)
            model_arch_params['d_model'] = run_config_snapshot.get("D_MODEL", config.D_MODEL)
            model_arch_params['n_heads'] = run_config_snapshot.get("N_HEADS", config.N_HEADS)
            model_arch_params['num_encoder_layers'] = run_config_snapshot.get("NUM_ENCODER_LAYERS", config.NUM_ENCODER_LAYERS)
            model_arch_params['num_decoder_layers'] = run_config_snapshot.get("NUM_DECODER_LAYERS", config.NUM_DECODER_LAYERS)
            model_arch_params['d_ff'] = run_config_snapshot.get("D_FF", config.D_FF)
            model_arch_params['dropout'] = run_config_snapshot.get("TRANSFORMER_DROPOUT", config.TRANSFORMER_DROPOUT)
            
            pe_max_len = run_config_snapshot.get(
                "POSITIONAL_ENCODING_MAX_LEN", 
                run_config_snapshot.get("MAX_TEXT_PRETRAIN_SEQ_LEN", config.POSITIONAL_ENCODING_MAX_LEN)
            )
            model_arch_params['positional_encoding_max_len'] = pe_max_len
            logger.info(f"  Using positional_encoding_max_len: {pe_max_len} (from snapshot or fallbacks)")

            model_arch_params['pad_token_id'] = run_config_snapshot.get("PAD_TOKEN_ID", config.PAD_TOKEN_ID)
            model_arch_params['bos_token_id'] = run_config_snapshot.get("BOS_TOKEN_ID", config.BOS_TOKEN_ID)
            model_arch_params['eos_token_id'] = run_config_snapshot.get("EOS_TOKEN_ID", config.EOS_TOKEN_ID)
            logger.info("Model architecture parameters loaded from snapshot.")

        except Exception as e:
            logger.error(f"Error loading or parsing config_snapshot {config_snapshot_path}: {e}. Training cannot reliably resume with potentially incompatible architecture.")
            if writer: writer.close()
            return False # Critical failure
    else: # New run, or resuming but snapshot missing (problematic for resume)
        if is_resuming_this_session: # Snapshot missing for a resume is bad
             logger.warning(f"Resuming run, but config_snapshot.json MISSING from {current_model_run_dir}. Using current global config values. This is DANGEROUS if the original model had different architecture.")
        else: # New run, save current config
            logger.info("New run. Using current global config for model architecture and saving snapshot.")
        
        # For new run, or if snapshot missing on resume, use current global config values
        model_arch_params['vocab_size'] = TOKENIZER_VOCAB_SIZE
        model_arch_params['d_model'] = config.D_MODEL
        model_arch_params['n_heads'] = config.N_HEADS
        model_arch_params['num_encoder_layers'] = config.NUM_ENCODER_LAYERS
        model_arch_params['num_decoder_layers'] = config.NUM_DECODER_LAYERS
        model_arch_params['d_ff'] = config.D_FF
        model_arch_params['dropout'] = config.TRANSFORMER_DROPOUT
        model_arch_params['positional_encoding_max_len'] = config.POSITIONAL_ENCODING_MAX_LEN
        model_arch_params['pad_token_id'] = config.PAD_TOKEN_ID
        model_arch_params['bos_token_id'] = config.BOS_TOKEN_ID
        model_arch_params['eos_token_id'] = config.EOS_TOKEN_ID

        # Save snapshot for new runs
        if not is_resuming_this_session:
            try:
                conf_dict_to_save = {}
                for key in dir(config):
                    if not key.startswith('__') and not callable(getattr(config, key)):
                        value = getattr(config, key)
                        if isinstance(value, (list, dict, str, int, float, bool, type(None))):
                            conf_dict_to_save[key] = value
                        else: conf_dict_to_save[key] = str(value)
                
                conf_dict_to_save["MODEL_RUN_NAME_BASE_USED"] = model_run_name_base
                conf_dict_to_save["ACTUAL_MODEL_RUN_DIR"] = current_model_run_dir
                conf_dict_to_save["TOKENIZER_VOCAB_SIZE_USED"] = TOKENIZER_VOCAB_SIZE 
                # Add the specific arch params used for this run to the snapshot
                conf_dict_to_save.update(model_arch_params) 
                
                with open(config_snapshot_path, 'w') as f:
                    json.dump(conf_dict_to_save, f, indent=4, sort_keys=True)
                logger.info(f"Configuration snapshot saved to {config_snapshot_path}")
            except Exception as e: logger.warning(f"Warning: Could not save config snapshot: {e}")

    # --- Initialize Model ---
    logger.info(f"Initializing model with effective params: Vocab={model_arch_params['vocab_size']}, D_model={model_arch_params['d_model']}, Heads={model_arch_params['n_heads']}, EncL={model_arch_params['num_encoder_layers']}, DecL={model_arch_params['num_decoder_layers']}, PE_Max_Len={model_arch_params['positional_encoding_max_len']}")
    try:
        model = SimpleTransformerSeq2Seq(
            model_run_dir=current_model_run_dir, 
            learning_rate=config.NN_LEARNING_RATE, # Optimizer params are less critical for loading state
            weight_decay=config.OPTIMIZER_WEIGHT_DECAY,
            **model_arch_params # Unpack all architecture parameters
        )
        model.to(model.device) 
    except Exception as e:
        logger.error(f"Failed to initialize SimpleTransformerSeq2Seq model: {e}")
        import traceback; traceback.print_exc()
        if writer: writer.close()
        return False

    # --- Training Lifecycle ---
    training_start_time = time.time()
    
    try:
        if config.PREFIXLM_TOTAL_PASSES > 0:
            logger.info("\n" + "="*15 + " Starting/Resuming PrefixLM Pre-training Phase " + "="*15)
            training_logic.run_prefix_lm_pretraining_phase(
                model=model, 
                writer=writer, 
                model_run_name=actual_run_dir_basename 
            )
            logger.info("="*15 + " PrefixLM Pre-training Phase Ended/Paused " + "="*15 + "\n")
        else:
            logger.info("PrefixLM Pre-training Phase skipped (PREFIXLM_TOTAL_PASSES <= 0).")

        if config.SUPERVISED_EPOCHS > 0:
            logger.info("\n" + "="*15 + " Starting/Resuming Supervised Fine-tuning Phase " + "="*15)
            training_logic.run_supervised_training_phase(
                model=model, 
                writer=writer, 
                model_run_name=actual_run_dir_basename 
            )
            logger.info("="*15 + " Supervised Fine-tuning Phase Ended/Paused " + "="*15 + "\n")
        else:
            logger.info("Supervised Fine-tuning Phase skipped (SUPERVISED_EPOCHS <= 0).")

    except KeyboardInterrupt:
        logger.warning("\nTraining interrupted by user (Ctrl+C). Relaying on periodic saves within training phases for interruption recovery.")
    except Exception as e_train:
        logger.error(f"An unexpected error occurred during training: {e_train}")
        import traceback; traceback.print_exc()
    finally:
        training_duration_seconds = time.time() - training_start_time
        logger.info(f"Training session finished or interrupted in {training_duration_seconds / 3600:.2f} hours.")
        if writer:
            writer.close()
            logger.info("TensorBoard writer closed.")
    
    return True


if __name__ == "__main__":
    logger.info("train_orchestrator.py executed directly (for testing).")
    # Example: Start a new test run
    # test_run_name = f"direct_orchestrator_test_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}"
    # start_or_resume_training(test_run_name, resume_existing=False)

    # Example: Resume your specific run
    # resume_this_run = "fineweb_clm_256-FeedForwardFix_Backup" # Exact name of your directory
    # if os.path.exists(os.path.join(config.BASE_MODELS_DIR, resume_this_run)):
    #    logger.info(f"Attempting to resume specific test run: {resume_this_run}")
    #    start_or_resume_training(resume_this_run, resume_existing=True)
    # else:
    #    logger.info(f"Specific resume test run '{resume_this_run}' not found in '{config.BASE_MODELS_DIR}'. Skipping resume test.")
    pass