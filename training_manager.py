# training_manager.py
import threading
import time
import config
from nn_model import SimpleTransformerSeq2Seq
from tokenizer_wrapper import global_tokenizer, TOKENIZER_VOCAB_SIZE
# CodeExecutor might still be useful for post-training evaluation if needed
# from code_executor import CodeExecutor
import training_logic # Will be updated for supervised training
import torch
import os
import json

# --- TrainingState Class (Simplified for Supervised Learning) ---
class TrainingState:
    def __init__(self):
        self.is_training = False
        self.should_stop = False
        self.current_phase = "Idle" # Will be "Supervised Training" or "Validating"
        self.current_epoch = 0
        self.total_epochs = config.SUPERVISED_EPOCHS # Changed from PRETRAIN_EPOCHS
        self.live_metrics = {
            "train_epoch_loss_history": [],      # list of (epoch, loss)
            "train_batch_loss_history": [],      # list of (global_batch_step, loss)
            "validation_loss_history": [],     # list of (epoch, loss)
            "validation_bleu_history": [],     # list of (epoch, bleu_score) - Example metric
            "current_train_batch_loss": 0.0,
            "current_validation_loss": 0.0,
            "log_messages": [],
            "current_global_train_batch_step": 0,
        }
        self.config_params = {
            "NN_LEARNING_RATE": config.NN_LEARNING_RATE,
            "SUPERVISED_EPOCHS": config.SUPERVISED_EPOCHS,
            "SAVE_EVERY_N_EPOCHS": config.SAVE_EVERY_N_EPOCHS,
            "VALIDATE_EVERY_N_BATCHES": config.VALIDATE_EVERY_N_BATCHES,
            "D_MODEL": config.D_MODEL, "N_HEADS": config.N_HEADS,
            "NUM_ENCODER_LAYERS": config.NUM_ENCODER_LAYERS,
            "NUM_DECODER_LAYERS": config.NUM_DECODER_LAYERS,
            "D_FF": config.D_FF, "TRANSFORMER_DROPOUT": config.TRANSFORMER_DROPOUT,
            "SUPERVISED_BATCH_SIZE": config.SUPERVISED_BATCH_SIZE, # Changed from PRETRAIN_BATCH_SIZE
            "LR_SCHEDULER_PATIENCE": config.LR_SCHEDULER_PATIENCE,
            "LR_SCHEDULER_FACTOR": config.LR_SCHEDULER_FACTOR,
            "CLM_PRETRAIN_EPOCHS": config.CLM_PRETRAIN_EPOCHS,
            "CLM_PRETRAIN_BATCH_SIZE": config.CLM_PRETRAIN_BATCH_SIZE,
            "MAX_TEXT_PRETRAIN_SEQ_LEN": config.MAX_TEXT_PRETRAIN_SEQ_LEN, # If editable
            "NUM_FINEWEB_FILES_TO_LOAD": config.NUM_FINEWEB_FILES_TO_LOAD, # If editable
        }
        self.lock = threading.Lock()

    def update_metric(self, key, value, step_or_epoch=None):
        with self.lock:
            if key.endswith("_history"):
                if key not in self.live_metrics: self.live_metrics[key] = []
                history_list = self.live_metrics[key]
                if step_or_epoch is not None:
                    history_list[:] = [item for item in history_list if item[0] != step_or_epoch]
                    history_list.append((step_or_epoch, value))
                    history_list.sort(key=lambda x: x[0])
                    if key == "train_batch_loss_history":
                        max_batch_history = 1000
                        if len(history_list) > max_batch_history:
                            self.live_metrics[key] = history_list[-max_batch_history:]
            else:
                self.live_metrics[key] = value

    def add_log_message(self, message):
        with self.lock:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            log_entry = f"[{timestamp}] {message}"
            if not self.live_metrics["log_messages"] or self.live_metrics["log_messages"][0] != log_entry:
                self.live_metrics["log_messages"].insert(0, log_entry)
            if len(self.live_metrics["log_messages"]) > 200:
                self.live_metrics["log_messages"].pop()

    def get_metrics(self):
        with self.lock:
            metrics_copy = {}
            for k, v in self.live_metrics.items():
                metrics_copy[k] = list(v) if isinstance(v, list) else v
            return metrics_copy

    def get_config(self):
        with self.lock: return self.config_params.copy()

    def update_config_param(self, key, value):
        with self.lock:
            if key in self.config_params:
                try:
                    original_type = type(self.config_params[key])
                    casted_value = value
                    if original_type == bool: casted_value = str(value).lower() in ['true', '1', 'yes', 'on']
                    elif original_type == int: casted_value = int(float(value))
                    elif original_type == float: casted_value = float(value)

                    self.config_params[key] = casted_value
                    if hasattr(config, key): setattr(config, key, casted_value)
                    self.add_log_message(f"Config '{key}' updated to {casted_value} (type: {type(casted_value)}).")

                    if key == "SUPERVISED_EPOCHS": self.total_epochs = casted_value
                    if key == "NN_LEARNING_RATE" and llm_model_global and hasattr(llm_model_global, 'optimizer') and llm_model_global.optimizer:
                        for param_group in llm_model_global.optimizer.param_groups:
                            param_group['lr'] = casted_value
                        self.add_log_message(f"Optimizer LR directly changed to {casted_value}. Note: Scheduler might override this.")
                    if key in ["D_MODEL", "N_HEADS", "NUM_ENCODER_LAYERS", "NUM_DECODER_LAYERS", "D_FF", "TRANSFORMER_DROPOUT"]:
                        self.add_log_message(f"Note: Model architecture param '{key}' changed. Full restart needed for effect.")
                    return True
                except ValueError as e:
                    self.add_log_message(f"Error: Invalid value '{value}' for '{key}'. Expected type {original_type}. {e}")
                    return False
                except Exception as e_gen:
                    self.add_log_message(f"Error updating config '{key}': {e_gen}")
                    return False
            self.add_log_message(f"Error: Config key '{key}' not found for update.")
            return False

shared_training_state = TrainingState()
training_thread_obj = None
llm_model_global = None # Only the NN model now
components_initialized_flag = False
current_model_run_dir_global = None

def initialize_training_components_once(model_run_name_for_dir):
    global llm_model_global, components_initialized_flag, current_model_run_dir_global

    target_run_dir = os.path.join(config.BASE_MODELS_DIR, model_run_name_for_dir)
    if components_initialized_flag and current_model_run_dir_global == target_run_dir:
        shared_training_state.add_log_message(f"Components already initialized for run: {model_run_name_for_dir}")
        return True

    if global_tokenizer is None:
        shared_training_state.add_log_message("CRITICAL ERROR: Global Tokenizer not loaded. Cannot initialize model.")
        return False

    current_model_run_dir_global = target_run_dir
    os.makedirs(current_model_run_dir_global, exist_ok=True)
    shared_training_state.add_log_message(f"Model run directory set to: {current_model_run_dir_global}")

    config_snapshot_path = os.path.join(current_model_run_dir_global, config.CONFIG_SNAPSHOT_FILENAME)
    try:
        conf_dict_to_save = {k: v for k, v in config.__dict__.items() if not k.startswith('__') and not isinstance(v, type(config))}
        conf_dict_to_save.update(shared_training_state.get_config())
        with open(config_snapshot_path, 'w') as f:
            json.dump(conf_dict_to_save, f, indent=4)
        shared_training_state.add_log_message(f"Config snapshot saved to {config_snapshot_path}")
    except Exception as e:
        shared_training_state.add_log_message(f"Warning: Could not save config snapshot: {e}")

    shared_training_state.add_log_message(f"PyTorch version: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    shared_training_state.add_log_message(f"Using Tokenizer. Vocab size for model: {TOKENIZER_VOCAB_SIZE}")

    current_config_for_init = shared_training_state.get_config()

    try:
        llm_model_global = SimpleTransformerSeq2Seq(
            vocab_size=TOKENIZER_VOCAB_SIZE,
            d_model=current_config_for_init.get("D_MODEL", config.D_MODEL),
            n_heads=current_config_for_init.get("N_HEADS", config.N_HEADS),
            num_encoder_layers=current_config_for_init.get("NUM_ENCODER_LAYERS", config.NUM_ENCODER_LAYERS),
            num_decoder_layers=current_config_for_init.get("NUM_DECODER_LAYERS", config.NUM_DECODER_LAYERS),
            d_ff=current_config_for_init.get("D_FF", config.D_FF),
            dropout=current_config_for_init.get("TRANSFORMER_DROPOUT", config.TRANSFORMER_DROPOUT),
            model_run_dir=current_model_run_dir_global
        )
        # Optimizer and LR Scheduler are now managed within training_logic or directly on the model
        # For simplicity, we can re-initialize optimizer here if LR changes, or training_logic handles it.
        # The model's own optimizer will be used.

        components_initialized_flag = True
        shared_training_state.add_log_message(f"Core model component initialized for run '{model_run_name_for_dir}'.")
        return True
    except Exception as e:
        shared_training_state.add_log_message(f"CRITICAL ERROR during component init for run '{model_run_name_for_dir}': {e}")
        import traceback; traceback.print_exc()
        components_initialized_flag = False
        return False

def training_loop_manager_target():
    global llm_model_global # Make sure this is the global model instance

    if not components_initialized_flag or not llm_model_global:
        shared_training_state.add_log_message("CRITICAL ERROR: Training thread started but components not initialized.")
        shared_training_state.is_training = False
        return

    shared_training_state.is_training = True
    shared_training_state.should_stop = False
    
    cfg = shared_training_state.get_config() # Get current config from UI/defaults

    # --- Phase 1: Optional CLM Pre-training on FineWeb Data ---
    run_clm_phase = cfg.get("CLM_PRETRAIN_EPOCHS", 0) > 0 # Check if CLM epochs are configured
    
    if run_clm_phase and not shared_training_state.should_stop:
        shared_training_state.add_log_message("--- Starting CLM Pre-training Phase (FineWeb Data) ---")
        
        # This phase loads/saves its own 'clm_pretrain' checkpoint type.
        # The model's internal epoch counter will be managed by load_checkpoint.
        # training_logic.run_clm_pretraining_phase loads 'clm_pretrain' checkpoint
        training_logic.run_clm_pretraining_phase(llm_model_global, shared_training_state)
        
        if shared_training_state.should_stop:
            shared_training_state.add_log_message("--- CLM Pre-training Phase Stopped by User ---")
        else:
            shared_training_state.add_log_message("--- CLM Pre-training Phase Completed ---")
        
        # After CLM pre-training, we want the supervised code training to start from its own epoch 0,
        # but using the weights from the CLM pre-trained model.
        # So, we don't necessarily reset shared_training_state.current_epoch here.
        # The load_checkpoint for 'supervised' in the next phase should handle this.
        # If no 'supervised' checkpoint exists, it starts from epoch 0, using current model weights.

    elif not run_clm_phase:
        shared_training_state.add_log_message("--- CLM Pre-training Phase Skipped (CLM_PRETRAIN_EPOCHS is 0 or not set) ---")


    # --- Phase 2: Main Supervised Code Training Phase ---
    if not shared_training_state.should_stop:
        shared_training_state.add_log_message("--- Starting Supervised Code Training Phase ---")
        
        # This will attempt to load a 'supervised' checkpoint.
        # If one exists, it resumes.
        # If not, it starts supervised training from epoch 0 using the current model weights
        # (which would be CLM pre-trained if that phase ran, or randomly initialized if not).
        training_logic.run_supervised_training_phase(llm_model_global, shared_training_state)
        
        if shared_training_state.should_stop:
            shared_training_state.add_log_message("--- Supervised Code Training Phase Stopped by User ---")
        else:
            shared_training_state.add_log_message("--- Supervised Code Training Phase Completed ---")

    else: # If training was stopped during or before CLM phase
        shared_training_state.add_log_message("Training stopped before main supervised code training loop could start.")

    shared_training_state.add_log_message("Training loop manager finished or was stopped.")
    shared_training_state.is_training = False
    shared_training_state.current_phase = "Idle"

def start_training_manager_thread(model_run_name: str):
    global training_thread_obj, components_initialized_flag, current_model_run_dir_global

    if not model_run_name or not model_run_name.strip():
        shared_training_state.add_log_message("Error: Model Run Name cannot be empty.")
        return False

    if shared_training_state.is_training:
        shared_training_state.add_log_message("Training is already in progress.")
        return False

    new_run_dir = os.path.join(config.BASE_MODELS_DIR, model_run_name)
    if not components_initialized_flag or current_model_run_dir_global != new_run_dir:
        components_initialized_flag = False
        if not initialize_training_components_once(model_run_name):
            shared_training_state.add_log_message(f"Failed to initialize components for run '{model_run_name}'.")
            return False

    shared_training_state.should_stop = False
    training_thread_obj = threading.Thread(target=training_loop_manager_target, daemon=True)
    training_thread_obj.start()
    shared_training_state.add_log_message(f"Training manager thread started for run '{model_run_name}'.")
    return True

def stop_training_manager_thread():
    global training_thread_obj
    if not shared_training_state.is_training and not (training_thread_obj and training_thread_obj.is_alive()):
        shared_training_state.add_log_message("Training is not currently running.")
        return False

    shared_training_state.add_log_message("Attempting to stop training...")
    shared_training_state.should_stop = True
    return True