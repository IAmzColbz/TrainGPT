# --- START OF FILE chat.py ---

# chat_cli/chat.py
import torch
import os
import sys
import json
import logging

# Add project root to sys.path
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.dirname(current_script_dir) # This assumes chat_cli is one level down
if project_root_dir not in sys.path:
    sys.path.append(project_root_dir)

try:
    from nn_model import SimpleTransformerSeq2Seq # Refactored model
    from tokenizer_wrapper import global_tokenizer as main_global_tokenizer, TokenizerWrapper
    import config as main_app_config # Main project config
except ImportError as e:
    print(f"Error importing necessary modules for chat_cli: {e}")
    print(f"Ensure your PYTHONPATH is set correctly or run from project root. Current sys.path: {sys.path}")
    sys.exit(1)

logger = logging.getLogger(__name__)
# Basic logging for chat CLI, can be configured further
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - ChatCLI - %(message)s')


DEFAULT_CHAT_CONFIG = {
    "generation_max_len": 256,
    "temperature": 0.7,
    "repetition_penalty": 1.15,
    "top_p": 0.92,
    "top_k": 0 # 0 means disabled
}

def load_chat_config() -> dict:
    """Loads chat generation parameters from chat_config.json or uses defaults."""
    config_path = os.path.join(current_script_dir, "chat_config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            # Merge with defaults, user config takes precedence
            conf = {**DEFAULT_CHAT_CONFIG, **user_config}
            logger.info(f"Loaded chat generation config from {config_path}")
            return conf
        except Exception as e:
            logger.error(f"Error loading chat_config.json: {e}. Using default chat config.")
    else:
        logger.info(f"Chat config file '{config_path}' not found. Using default chat config.")
    return DEFAULT_CHAT_CONFIG.copy()


def load_model_for_chat(model_run_dir: str):
    """
    Loads a trained model for chat.
    Prioritizes supervised checkpoints, then PrefixLM, then old CLM.
    Uses config_snapshot.json for model architecture parameters.
    """
    logger.info(f"Attempting to load model for chat from run directory: {model_run_dir}")

    if main_global_tokenizer is None:
        logger.critical("CRITICAL Error: Global tokenizer (main_global_tokenizer) not initialized.")
        return None, None, "Tokenizer error"
    tokenizer_to_use = main_global_tokenizer

    # --- Determine which checkpoint to load ---
    # Priority: Best Supervised > Last Supervised > PrefixLM > Old CLM (fallback for very old models)
    ckpt_path_best_sup = os.path.join(model_run_dir, main_app_config.BEST_MODEL_SUPERVISED_CHECKPOINT_FILENAME)
    ckpt_path_last_sup = os.path.join(model_run_dir, main_app_config.SUPERVISED_CHECKPOINT_FILENAME)
    ckpt_path_prefixlm = os.path.join(model_run_dir, main_app_config.PREFIXLM_CHECKPOINT_FILENAME)
    # Fallback for the user's specific old checkpoint name, if others aren't found
    ckpt_path_old_clm = os.path.join(model_run_dir, "model_clm_pretrained_checkpoint.pth")


    checkpoint_to_load_path = None
    loaded_checkpoint_type = "Unknown"

    if os.path.exists(ckpt_path_best_sup):
        checkpoint_to_load_path = ckpt_path_best_sup
        loaded_checkpoint_type = "Best Supervised"
    elif os.path.exists(ckpt_path_last_sup):
        checkpoint_to_load_path = ckpt_path_last_sup
        loaded_checkpoint_type = "Last Supervised"
    elif os.path.exists(ckpt_path_prefixlm):
        checkpoint_to_load_path = ckpt_path_prefixlm
        loaded_checkpoint_type = "PrefixLM Pre-trained"
    elif os.path.exists(ckpt_path_old_clm):
        checkpoint_to_load_path = ckpt_path_old_clm
        loaded_checkpoint_type = "Old CLM Pre-trained (Fallback)"
        logger.warning(f"Loading fallback old CLM checkpoint: {os.path.basename(checkpoint_to_load_path)}")
    else:
        logger.error(f"No suitable checkpoint (Supervised, PrefixLM, or old CLM) found in {model_run_dir}.")
        return None, None, "No checkpoint found"
    
    logger.info(f"Selected checkpoint type: {loaded_checkpoint_type} from {os.path.basename(checkpoint_to_load_path)}")

    # --- Load Model Architecture from config_snapshot.json ---
    model_arch_params = {}
    config_snapshot_path = os.path.join(model_run_dir, main_app_config.CONFIG_SNAPSHOT_FILENAME)
    
    if os.path.exists(config_snapshot_path):
        logger.info(f"Loading architecture from config_snapshot: {config_snapshot_path}")
        try:
            with open(config_snapshot_path, 'r') as f:
                run_config_snapshot = json.load(f)
            
            model_arch_params['vocab_size'] = run_config_snapshot.get("TOKENIZER_VOCAB_SIZE_USED", tokenizer_to_use.vocab_size)
            model_arch_params['d_model'] = run_config_snapshot.get("D_MODEL", main_app_config.D_MODEL)
            model_arch_params['n_heads'] = run_config_snapshot.get("N_HEADS", main_app_config.N_HEADS)
            model_arch_params['num_encoder_layers'] = run_config_snapshot.get("NUM_ENCODER_LAYERS", main_app_config.NUM_ENCODER_LAYERS)
            model_arch_params['num_decoder_layers'] = run_config_snapshot.get("NUM_DECODER_LAYERS", main_app_config.NUM_DECODER_LAYERS)
            model_arch_params['d_ff'] = run_config_snapshot.get("D_FF", main_app_config.D_FF)
            model_arch_params['dropout'] = run_config_snapshot.get("TRANSFORMER_DROPOUT", main_app_config.TRANSFORMER_DROPOUT)
            
            # Robust PE max_len determination from snapshot
            pe_max_len = run_config_snapshot.get(
                "POSITIONAL_ENCODING_MAX_LEN", # Ideal key
                run_config_snapshot.get(
                    "MAX_TEXT_PRETRAIN_SEQ_LEN", # Fallback key from user's old snapshot for CLM
                     main_app_config.POSITIONAL_ENCODING_MAX_LEN # Final fallback to current global config
                )
            )
            model_arch_params['positional_encoding_max_len'] = pe_max_len
            logger.info(f"  Using positional_encoding_max_len: {pe_max_len} (from snapshot or fallbacks)")

            model_arch_params['pad_token_id'] = run_config_snapshot.get("PAD_TOKEN_ID", main_app_config.PAD_TOKEN_ID)
            model_arch_params['bos_token_id'] = run_config_snapshot.get("BOS_TOKEN_ID", main_app_config.BOS_TOKEN_ID)
            model_arch_params['eos_token_id'] = run_config_snapshot.get("EOS_TOKEN_ID", main_app_config.EOS_TOKEN_ID)

        except Exception as e:
            logger.error(f"Error loading or parsing {config_snapshot_path}: {e}. Model init will use current global defaults. THIS MAY FAIL for old models.")
            # Fallback to global config defaults if snapshot is missing or corrupt - this is risky for old models
            model_arch_params['vocab_size'] = tokenizer_to_use.vocab_size
            model_arch_params['d_model'] = main_app_config.D_MODEL
            model_arch_params['n_heads'] = main_app_config.N_HEADS
            model_arch_params['num_encoder_layers'] = main_app_config.NUM_ENCODER_LAYERS
            model_arch_params['num_decoder_layers'] = main_app_config.NUM_DECODER_LAYERS
            model_arch_params['d_ff'] = main_app_config.D_FF
            model_arch_params['dropout'] = main_app_config.TRANSFORMER_DROPOUT
            model_arch_params['positional_encoding_max_len'] = main_app_config.POSITIONAL_ENCODING_MAX_LEN
            model_arch_params['pad_token_id'] = main_app_config.PAD_TOKEN_ID
            model_arch_params['bos_token_id'] = main_app_config.BOS_TOKEN_ID
            model_arch_params['eos_token_id'] = main_app_config.EOS_TOKEN_ID
    else:
        logger.warning(f"Config snapshot {config_snapshot_path} not found. Using current global defaults for model architecture. This might fail for older models.")
        # Fallback to global config defaults
        model_arch_params['vocab_size'] = tokenizer_to_use.vocab_size
        model_arch_params['d_model'] = main_app_config.D_MODEL
        model_arch_params['n_heads'] = main_app_config.N_HEADS
        model_arch_params['num_encoder_layers'] = main_app_config.NUM_ENCODER_LAYERS
        model_arch_params['num_decoder_layers'] = main_app_config.NUM_DECODER_LAYERS
        model_arch_params['d_ff'] = main_app_config.D_FF
        model_arch_params['dropout'] = main_app_config.TRANSFORMER_DROPOUT
        model_arch_params['positional_encoding_max_len'] = main_app_config.POSITIONAL_ENCODING_MAX_LEN
        model_arch_params['pad_token_id'] = main_app_config.PAD_TOKEN_ID
        model_arch_params['bos_token_id'] = main_app_config.BOS_TOKEN_ID
        model_arch_params['eos_token_id'] = main_app_config.EOS_TOKEN_ID

    logger.info(f"Initializing model with params: Vocab={model_arch_params['vocab_size']}, D_model={model_arch_params['d_model']}, Heads={model_arch_params['n_heads']}, EncL={model_arch_params['num_encoder_layers']}, DecL={model_arch_params['num_decoder_layers']}, PE_Max_Len={model_arch_params['positional_encoding_max_len']}")
    
    try:
        model_instance = SimpleTransformerSeq2Seq(
            model_run_dir=model_run_dir, 
            **model_arch_params
        )
    except Exception as e_init:
        logger.error(f"Failed to initialize SimpleTransformerSeq2Seq model: {e_init}")
        import traceback
        traceback.print_exc()
        return None, None, "Model initialization error"

    logger.info(f"Loading model state from checkpoint: {os.path.basename(checkpoint_to_load_path)}")
    try:
        resumption_step_or_epoch = model_instance.load_checkpoint(checkpoint_name=os.path.basename(checkpoint_to_load_path))
        
        if resumption_step_or_epoch == 0 and not os.path.exists(checkpoint_to_load_path):
             logger.error(f"Checkpoint file {checkpoint_to_load_path} reported as not found by load_checkpoint, but initial check passed.")
        elif resumption_step_or_epoch == 0: 
            logger.error(f"Model state loading failed (e.g., architecture mismatch, PE size mismatch, or other error). Check logs from nn_model.load_checkpoint.")
            return None, None, "Model load error (compatibility or file issue)"

        model_instance.eval() 
        logger.info(f"Model loaded successfully. Last completed phase step/epoch from checkpoint: {model_instance.current_phase_step_or_epoch}.")
        return model_instance, tokenizer_to_use, loaded_checkpoint_type
    
    except Exception as e_load:
        logger.error(f"Unexpected error during model state loading: {e_load}")
        import traceback
        traceback.print_exc()
        return None, None, "Unexpected load error"


def chat_with_model(model: SimpleTransformerSeq2Seq, tokenizer: TokenizerWrapper, model_type_loaded: str, chat_gen_params: dict):
    """Interactive chat session with the loaded model."""
    print("\n--- Starting Chat Session ---")
    print(f"Model: {model_type_loaded}")
    print(f"Generation Params: MaxLen={chat_gen_params['generation_max_len']}, Temp={chat_gen_params['temperature']}, "
          f"Top-P={chat_gen_params['top_p']}, Top-K={chat_gen_params['top_k']}, RepPenalty={chat_gen_params['repetition_penalty']}")
    print("Type your prompt. Use 'exit' or 'quit' to end the session.")
    
    model.eval() 

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ['exit', 'quit']:
                print("Exiting chat session.")
                break
            if not user_input.strip():
                print("Model: ... (empty input, please type something)")
                continue
            
            # Use the model's actual positional_encoding_max_len for prompt length constraint
            # This is set during model __init__ based on snapshot or config
            max_prompt_len = model.positional_encoding.pe.size(1) // 2 # Allow half for prompt, half for generation context if needed
            if max_prompt_len <=0 : max_prompt_len = 128 # A small default if PE is tiny

            src_token_ids = torch.tensor([tokenizer.encode(
                user_input,
                add_bos=True, 
                add_eos=False, 
                max_length=max_prompt_len 
            )], device=model.device)

            generated_texts = model.generate_solution(
                src_token_ids_batch=src_token_ids,
                max_len=chat_gen_params['generation_max_len'],
                temperature=float(chat_gen_params['temperature']),
                repetition_penalty=float(chat_gen_params['repetition_penalty']),
                top_p=float(chat_gen_params['top_p']),
                top_k=int(chat_gen_params['top_k'])
            )
            
            model_response = generated_texts[0] if generated_texts else "Sorry, I couldn't generate a response."
            
            print(f"Model:\n{model_response}")

        except KeyboardInterrupt:
            print("\nChat session interrupted. Exiting.")
            break
        except Exception as e:
            logger.error(f"An error occurred during chat: {e}")
            import traceback
            traceback.print_exc()
            print("Sorry, an error occurred. Please try again.")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - ChatCLI - %(message)s')

    if main_global_tokenizer is None:
        logger.critical("FATAL: Global tokenizer (main_global_tokenizer) not loaded. Chat CLI cannot run."); sys.exit(1)
        
    chat_generation_parameters = load_chat_config()

    base_models_path = main_app_config.BASE_MODELS_DIR
    if not os.path.exists(base_models_path) or not os.listdir(base_models_path):
        logger.critical(f"No trained models found in '{base_models_path}'. Train a model first."); sys.exit(1)

    logger.info("Available trained model runs (directories):")
    model_dirs = sorted([d for d in os.listdir(base_models_path) if os.path.isdir(os.path.join(base_models_path, d))])
    
    if not model_dirs:
        logger.critical("No model run directories found inside BASE_MODELS_DIR."); sys.exit(1)

    for i, model_name in enumerate(model_dirs):
        print(f"  {i+1}. {model_name}")
    
    selected_model_run_dir = None
    while True:
        try:
            choice_str = input(f"Select model run by number (1-{len(model_dirs)}) or name: ").strip()
            if choice_str.lower() in ['exit', 'quit']: break

            if choice_str.isdigit():
                choice_idx = int(choice_str) - 1
                if 0 <= choice_idx < len(model_dirs):
                    selected_model_run_dir = os.path.join(base_models_path, model_dirs[choice_idx])
            elif choice_str in model_dirs: 
                selected_model_run_dir = os.path.join(base_models_path, choice_str)
            
            if selected_model_run_dir and os.path.isdir(selected_model_run_dir):
                logger.info(f"Loading model from: {selected_model_run_dir}")
                model_instance, tokenizer_instance, model_type_str = load_model_for_chat(selected_model_run_dir)
                
                if model_instance and tokenizer_instance:
                    chat_with_model(model_instance, tokenizer_instance, model_type_str, chat_generation_parameters)
                else:
                    logger.error(f"Failed to load model from {selected_model_run_dir}. Please check logs.")
                break 
            else:
                logger.warning("Invalid selection or directory does not exist. Please try again.")
        except ValueError:
            logger.warning("Invalid input. Please enter a number or a valid model run name.")
        except KeyboardInterrupt:
            logger.info("\nSelection interrupted. Exiting.")
            break
        except Exception as e_main:
            logger.error(f"Unexpected error in selection loop: {e_main}")
            import traceback
            traceback.print_exc()
            break