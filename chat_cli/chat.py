# chat_cli/chat.py
import torch
import os
import sys
import json

# Add project root to sys.path
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.dirname(current_script_dir)
sys.path.append(project_root_dir)

try:
    from nn_model import SimpleTransformerSeq2Seq
    from tokenizer_wrapper import global_tokenizer as main_global_tokenizer
    import config as main_app_config
    # from utils import parse_think_answer # Not strictly needed for pure text completion
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    print(f"Ensure your PYTHONPATH is set correctly or run from project root. Current sys.path: {sys.path}")
    sys.exit(1)

DEFAULT_CHAT_CONFIG = {
    "generation_max_len": 200,
    "temperature": 0.7,
    "repetition_penalty": 1.1, # Slightly lower default
    "top_p": 0.9,
    "top_k": 0
}

def load_chat_config():
    # Assumes chat_config.json is in the same directory as chat.py
    config_path = os.path.join(current_script_dir, "chat_config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            conf = DEFAULT_CHAT_CONFIG.copy()
            conf.update(user_config) # User config overrides defaults
            print(f"Loaded chat config from {config_path}")
            return conf
        except Exception as e:
            print(f"Error loading chat_config.json: {e}. Using default chat config.")
    else:
        print(f"{config_path} not found. Using default chat config.")
    return DEFAULT_CHAT_CONFIG.copy()

def load_model_for_chat(model_run_dir):
    config_snapshot_path = os.path.join(model_run_dir, main_app_config.CONFIG_SNAPSHOT_FILENAME)
    
    # For a base CLM model, we'd primarily look for the CLM checkpoint
    ckpt_path_clm_pretrained = os.path.join(model_run_dir, main_app_config.CLM_PRETRAIN_CHECKPOINT_FILENAME)
    # As a fallback, if you later fine-tune, you might want to load those. For now, focus on CLM.
    ckpt_path_best_supervised = os.path.join(model_run_dir, main_app_config.BEST_MODEL_SUPERVISED_CHECKPOINT_FILENAME)
    ckpt_path_last_supervised = os.path.join(model_run_dir, main_app_config.SUPERVISED_CHECKPOINT_FILENAME)

    checkpoint_to_load = None
    loaded_checkpoint_type = "Unknown"

    if os.path.exists(ckpt_path_clm_pretrained): # Prioritize CLM for current focus
        checkpoint_to_load = ckpt_path_clm_pretrained
        loaded_checkpoint_type = "CLM Pre-trained"
    elif os.path.exists(ckpt_path_best_supervised): # Fallback if CLM not found but supervised is
        checkpoint_to_load = ckpt_path_best_supervised
        loaded_checkpoint_type = "Best Supervised (Loaded for CLM-like chat)"
    elif os.path.exists(ckpt_path_last_supervised):
        checkpoint_to_load = ckpt_path_last_supervised
        loaded_checkpoint_type = "Last Supervised (Loaded for CLM-like chat)"
    else:
        print(f"No suitable CLM or Supervised checkpoint found in {model_run_dir}.")
        return None, None, "No checkpoint"

    print(f"Found checkpoint type: {loaded_checkpoint_type} at {os.path.basename(checkpoint_to_load)}")

    model_specific_config = {}
    if os.path.exists(config_snapshot_path):
        try:
            with open(config_snapshot_path, 'r') as f: model_specific_config = json.load(f)
            print(f"Loaded model architecture config from: {config_snapshot_path}")
        except Exception as e: print(f"Warning: Error loading config_snapshot.json: {e}.")
    else: print(f"Warning: Config snapshot {config_snapshot_path} not found. Using default model params from main config.")

    if main_global_tokenizer is None:
        print("CRITICAL Error: Global tokenizer not initialized."); return None, None, "Tokenizer error"
    tokenizer = main_global_tokenizer
    print(f"Using tokenizer with vocab size: {tokenizer.vocab_size}")

    d_model = model_specific_config.get("D_MODEL", main_app_config.D_MODEL)
    n_heads = model_specific_config.get("N_HEADS", main_app_config.N_HEADS)
    num_encoder_layers = model_specific_config.get("NUM_ENCODER_LAYERS", main_app_config.NUM_ENCODER_LAYERS)
    num_decoder_layers = model_specific_config.get("NUM_DECODER_LAYERS", main_app_config.NUM_DECODER_LAYERS)
    d_ff = model_specific_config.get("D_FF", main_app_config.D_FF)
    dropout = model_specific_config.get("TRANSFORMER_DROPOUT", main_app_config.TRANSFORMER_DROPOUT)
    model_vocab_size = tokenizer.vocab_size

    print(f"Initializing model with Vocab: {model_vocab_size}, D_model: {d_model}, Heads: {n_heads}...")
    model_instance = SimpleTransformerSeq2Seq(
        vocab_size=model_vocab_size, d_model=d_model, n_heads=n_heads,
        num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
        d_ff=d_ff, dropout=dropout, model_run_dir=model_run_dir
    )

    print(f"Loading model state from: {os.path.basename(checkpoint_to_load)}")
    try:
        loaded_epoch_info = model_instance.load_checkpoint(checkpoint_name=os.path.basename(checkpoint_to_load))
        model_instance.eval()
        print(f"Model loaded successfully. Checkpoint info (epoch/step): {model_instance.current_phase_epoch} (resumes from {loaded_epoch_info})")
        return model_instance, tokenizer, loaded_checkpoint_type
    except Exception as e:
        print(f"Error loading model state: {e}"); import traceback; traceback.print_exc()
        return None, None, "Load error"


def chat_with_model(model, tokenizer, model_type_loaded, chat_gen_config):
    print("\nStarting text completion session. Type 'exit' or 'quit' to end.")
    print(f"Model loaded: {model_type_loaded} (Treated as text completion model)")
    print(f"Generation params: MaxLen={chat_gen_config['generation_max_len']}, Temp={chat_gen_config['temperature']}, Top-p={chat_gen_config['top_p']}, Top-k={chat_gen_config['top_k']}, RepPenalty={chat_gen_config['repetition_penalty']}")
    model.eval()

    while True:
        try:
            user_input = input("You (text prompt to complete): ")
            if user_input.lower() in ['exit', 'quit']: break
            if not user_input.strip():
                print("Model: ... (empty input, try again)") # Or some other placeholder for empty input
                continue
            
            # For CLM, the user input is the beginning of the sequence.
            # The model is expected to continue it.
            prompt_for_model = user_input 
            
            # Use a reasonable max source length for the prompt itself
            max_src_len = main_app_config.MAX_TEXT_PRETRAIN_SEQ_LEN 

            src_token_ids = torch.tensor([tokenizer.encode(
                prompt_for_model, 
                add_bos=True,  # Add BOS to the prompt to signal start of sequence for decoder
                add_eos=False, # Typically no EOS for a prompt to be continued
                max_length=max_src_len 
            )], device=model.device)

            with torch.no_grad():
                generated_text = model.generate_solution( # This is our general generation method
                    src_token_ids,
                    max_len=chat_gen_config['generation_max_len'],
                    temperature=float(chat_gen_config['temperature']),
                    repetition_penalty=float(chat_gen_config['repetition_penalty']),
                    top_p=float(chat_gen_config['top_p']),
                    top_k=int(chat_gen_config['top_k'])
                )
            
            # The `generated_text` from `generate_solution` already decodes tokens and handles BOS/EOS stripping.
            # It starts generation with BOS and stops on EOS or max_len.
            # The output will be the continuation *including* the prompt if the model echoes it,
            # or just the continuation if the model behaves like a pure decoder.
            # Our current `generate_solution` uses src_token_ids as encoder input and then starts decoder with BOS.
            # So, the output will be the generated sequence, not necessarily an echo of the prompt + continuation.
            # This is fine for text completion.
            print(f"Model Continuation:\n{generated_text}")

        except KeyboardInterrupt: print("\nExiting chat session."); break
        except Exception as e: print(f"An error occurred: {e}"); import traceback; traceback.print_exc()

if __name__ == '__main__':
    if main_global_tokenizer is None:
        print("FATAL: Global tokenizer not loaded."); sys.exit(1)
        
    chat_generation_params = load_chat_config()

    base_models_path = main_app_config.BASE_MODELS_DIR
    if not os.path.exists(base_models_path) or not os.listdir(base_models_path):
        print(f"No trained models found in '{base_models_path}'. Train a model first."); sys.exit(1)

    print("Available trained model runs (directories):")
    model_dirs = [d for d in os.listdir(base_models_path) if os.path.isdir(os.path.join(base_models_path, d))]
    if not model_dirs: print("No model run directories found."); sys.exit(1)

    for i, model_name in enumerate(model_dirs): print(f"  {i+1}. {model_name}")
    
    while True:
        try:
            choice = input(f"Select model run by number (1-{len(model_dirs)}) or name: ")
            selected_model_run_dir = None
            if choice.isdigit():
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(model_dirs): selected_model_run_dir = os.path.join(base_models_path, model_dirs[choice_idx])
            elif choice in model_dirs: selected_model_run_dir = os.path.join(base_models_path, choice)
            
            if selected_model_run_dir and os.path.exists(selected_model_run_dir):
                print(f"Loading model from: {selected_model_run_dir}")
                model_instance, tokenizer_instance, model_type = load_model_for_chat(selected_model_run_dir)
                if model_instance and tokenizer_instance:
                    chat_with_model(model_instance, tokenizer_instance, model_type, chat_generation_params)
                else: print(f"Failed to load model from {selected_model_run_dir}.")
                break 
            else: print("Invalid selection or directory does not exist.")
        except ValueError: print("Invalid input. Please enter a number or model run name.")
        except KeyboardInterrupt: print("\nExiting."); break
        except Exception as e_main: print(f"Unexpected error: {e_main}"); import traceback; traceback.print_exc(); break