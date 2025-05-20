# --- START OF FILE clui.py ---

# clui.py (Command Line User Interface)
import os
import sys
import datetime
import questionary # Import the library
import logging

# Ensure project root is in sys.path if running clui.py directly from project root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Setup basic logging for the CLUI itself if needed
clui_logger = logging.getLogger("CLUI") # Use a specific logger name
# Example: clui_logger.addHandler(logging.StreamHandler())
# clui_logger.setLevel(logging.INFO)
# Note: train_orchestrator and other modules set up their own logging.

try:
    import config
    from tokenizer_wrapper import global_tokenizer #, TOKENIZER_VOCAB_SIZE # TOKENIZER_VOCAB_SIZE not directly used here
    from clui_utils import list_model_runs_choices, select_model_run_interactive, \
                           delete_model_run_interactive, rename_model_run_interactive
    from train_orchestrator import start_or_resume_training
    # Import chat functionalities from chat_cli.chat
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_cli")) # Ensure chat_cli is findable
    from chat_cli.chat import load_model_for_chat as load_chat_model, \
                              chat_with_model as run_chat_session, \
                              load_chat_config
except ImportError as e:
    # Use questionary for user-facing error message if available, else print
    msg = f"Failed to import necessary modules for CLUI: {e}\n" \
          "Please ensure all project files are correctly placed and PYTHONPATH is set if needed.\n" \
          f"Current sys.path: {sys.path}"
    try:
        questionary.print(msg, style="bold fg:red")
    except NameError: # questionary might not have loaded
        print(f"CRITICAL CLUI ERROR: {msg}")
    sys.exit(1)

def handle_new_training():
    questionary.print("\n--- Create and Train New Model ---", style="bold underline")
    if global_tokenizer is None:
        questionary.print("ERROR: Global Tokenizer not loaded. Cannot start new training. "
                          "Please ensure tokenizer model exists and is configured correctly.", style="fg:red")
        return

    run_name_base = questionary.text(
        "Enter a base name for the new model run (e.g., 'my_model_v1'):",
        validate=lambda text: True if text.strip() else "Run name cannot be empty."
    ).ask()

    if not run_name_base or not run_name_base.strip(): # Handles empty input or Esc
        questionary.print("Model run name not provided or cancelled. Aborting new training.", style="fg:yellow")
        return
    
    run_name_base = run_name_base.strip()

    questionary.print(f"\nStarting new training run with base name: {run_name_base}", style="fg:ansiblue")
    questionary.print("Current configuration from config.py will be used for this new run:", style="fg:ansiblue")
    questionary.print(f"  - PrefixLM Total Passes: {config.PREFIXLM_TOTAL_PASSES}", style="fg:ansiblue") # Corrected from CLM
    questionary.print(f"  - Docs per PrefixLM Chunk: {config.DOCS_CHUNK_SIZE_PER_PREFIXLM_EPOCH}", style="fg:ansiblue")
    questionary.print(f"  - Supervised Epochs: {config.SUPERVISED_EPOCHS}", style="fg:ansiblue")
    questionary.print(f"  - Model D_MODEL: {config.D_MODEL}, N_HEADS: {config.N_HEADS}, EncL: {config.NUM_ENCODER_LAYERS}, DecL: {config.NUM_DECODER_LAYERS}", style="fg:ansiblue")
    questionary.print(f"  - Learning Rate: {config.NN_LEARNING_RATE}, Positional Encoding Max Len: {config.POSITIONAL_ENCODING_MAX_LEN}", style="fg:ansiblue")
    
    confirm = questionary.confirm(
        "Proceed with these settings to start a new training run?",
        default=True,
        auto_enter=False
    ).ask()

    if confirm:
        start_or_resume_training(run_name_base, resume_existing=False)
    else:
        questionary.print("New training cancelled.", style="fg:yellow")

def handle_resume_training():
    questionary.print("\n--- Resume Training Existing Model ---", style="bold underline")
    if global_tokenizer is None:
        questionary.print("ERROR: Global Tokenizer not loaded. Cannot resume training.", style="fg:red")
        return

    model_dirs = list_model_runs_choices()
    if not model_dirs:
        questionary.print("No model runs found in 'trained_models/' to resume.", style="fg:yellow")
        return

    selected_run_dir_path = select_model_run_interactive(
        prompt_message="Select model run to RESUME:",
        model_dirs=model_dirs
    )

    if selected_run_dir_path: # Not None if user selected something
        run_name_to_resume = os.path.basename(selected_run_dir_path)
        questionary.print(f"\nAttempting to resume training for: {run_name_to_resume}", style="fg:ansiblue")
        questionary.print("Model architecture will be loaded from its 'config_snapshot.json'.", style="fg:ansiblue")
        questionary.print("Current 'config.py' settings (e.g., epochs, passes, LR) will apply for *new* phases or continuation.", style="fg:ansiblue")
        
        confirm = questionary.confirm(
            "Proceed with resuming this training run?",
            default=True,
            auto_enter=False
        ).ask()
        if confirm:
            start_or_resume_training(run_name_to_resume, resume_existing=True)
        else:
            questionary.print("Resume training cancelled.", style="fg:yellow")
    else: # User cancelled selection
        questionary.print("No model selected for resume. Operation cancelled.", style="fg:yellow")


def handle_chat():
    questionary.print("\n--- Chat with Existing Model ---", style="bold underline")
    if global_tokenizer is None:
        questionary.print("ERROR: Global Tokenizer not loaded. Cannot start chat.", style="fg:red")
        return
        
    model_dirs = list_model_runs_choices()
    if not model_dirs:
        questionary.print("No model runs found in 'trained_models/' to chat with.", style="fg:yellow")
        return

    selected_run_dir_path = select_model_run_interactive(
        prompt_message="Select model run to CHAT with:",
        model_dirs=model_dirs
    )

    if selected_run_dir_path: # Not None
        questionary.print(f"Loading model from: {selected_run_dir_path} for chat...", style="fg:ansiblue")
        chat_gen_params = load_chat_config() 
        # load_chat_model now handles reading config_snapshot.json for architecture
        model_instance, tokenizer_instance, model_type = load_chat_model(selected_run_dir_path)
        if model_instance and tokenizer_instance:
            run_chat_session(model_instance, tokenizer_instance, model_type, chat_gen_params)
        else:
            questionary.print(f"Failed to load model for chat from {selected_run_dir_path}. Check logs for errors.", style="fg:red")
    else:
        questionary.print("No model selected for chat. Operation cancelled.", style="fg:yellow")


def handle_manage_models():
    while True:
        questionary.print("\n--- Manage Models ---", style="bold underline")
        
        model_dirs = list_model_runs_choices() 
        if not model_dirs:
             questionary.print("No model runs found in 'trained_models/' to manage.", style="fg:yellow")
             go_back = questionary.confirm("Go back to main menu?", default=True).ask()
             if go_back or go_back is None: return 
             continue 

        questionary.print("Current model runs (most recent first):", style="fg:cyan")
        for md_name in model_dirs: # model_dirs is already sorted
            questionary.print(f"  - {md_name}", style="fg:cyan")

        sub_choice_action = questionary.select(
            "Model Management Options:",
            choices=[
                "Rename a Model Run",
                "Delete a Model Run",
                questionary.Separator(),
                "Back to Main Menu"
            ],
            use_shortcuts=True
        ).ask()

        if sub_choice_action == "Rename a Model Run":
            selected_run_to_rename_path = select_model_run_interactive(
                prompt_message="Select model run to RENAME:",
                model_dirs=model_dirs # Pass the already fetched and sorted list
            )
            if selected_run_to_rename_path: # Not None
                rename_model_run_interactive(selected_run_to_rename_path)
        elif sub_choice_action == "Delete a Model Run":
            selected_run_to_delete_path = select_model_run_interactive(
                prompt_message="Select model run to DELETE:",
                model_dirs=model_dirs
            )
            if selected_run_to_delete_path: # Not None
                delete_model_run_interactive(selected_run_to_delete_path)
        elif sub_choice_action == "Back to Main Menu" or sub_choice_action is None: # None if Esc
            break 
        else: 
            questionary.print("Invalid choice.", style="fg:red") # Should not happen

def main_menu_interactive():
    # os.system('cls' if os.name == 'nt' else 'clear') # Optional: clear screen
    questionary.print("============================================", style="bold fg:ansimagenta")
    questionary.print("   Absolute Zero - Model Training Suite   ", style="bold fg:ansimagenta")
    questionary.print("============================================", style="bold fg:ansimagenta")

    while True:
        action = questionary.select(
            "Main Menu - What would you like to do?",
            choices=[
                questionary.Choice("🚀 Create and Train New Model", value="new_train"),
                questionary.Choice("🔄 Resume Training Existing Model", value="resume_train"),
                questionary.Choice("💬 Chat with Existing Model", value="chat"),
                questionary.Choice("🗂️ Manage Models (List, Rename, Delete)", value="manage"),
                questionary.Separator(),
                questionary.Choice("🚪 Exit", value="exit")
            ],
            use_shortcuts=True,
        ).ask()

        if action == "new_train":
            handle_new_training()
        elif action == "resume_train":
            handle_resume_training()
        elif action == "chat":
            handle_chat()
        elif action == "manage":
            handle_manage_models()
        elif action == "exit" or action is None: # None if Ctrl+C or Esc
            questionary.print("Exiting CLUI. Goodbye!", style="fg:green")
            break
        
        if action != "exit" and action is not None:
            questionary.press_any_key_to_continue("Press any key to return to the main menu...").ask()
            # os.system('cls' if os.name == 'nt' else 'clear') 

if __name__ == "__main__":
    # Setup basic logging for the entire application if not configured elsewhere
    # This will catch logs from all modules unless they have more specific handlers.
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(name)s - %(module)s - %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)]) # Ensure logs go to stdout

    if global_tokenizer is None:
        msg_critical = (
            "="*60 + "\n"
            "CRITICAL ERROR: Global tokenizer failed to load.\n"
            f"Please ensure '{config.TOKENIZER_MODEL_PATH}' exists and is valid.\n"
            "You might need to run 'python tokenizer_training.py' first to create the tokenizer model.\n"
            "Without a tokenizer, most functionalities (training, chat) will NOT work.\n"
            "="*60
        )
        questionary.print(msg_critical, style="bold fg:red")
        
        # Allow limited functionality (like model management) or exit
        if not questionary.confirm(
            "Tokenizer failed. Functionality will be severely limited. Continue to model management only?", 
            default=False, auto_enter=False).ask():
            sys.exit(1)
        # If user continues, they'll find most options don't work, which is expected.
    
    main_menu_interactive()