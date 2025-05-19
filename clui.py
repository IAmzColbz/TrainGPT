# clui.py (Command Line User Interface)
import os
import sys
import datetime
import questionary # Import the library

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import config
    from tokenizer_wrapper import global_tokenizer
    from clui_utils import list_model_runs_choices, select_model_run_interactive, \
                           delete_model_run_interactive, rename_model_run_interactive
    from train_orchestrator import start_or_resume_training
    from chat_cli.chat import load_model_for_chat as load_chat_model, \
                              chat_with_model as run_chat_session, \
                              load_chat_config
except ImportError as e:
    questionary.print(f"Failed to import necessary modules for CLUI: {e}", style="bold fg:red")
    questionary.print("Please ensure all project files are correctly placed and PYTHONPATH is set if needed.", style="fg:red")
    sys.exit(1)

def handle_new_training():
    questionary.print("\n--- Create and Train New Model ---", style="bold underline")
    if global_tokenizer is None:
        questionary.print("ERROR: Tokenizer not loaded. Please train/configure tokenizer first.", style="fg:red")
        return

    run_name_base = questionary.text(
        "Enter a base name for the new model run (e.g., 'my_model_v1'):",
        validate=lambda text: True if text.strip() else "Run name cannot be empty."
    ).ask()

    if not run_name_base or not run_name_base.strip():
        questionary.print("Model run name not provided. Aborting new training.", style="fg:yellow")
        return
    
    run_name_base = run_name_base.strip()

    questionary.print(f"\nStarting new training run with base name: {run_name_base}", style="fg:ansiblue")
    questionary.print("Current configuration from config.py will be used:", style="fg:ansiblue")
    questionary.print(f"  - CLM Total Passes: {config.CLM_PRETRAIN_TOTAL_PASSES}", style="fg:ansiblue")
    questionary.print(f"  - Docs per CLM Chunk: {config.DOCS_CHUNK_SIZE_PER_PREFIXLM_EPOCH}", style="fg:ansiblue")
    questionary.print(f"  - Supervised Epochs: {config.SUPERVISED_EPOCHS}", style="fg:ansiblue")
    questionary.print(f"  - Model D_MODEL: {config.D_MODEL}, D_FF: {config.D_FF}", style="fg:ansiblue")
    # Add more key configs if desired
    
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
        questionary.print("ERROR: Tokenizer not loaded.", style="fg:red")
        return

    model_dirs = list_model_runs_choices()
    if not model_dirs:
        questionary.print("No model runs found to resume.", style="fg:yellow")
        return

    selected_run_dir_path = select_model_run_interactive(
        prompt_message="Select model run to RESUME:",
        model_dirs=model_dirs
    )

    if selected_run_dir_path:
        run_name_to_resume = os.path.basename(selected_run_dir_path)
        questionary.print(f"\nAttempting to resume training for: {run_name_to_resume}", style="fg:ansiblue")
        questionary.print("Current configuration from config.py will be used for any new epochs/passes.", style="fg:ansiblue")
        
        confirm = questionary.confirm(
            "Proceed with resuming this training run?",
            default=True,
            auto_enter=False
        ).ask()
        if confirm:
            start_or_resume_training(run_name_to_resume, resume_existing=True)
        else:
            questionary.print("Resume training cancelled.", style="fg:yellow")

def handle_chat():
    questionary.print("\n--- Chat with Existing Model ---", style="bold underline")
    if global_tokenizer is None:
        questionary.print("ERROR: Tokenizer not loaded.", style="fg:red")
        return
        
    model_dirs = list_model_runs_choices()
    if not model_dirs:
        questionary.print("No model runs found to chat with.", style="fg:yellow")
        return

    selected_run_dir_path = select_model_run_interactive(
        prompt_message="Select model run to CHAT with:",
        model_dirs=model_dirs
    )

    if selected_run_dir_path:
        questionary.print(f"Loading model from: {selected_run_dir_path} for chat...", style="fg:ansiblue")
        chat_gen_params = load_chat_config() # Load chat generation parameters
        model_instance, tokenizer_instance, model_type = load_chat_model(selected_run_dir_path)
        if model_instance and tokenizer_instance:
            run_chat_session(model_instance, tokenizer_instance, model_type, chat_gen_params)
        else:
            questionary.print(f"Failed to load model for chat from {selected_run_dir_path}.", style="fg:red")

def handle_manage_models():
    while True:
        questionary.print("\n--- Manage Models ---", style="bold underline")
        
        model_dirs = list_model_runs_choices() # Get sorted list of model names
        if not model_dirs:
             questionary.print("No model runs found to manage.", style="fg:yellow")
             # Offer to go back if no models
             go_back = questionary.confirm("Go back to main menu?", default=True).ask()
             if go_back or go_back is None: return # None if Ctrl+C
             continue # Should not happen if no models

        # Display current models without numbering for this sub-menu
        questionary.print("Current model runs:", style="fg:cyan")
        for md in model_dirs:
            questionary.print(f"  - {md}", style="fg:cyan")


        sub_choice_action = questionary.select(
            "Model Management Options:",
            choices=[
                "Rename a Model Run",
                "Delete a Model Run",
                questionary.Separator(),
                "Back to Main Menu"
            ]
        ).ask()

        if sub_choice_action == "Rename a Model Run":
            selected_run_to_rename = select_model_run_interactive(
                prompt_message="Select model run to RENAME:",
                model_dirs=model_dirs # Pass the already fetched list
            )
            if selected_run_to_rename:
                rename_model_run_interactive(selected_run_to_rename)
        elif sub_choice_action == "Delete a Model Run":
            selected_run_to_delete = select_model_run_interactive(
                prompt_message="Select model run to DELETE:",
                model_dirs=model_dirs
            )
            if selected_run_to_delete:
                delete_model_run_interactive(selected_run_to_delete)
        elif sub_choice_action == "Back to Main Menu" or sub_choice_action is None:
            break # Exit manage models loop
        else: # Should not happen with select
            questionary.print("Invalid choice.", style="fg:red")

def main_menu_interactive():
    # Clear screen for a cleaner start (optional)
    # os.system('cls' if os.name == 'nt' else 'clear') 
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
            # style=custom_style # You can define custom styles for questionary
        ).ask()

        if action == "new_train":
            handle_new_training()
        elif action == "resume_train":
            handle_resume_training()
        elif action == "chat":
            handle_chat()
        elif action == "manage":
            handle_manage_models()
        elif action == "exit" or action is None: # None if Ctrl+C
            questionary.print("Exiting CLUI. Goodbye!", style="fg:green")
            break
        
        # Pause for user to read output before re-displaying menu
        if action != "exit" and action is not None:
            questionary.press_any_key_to_continue("Press any key to return to the main menu...").ask()
            # os.system('cls' if os.name == 'nt' else 'clear') # Optional: clear screen again

if __name__ == "__main__":
    if global_tokenizer is None:
        questionary.print("="*60, style="bold fg:red")
        questionary.print("CRITICAL ERROR: Global tokenizer failed to load.", style="bold fg:red")
        questionary.print(f"Please ensure '{config.TOKENIZER_MODEL_PATH}' exists and is valid.", style="fg:red")
        questionary.print("You might need to run 'python tokenizer_training.py' first.", style="fg:red")
        questionary.print("="*60, style="bold fg:red")
        # Allow to proceed if user wants to manage models, but warn.
        # For now, exit is safer if tokenizer is fundamental.
        if not questionary.confirm("Tokenizer failed to load. Some functionalities will be broken. Continue to model management only?", default=False).ask():
            sys.exit(1)
        # If they choose to continue, they'll only be able to use "Manage Models" effectively.

    main_menu_interactive()