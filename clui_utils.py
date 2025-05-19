# clui_utils.py
import os
import shutil
import glob
import config # To get BASE_MODELS_DIR
import questionary # Import the library

def list_model_runs_choices():
    """Returns a list of choices suitable for questionary from model run directories."""
    if not os.path.exists(config.BASE_MODELS_DIR):
        # questionary.print(f"Models directory '{config.BASE_MODELS_DIR}' does not exist.", style="fg:red")
        return [] # Return empty list, caller should handle
    
    model_dirs = [
        d for d in os.listdir(config.BASE_MODELS_DIR)
        if os.path.isdir(os.path.join(config.BASE_MODELS_DIR, d))
    ]
    
    if not model_dirs:
        # questionary.print("No model runs found.", style="fg:yellow")
        return []

    # Sort by modification time (most recent first) for better user experience
    try:
        model_dirs_with_time = []
        for dir_name in model_dirs:
            dir_path = os.path.join(config.BASE_MODELS_DIR, dir_name)
            mod_time = os.path.getmtime(dir_path)
            model_dirs_with_time.append((dir_name, mod_time))
        
        # Sort by modification time, descending
        model_dirs_with_time.sort(key=lambda item: item[1], reverse=True)
        sorted_model_dirs = [item[0] for item in model_dirs_with_time]
        return sorted_model_dirs
    except Exception as e:
        questionary.print(f"Warning: Could not sort model runs by time: {e}", style="fg:yellow")
        return sorted(model_dirs) # Fallback to alphabetical sort

def select_model_run_interactive(prompt_message="Select a model run:", model_dirs=None):
    """Allows user to select a model run using questionary list prompt."""
    if model_dirs is None:
        model_dirs = list_model_runs_choices()

    if not model_dirs:
        questionary.print("No model runs available to select.", style="fg:yellow")
        return None

    choices = model_dirs + [questionary.Separator(), "Cancel"]
    
    selected_name = questionary.select(
        prompt_message,
        choices=choices,
        use_shortcuts=True
    ).ask()

    if selected_name == "Cancel" or selected_name is None:
        return None
    return os.path.join(config.BASE_MODELS_DIR, selected_name)


def delete_model_run_interactive(model_run_dir_path):
    """Deletes a specified model run directory with confirmation."""
    if not model_run_dir_path or not os.path.exists(model_run_dir_path):
        questionary.print(f"Error: Model run directory '{model_run_dir_path}' not found.", style="fg:red")
        return False
    
    run_basename = os.path.basename(model_run_dir_path)
    confirm = questionary.confirm(
        f"Are you sure you want to PERMANENTLY DELETE the model run '{run_basename}' and its logs?",
        default=False,
        auto_enter=False
    ).ask()

    if confirm:
        try:
            shutil.rmtree(model_run_dir_path)
            tb_log_dir_name = run_basename # Assuming TB log dir has same name as model run dir
            tb_log_path = os.path.join(config.LOG_DIR, tb_log_dir_name)
            if os.path.exists(tb_log_path):
                shutil.rmtree(tb_log_path)
                questionary.print(f"Deleted TensorBoard logs: {tb_log_path}", style="fg:green")
            questionary.print(f"Model run '{run_basename}' deleted successfully.", style="fg:green")
            return True
        except Exception as e:
            questionary.print(f"Error deleting model run: {e}", style="fg:red")
            return False
    else:
        questionary.print("Deletion cancelled.", style="fg:yellow")
        return False

def rename_model_run_interactive(old_model_run_dir_path):
    """Renames a specified model run directory interactively."""
    if not old_model_run_dir_path or not os.path.exists(old_model_run_dir_path):
        questionary.print(f"Error: Model run directory '{old_model_run_dir_path}' not found.", style="fg:red")
        return False
    
    old_basename = os.path.basename(old_model_run_dir_path)
    
    new_basename = questionary.text(
        f"Enter new name for model run '{old_basename}' (press Enter to cancel):",
        validate=lambda text: True if text.strip() else "Name cannot be empty if not cancelling." # Basic validation
    ).ask()

    if not new_basename or not new_basename.strip(): # User pressed Enter or provided empty string
        questionary.print("Rename cancelled.", style="fg:yellow")
        return False
    
    new_basename = new_basename.strip()

    if new_basename == old_basename:
        questionary.print("New name is the same as the old name. No changes made.", style="fg:yellow")
        return False

    new_model_run_dir_path = os.path.join(config.BASE_MODELS_DIR, new_basename)

    if os.path.exists(new_model_run_dir_path):
        questionary.print(f"Error: A directory with the name '{new_basename}' already exists.", style="fg:red")
        return False
    
    try:
        os.rename(old_model_run_dir_path, new_model_run_dir_path)
        old_tb_log_path = os.path.join(config.LOG_DIR, old_basename)
        new_tb_log_path = os.path.join(config.LOG_DIR, new_basename)
        if os.path.exists(old_tb_log_path):
            os.rename(old_tb_log_path, new_tb_log_path)
            questionary.print(f"Renamed TensorBoard logs from '{old_basename}' to '{new_basename}'", style="fg:green")
        questionary.print(f"Model run '{old_basename}' renamed to '{new_basename}' successfully.", style="fg:green")
        return True
    except Exception as e:
        questionary.print(f"Error renaming model run: {e}", style="fg:red")
        if not os.path.exists(old_model_run_dir_path) and os.path.exists(new_model_run_dir_path):
             try: os.rename(new_model_run_dir_path, old_model_run_dir_path) # Try to revert
             except: pass
        return False