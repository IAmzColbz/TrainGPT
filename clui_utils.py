# --- START OF FILE clui_utils.py ---

# clui_utils.py
import os
import shutil
import glob
import config # To get BASE_MODELS_DIR
import questionary # Import the library
import logging

logger = logging.getLogger(__name__)

def list_model_runs_choices():
    """Returns a list of choices suitable for questionary from model run directories."""
    if not os.path.exists(config.BASE_MODELS_DIR):
        # questionary.print is for direct user feedback; here, return for caller
        logger.warning(f"Models directory '{config.BASE_MODELS_DIR}' does not exist.")
        return [] 
    
    try:
        model_dirs = [
            d for d in os.listdir(config.BASE_MODELS_DIR)
            if os.path.isdir(os.path.join(config.BASE_MODELS_DIR, d))
        ]
    except OSError as e:
        logger.error(f"Error listing model directories in '{config.BASE_MODELS_DIR}': {e}")
        return [] # Cannot list directories
    
    if not model_dirs:
        logger.info("No model runs found in BASE_MODELS_DIR.")
        return []

    # Sort by modification time (most recent first) for better user experience
    try:
        model_dirs_with_time = []
        for dir_name in model_dirs:
            dir_path = os.path.join(config.BASE_MODELS_DIR, dir_name)
            mod_time = os.path.getmtime(dir_path)
            model_dirs_with_time.append((dir_name, mod_time))
        
        model_dirs_with_time.sort(key=lambda item: item[1], reverse=True)
        sorted_model_dirs = [item[0] for item in model_dirs_with_time]
        return sorted_model_dirs
    except Exception as e_sort: # Catch any error during stat or sort
        logger.warning(f"Could not sort model runs by time: {e_sort}. Falling back to alphabetical sort.")
        return sorted(model_dirs) 

def select_model_run_interactive(prompt_message="Select a model run:", model_dirs=None):
    """Allows user to select a model run using questionary list prompt."""
    if model_dirs is None: # If not provided, fetch them
        model_dirs = list_model_runs_choices()

    if not model_dirs:
        questionary.print("No model runs available to select.", style="fg:yellow")
        return None

    choices = model_dirs + [questionary.Separator(), "Cancel"]
    
    selected_name = questionary.select(
        prompt_message,
        choices=choices,
        use_shortcuts=True # Allows typing to filter
    ).ask() # Returns the selected string or None if Escaped

    if selected_name == "Cancel" or selected_name is None:
        return None # User cancelled or escaped
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
        auto_enter=False # Require explicit y/n
    ).ask()

    if confirm:
        try:
            shutil.rmtree(model_run_dir_path)
            logger.info(f"Deleted model run directory: {model_run_dir_path}")
            
            # Attempt to delete corresponding TensorBoard log directory
            tb_log_dir_name = run_basename # Assuming TB log dir has same name as model run dir
            tb_log_path = os.path.join(config.LOG_DIR, tb_log_dir_name)
            if os.path.exists(tb_log_path):
                shutil.rmtree(tb_log_path)
                questionary.print(f"Deleted TensorBoard logs: {tb_log_path}", style="fg:green")
                logger.info(f"Deleted TensorBoard logs: {tb_log_path}")
            else:
                logger.info(f"No corresponding TensorBoard log directory found at {tb_log_path} to delete.")

            questionary.print(f"Model run '{run_basename}' deleted successfully.", style="fg:green")
            return True
        except Exception as e:
            questionary.print(f"Error deleting model run '{run_basename}': {e}", style="fg:red")
            logger.error(f"Error deleting model run '{run_basename}': {e}")
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
        f"Enter new name for model run '{old_basename}' (press Enter or Esc to cancel):",
        validate=lambda text: True if text.strip() else "Name cannot be empty if not cancelling." 
                                                        # This validation fires even on empty if user intends to cancel.
                                                        # Better to handle empty as cancel.
    ).ask()

    if not new_basename or not new_basename.strip(): # User pressed Enter, Esc, or provided empty string
        questionary.print("Rename cancelled.", style="fg:yellow")
        return False
    
    new_basename = new_basename.strip() # Clean up whitespace

    if new_basename == old_basename:
        questionary.print("New name is the same as the old name. No changes made.", style="fg:yellow")
        return False

    new_model_run_dir_path = os.path.join(config.BASE_MODELS_DIR, new_basename)

    if os.path.exists(new_model_run_dir_path):
        questionary.print(f"Error: A directory with the name '{new_basename}' already exists at '{config.BASE_MODELS_DIR}'.", style="fg:red")
        return False
    
    try:
        # Rename model directory
        os.rename(old_model_run_dir_path, new_model_run_dir_path)
        logger.info(f"Renamed model run directory from '{old_model_run_dir_path}' to '{new_model_run_dir_path}'")

        # Rename corresponding TensorBoard log directory
        old_tb_log_path = os.path.join(config.LOG_DIR, old_basename)
        new_tb_log_path = os.path.join(config.LOG_DIR, new_basename)
        if os.path.exists(old_tb_log_path):
            os.rename(old_tb_log_path, new_tb_log_path)
            questionary.print(f"Renamed TensorBoard logs from '{old_basename}' to '{new_basename}'", style="fg:green")
            logger.info(f"Renamed TensorBoard logs from '{old_tb_log_path}' to '{new_tb_log_path}'")
        else:
            logger.info(f"No corresponding TensorBoard log directory found at {old_tb_log_path} to rename.")

        questionary.print(f"Model run '{old_basename}' renamed to '{new_basename}' successfully.", style="fg:green")
        return True
    except Exception as e:
        questionary.print(f"Error renaming model run: {e}", style="fg:red")
        logger.error(f"Error renaming model run '{old_basename}' to '{new_basename}': {e}")
        # Attempt to revert rename of model directory if TensorBoard rename failed or other issue
        if not os.path.exists(old_model_run_dir_path) and os.path.exists(new_model_run_dir_path):
             try: 
                 os.rename(new_model_run_dir_path, old_model_run_dir_path) 
                 logger.info(f"Attempted to revert rename of model directory to '{old_model_run_dir_path}' due to error.")
             except Exception as e_revert:
                 logger.error(f"Failed to revert model directory rename: {e_revert}")
        return False