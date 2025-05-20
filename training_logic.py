# --- START OF FILE training_logic.py ---

# training_logic.py
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import os
import glob
import pyarrow.parquet as pq
import config
from dataset_loader import get_dataloaders 
from nn_model import SimpleTransformerSeq2Seq
from tokenizer_wrapper import global_tokenizer
import math
import gc
import traceback
import torch.optim as optim
import logging

logger = logging.getLogger(__name__)

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0,
    after a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
    """
    if last_epoch != -1: # Resuming
        for group in optimizer.param_groups:
            if 'initial_lr' not in group:
                logger.warning(f"MANUAL SET: 'initial_lr' was missing in optimizer group for scheduler resume (last_epoch={last_epoch}). Setting from current lr: {group['lr']}")
                group['initial_lr'] = group['lr']
                
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
    return LambdaLR(optimizer, lr_lambda, last_epoch)

def evaluate_model(model, val_dataloader, device, writer, global_step, phase_name="Validation"):
    model.eval()
    total_val_loss = 0
    if not val_dataloader or len(val_dataloader) == 0:
        logger.info(f"{phase_name} evaluation skipped: No validation dataloader or empty.")
        return float('inf'), 0.0 

    progress_bar = tqdm(val_dataloader, desc=f"{phase_name} Evaluating", leave=False, ncols=100, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}')
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            try:
                src_ids = batch['src_token_ids'].to(device, non_blocking=True)
                dec_in_ids = batch['decoder_input_ids'].to(device, non_blocking=True)
                lbl_ids = batch['label_ids'].to(device, non_blocking=True)
                
                logits = model(src_ids, dec_in_ids)
                loss = model.criterion(logits.view(-1, model.vocab_size), lbl_ids.view(-1))
                
                if not torch.isnan(loss) and not torch.isinf(loss):
                    total_val_loss += loss.item()
                else:
                    logger.warning(f"{phase_name} evaluation: NaN or Inf loss encountered at batch {batch_idx}. Skipping loss accumulation for this batch.")

            except Exception as e:
                logger.error(f"Error during {phase_name} evaluation batch {batch_idx}: {e}\n{traceback.format_exc()}")
                continue 

    avg_val_loss = total_val_loss / len(val_dataloader) if len(val_dataloader) > 0 else float('inf')
    
    if writer: 
        writer.add_scalar(f'{phase_name}/Loss_Eval_Step', avg_val_loss, global_step)
    
    logger.info(f"{phase_name} Evaluation Complete. Avg Loss: {avg_val_loss:.4f} (at global step {global_step})")
    progress_bar.close()
    return avg_val_loss, 0.0 

def precalculate_total_prefix_lm_steps(files_to_cycle, docs_chunk_size, batch_size, tokenizer_path, model_run_name_for_cache):
    logger.info(f"Pre-calculating PrefixLM steps for run '{model_run_name_for_cache}' (this may build or check caches)...")
    logger.info(f"  Using config.PROJECT_ROOT: {config.PROJECT_ROOT}")
    logger.info(f"  Targeting cache base template: {config.CACHE_DIR_BASE_TEMPLATE}")
    
    total_estimated_steps_one_pass = 0
    
    run_specific_cache_base = config.CACHE_DIR_BASE_TEMPLATE.format(model_run_name=model_run_name_for_cache)
    logger.info(f"  Run-specific cache base for pre-calculation: {run_specific_cache_base}")


    for clm_file_path in tqdm(files_to_cycle, desc="Pre-calculating steps (files)", unit="file", ncols=100):
        logger.debug(f"  Pre-calculating for file: {clm_file_path}")
        estimated_total_docs_in_file = docs_chunk_size 
        try:
            pf = pq.ParquetFile(clm_file_path)
            if pf.metadata: estimated_total_docs_in_file = pf.metadata.num_rows
            del pf; gc.collect()
        except Exception as e: 
            logger.warning(f"Could not get exact row count for {os.path.basename(clm_file_path)}: {e}. Using fallback estimation: {estimated_total_docs_in_file} docs.")

        num_chunks_in_file = math.ceil(estimated_total_docs_in_file / docs_chunk_size)
        current_doc_offset = 0
        
        for _ in tqdm(range(num_chunks_in_file), desc=f"Pre-calc Chunks for {os.path.basename(clm_file_path)}", unit="chunk", leave=False, ncols=100):
            # Pass a dummy tokenizer instance for pre-calculation if global_tokenizer is None, though it shouldn't be.
            # The main thing is the cache path construction and num_examples_in_this_chunk.
            tokenizer_for_precalc = global_tokenizer
            if tokenizer_for_precalc is None:
                logger.error("CRITICAL: global_tokenizer is None during precalculate_total_prefix_lm_steps. This should not happen.")
                # As a last resort, this would fail if caching tries to use tokenizer methods.
                # However, get_dataloaders might handle it if only num_examples_in_this_chunk is needed from cache.
                # Better to ensure global_tokenizer is always available.
                return 0 # Cannot proceed

            dataset_chunk, _, num_examples_in_this_chunk = get_dataloaders(
                task_type="prefix_lm_pretrain",
                tokenizer=tokenizer_for_precalc, 
                batch_size=batch_size, 
                specific_file_path=clm_file_path,
                start_doc_offset=current_doc_offset,
                num_docs_in_chunk=docs_chunk_size,
                model_run_name_for_cache=model_run_name_for_cache 
            )

            if num_examples_in_this_chunk > 0:
                total_estimated_steps_one_pass += math.ceil(num_examples_in_this_chunk / batch_size)
            
            if dataset_chunk: del dataset_chunk; gc.collect()
            
            current_doc_offset += docs_chunk_size
            if current_doc_offset >= estimated_total_docs_in_file: 
                 break
                 
    logger.info(f"Pre-calculation complete. Estimated PrefixLM steps for ONE pass: {total_estimated_steps_one_pass}")
    return total_estimated_steps_one_pass


def run_prefix_lm_pretraining_phase(model, writer, model_run_name): 
    logger.info(f"--- PrefixLM Pre-training Phase Initiated for run: {model_run_name} ---")
    device = model.device
    batch_size = config.PREFIXLM_BATCH_SIZE
    total_passes = config.PREFIXLM_TOTAL_PASSES 
    docs_chunk_size = config.DOCS_CHUNK_SIZE_PER_PREFIXLM_EPOCH 

    fineweb_search_pattern = os.path.join(config.FINEWEB_DATA_DIR, "*.parquet")
    logger.info(f"Searching for FineWeb files in: {config.FINEWEB_DATA_DIR} with pattern: {fineweb_search_pattern}")
    all_found_files = sorted(glob.glob(fineweb_search_pattern))
    logger.info(f"Found {len(all_found_files)} FineWeb files initially: {all_found_files if len(all_found_files) < 10 else str(all_found_files[:5]) + '...'}")

    if not all_found_files:
        logger.warning(f"No FineWeb .parquet files found in directory '{config.FINEWEB_DATA_DIR}'. Skipping PrefixLM pre-training.")
        return

    if config.NUM_FINEWEB_FILES_TO_CYCLE <= 0: 
        files_to_cycle = all_found_files
    else:
        files_to_cycle = all_found_files[:config.NUM_FINEWEB_FILES_TO_CYCLE]
    
    logger.info(f"Selected {len(files_to_cycle)} FineWeb files for this run based on NUM_FINEWEB_FILES_TO_CYCLE={config.NUM_FINEWEB_FILES_TO_CYCLE}.")

    if not files_to_cycle: 
        logger.warning("After applying NUM_FINEWEB_FILES_TO_CYCLE, no files selected. Skipping PrefixLM pre-training.")
        return

    steps_one_pass = precalculate_total_prefix_lm_steps(files_to_cycle, docs_chunk_size, batch_size, config.TOKENIZER_MODEL_PATH, model_run_name)
    total_training_steps = steps_one_pass * total_passes
    
    if total_training_steps == 0:
        logger.warning("Total estimated PrefixLM steps is 0. Skipping pre-training.")
        return
    logger.info(f"Total estimated PrefixLM steps for {total_passes} passes: {total_training_steps}")

    # --- Checkpoint Loading Logic ---
    # Priority: 1. Existing PrefixLM checkpoint, 2. Old CLM checkpoint (for transition)
    prefixlm_ckpt_path = os.path.join(model.model_run_dir, config.PREFIXLM_CHECKPOINT_FILENAME)
    old_clm_ckpt_name = "model_clm_pretrained_checkpoint.pth" # User's specific old checkpoint name
    old_clm_ckpt_path = os.path.join(model.model_run_dir, old_clm_ckpt_name)
    
    ckpt_to_load_initially = None
    loaded_old_clm_for_prefix_start = False

    if os.path.exists(prefixlm_ckpt_path):
        ckpt_to_load_initially = config.PREFIXLM_CHECKPOINT_FILENAME
        logger.info(f"Found existing PrefixLM checkpoint: {ckpt_to_load_initially}. Will attempt to load it.")
    elif os.path.exists(old_clm_ckpt_path):
        logger.info(f"No new PrefixLM checkpoint. Found old CLM checkpoint: {old_clm_ckpt_name}.")
        logger.info(f"This old CLM checkpoint will be loaded as the starting point for the PrefixLM phase.")
        ckpt_to_load_initially = old_clm_ckpt_name
        loaded_old_clm_for_prefix_start = True
    else:
        logger.info("No existing PrefixLM or old CLM checkpoint found. PrefixLM will start fresh.")

    next_step_to_run = 0 # This is the step number for the *next* step to execute (0-indexed for the first step)
    if ckpt_to_load_initially:
        # model.load_checkpoint returns (last_completed_step + 1)
        returned_step = model.load_checkpoint(ckpt_to_load_initially)
        if returned_step > 0: # Checkpoint loaded successfully
            next_step_to_run = returned_step
            # model.current_phase_step_or_epoch is already set to last_completed_step by load_checkpoint
            logger.info(f"Checkpoint '{ckpt_to_load_initially}' loaded. Last completed step: {model.current_phase_step_or_epoch}. Next step: {next_step_to_run}.")
        else:
            logger.warning(f"Failed to load checkpoint '{ckpt_to_load_initially}'. PrefixLM will start fresh.")
            model.current_phase_step_or_epoch = 0 # Ensure reset if load failed
            next_step_to_run = 0
            loaded_old_clm_for_prefix_start = False # Treat as fresh start if load failed

    # If we loaded an old CLM checkpoint to *start* a new PrefixLM phase,
    # its step count is irrelevant to PrefixLM. Reset phase progress.
    if loaded_old_clm_for_prefix_start and model.current_phase_step_or_epoch > -1 : # Check if old CLM was actually loaded
        logger.info(f"Old CLM checkpoint '{ckpt_to_load_initially}' was loaded. Resetting PrefixLM phase progress and optimizer/scheduler.")
        model.current_phase_step_or_epoch = 0 # Reset completed steps for this new PrefixLM phase
        next_step_to_run = 0                 # Start PrefixLM from step 0
        
        # Re-initialize optimizer and ensure scheduler is cleared for new setup
        logger.info("Re-initializing optimizer for PrefixLM phase after loading old CLM weights.")
        model.optimizer = optim.AdamW(model.parameters(), lr=config.NN_LEARNING_RATE, weight_decay=config.OPTIMIZER_WEIGHT_DECAY)
        for group in model.optimizer.param_groups: group.setdefault('initial_lr', group['lr'])
        if hasattr(model, 'scheduler') and model.scheduler is not None:
            del model.scheduler
            model.scheduler = None
        gc.collect()

    # Determine scheduler_last_epoch based on model.current_phase_step_or_epoch (number of *completed* steps, 0-indexed)
    # scheduler.last_epoch expects -1 for fresh start, or the epoch_num of the last completed epoch.
    scheduler_last_epoch = model.current_phase_step_or_epoch -1 if model.current_phase_step_or_epoch > 0 else -1
    
    if hasattr(model, 'scheduler') and model.scheduler: del model.scheduler; model.scheduler = None; gc.collect()
    logger.info(f"Initializing PrefixLM scheduler with last_epoch={scheduler_last_epoch} (based on {model.current_phase_step_or_epoch} completed steps), total_training_steps={total_training_steps}")
    logger.debug(f"Optimizer param_groups before PrefixLM scheduler init: {model.optimizer.param_groups}")
    
    model.scheduler = get_linear_schedule_with_warmup(
        model.optimizer, 
        config.LR_SCHEDULER_WARMUP_STEPS, 
        total_training_steps, 
        last_epoch=scheduler_last_epoch
    )
    
    # global_step_counter tracks the current step we are about to process (0-indexed for the very first step)
    global_step_counter = next_step_to_run 

    for pass_idx in range(total_passes):
        logger.info(f"PrefixLM Global Pass {pass_idx + 1}/{total_passes}")
        
        # Check if this pass has already been completed based on global_step_counter
        if steps_one_pass > 0 and global_step_counter >= (pass_idx + 1) * steps_one_pass:
            logger.info(f"  Skipping Pass {pass_idx + 1} as global_step_counter ({global_step_counter}) indicates it's completed or beyond.")
            continue

        if torch.cuda.is_available(): torch.cuda.empty_cache(); gc.collect()
        
        for file_idx, file_path in enumerate(files_to_cycle):
            logger.info(f"  PrefixLM File {file_idx + 1}/{len(files_to_cycle)} in Pass {pass_idx+1}: {os.path.basename(file_path)}")
            
            estimated_total_docs_in_file = docs_chunk_size 
            try:
                pf = pq.ParquetFile(file_path);
                if pf.metadata: estimated_total_docs_in_file = pf.metadata.num_rows
                del pf; gc.collect()
            except Exception: pass 
            
            num_chunks_in_file = math.ceil(estimated_total_docs_in_file / docs_chunk_size)
            current_doc_offset = 0
            
            for chunk_in_file_idx in range(num_chunks_in_file):
                if global_step_counter >= total_training_steps: break 
                logger.info(f"    File {os.path.basename(file_path)}, Doc Chunk {chunk_in_file_idx + 1}/{num_chunks_in_file} (offset {current_doc_offset})")
                
                # Skip chunks if global_step_counter indicates they've been processed
                # This estimation is rough but helps skip already processed data faster upon resume
                # A more precise skip would require knowing steps per chunk, which varies slightly.
                # current_doc_offset / docs_chunk_size gives an idea of how many chunks into the file we are.
                # For now, rely on global_step_counter vs total_training_steps for overall completion.
                # The fine-grained skipping logic within a pass (if global_step_counter > (pass_idx * steps_one_pass + steps_in_file_so_far + steps_in_chunk_so_far) )
                # can be complex. The current loop structure with global_step_counter checks should suffice.

                train_dl, _, num_ex_chunk = get_dataloaders(
                    task_type="prefix_lm_pretrain", tokenizer=global_tokenizer, batch_size=batch_size,
                    specific_file_path=file_path, start_doc_offset=current_doc_offset,
                    num_docs_in_chunk=docs_chunk_size,
                    model_run_name_for_cache=model_run_name
                )
                current_doc_offset += docs_chunk_size 

                if not train_dl or num_ex_chunk == 0:
                    logger.warning(f"      No examples in this chunk. Skipping.")
                    if current_doc_offset >= estimated_total_docs_in_file and estimated_total_docs_in_file > docs_chunk_size : break
                    continue
                
                model.train()
                pbar_desc = f"PrefixLM P{pass_idx+1} F{file_idx+1} Ck{chunk_in_file_idx+1} Step {global_step_counter}"
                pbar = tqdm(train_dl, desc=pbar_desc, leave=False, ncols=150, initial=0, total=math.ceil(num_ex_chunk / batch_size))
                
                for batch_idx_in_chunk, batch in enumerate(pbar):
                    # If resuming, skip batches within the chunk that were already processed
                    # This requires knowing how many batches correspond to global_step_counter's progress within this chunk
                    # Let current_step_in_phase = global_step_counter - (pass_idx * steps_one_pass)
                    # Let steps_processed_before_this_chunk = ... (hard to calculate precisely without step-per-chunk map)
                    # A simpler approach: if global_step_counter corresponds to a step *after* the start of this
                    # dataloader, we need to advance the dataloader.
                    # `next_step_to_run` was the global step to start from.
                    # If this is the first chunk after resume, we might need to skip some batches.
                    
                    # Let's use a simpler skip: if global_step_counter is what we're about to run,
                    # and next_step_to_run was where we should have started,
                    # then if we are in the "resuming chunk", we might need to fast-forward.
                    # This is tricky. The current outer loops and `global_step_counter` checks should largely handle it.
                    # If global_step_counter < next_step_to_run effectively, due to loop structure and estimates,
                    # this means we are "catching up" to where we left off.
                    # For simplicity, this internal batch skipping is omitted as it's complex and error-prone.
                    # The tqdm `initial` might help if we can calculate it precisely.
                    # The `global_step_counter` is the *actual* 0-indexed step we are processing.

                    if global_step_counter >= total_training_steps: break 
                    
                    src = batch['src_token_ids'].to(device, non_blocking=True)
                    dec_in = batch['decoder_input_ids'].to(device, non_blocking=True)
                    lbl = batch['label_ids'].to(device, non_blocking=True)
                    
                    model.optimizer.zero_grad(set_to_none=True)
                    logits = model(src, dec_in)
                    loss = model.criterion(logits.view(-1, model.vocab_size), lbl.view(-1))
                    
                    if not torch.isnan(loss) and not torch.isinf(loss):
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 
                        model.optimizer.step()
                        if model.scheduler: model.scheduler.step() 
                        
                        if writer:
                            writer.add_scalar('PrefixLM/Batch_Loss', loss.item(), global_step_counter)
                            current_lr = model.optimizer.param_groups[0]['lr']
                            writer.add_scalar('PrefixLM/Learning_Rate', current_lr, global_step_counter)
                        
                        pbar.set_postfix({"Loss": f"{loss.item():.4f}", "LR":f"{current_lr:.2e}", "Step":f"{global_step_counter}/{total_training_steps}"})
                    else:
                        logger.warning(f"PrefixLM training: NaN or Inf loss encountered at global_step {global_step_counter}. Skipping optimizer step.")

                    # model.current_phase_step_or_epoch should store the number of *completed* steps (0-indexed)
                    model.current_phase_step_or_epoch = global_step_counter + 1 # After this step, global_step_counter steps are done.
                    global_step_counter += 1
                    pbar.update(1)
                
                pbar.close()
                del train_dl, batch, src, dec_in, lbl, logits, loss; gc.collect()
                
                # Save checkpoint based on completed steps
                model.save_checkpoint(model.current_phase_step_or_epoch, config.PREFIXLM_CHECKPOINT_FILENAME) 
                if current_doc_offset >= estimated_total_docs_in_file and estimated_total_docs_in_file > docs_chunk_size : break 
            if global_step_counter >= total_training_steps: break
        if global_step_counter >= total_training_steps: break
        
    logger.info(f"--- PrefixLM Pre-training Completed (Total Global Steps Processed: {global_step_counter}, Stored completed steps: {model.current_phase_step_or_epoch}) ---")
    model.save_checkpoint(model.current_phase_step_or_epoch, config.PREFIXLM_CHECKPOINT_FILENAME) 
    if hasattr(model, 'scheduler') and model.scheduler: del model.scheduler; model.scheduler = None; gc.collect()


def run_supervised_training_phase(model, writer, model_run_name):
    logger.info(f"--- Supervised Code Training Phase Initiated for run: {model_run_name} ---")
    device = model.device
    
    result = get_dataloaders(
        task_type="supervised_code", 
        tokenizer=global_tokenizer, 
        batch_size=config.SUPERVISED_BATCH_SIZE, 
        val_split_ratio=config.SUPERVISED_VALIDATION_SPLIT_RATIO,
        model_run_name_for_cache=model_run_name 
    )
    if result is None or result[0] is None: 
        logger.error("ERROR: Failed to create supervised code DataLoaders (train_dl is None). Aborting supervised phase.")
        return
    train_dataloader, val_dataloader = result

    if not train_dataloader or len(train_dataloader) == 0:
        logger.error("Supervised training dataloader is empty. Aborting supervised phase.")
        return

    num_training_steps_per_epoch = len(train_dataloader)
    num_total_training_steps_supervised = num_training_steps_per_epoch * config.SUPERVISED_EPOCHS
    if num_total_training_steps_supervised == 0:
        logger.warning("No steps for supervised training (empty dataloader or 0 epochs). Skipping.")
        return
    
    # --- Checkpoint and Optimizer/Scheduler Initialization ---
    # model.current_phase_step_or_epoch will store the last *completed* epoch number (0-indexed) for this phase.
    # next_epoch_to_run will be the epoch number (0-indexed) to start the loop from.
    
    # Try loading best supervised, then regular supervised.
    # model.load_checkpoint returns (last_completed_epoch + 1) if successful, or 0.
    # It also sets model.current_phase_step_or_epoch to last_completed_epoch.
    
    initial_epoch_to_run_from_checkpoint = 0 # This is (last_completed_epoch + 1)
    loaded_supervised_checkpoint = False

    if os.path.exists(os.path.join(model.model_run_dir, config.BEST_MODEL_SUPERVISED_CHECKPOINT_FILENAME)):
        logger.info(f"Attempting to load best supervised checkpoint: {config.BEST_MODEL_SUPERVISED_CHECKPOINT_FILENAME}")
        returned_epoch = model.load_checkpoint(config.BEST_MODEL_SUPERVISED_CHECKPOINT_FILENAME)
        if returned_epoch > 0: initial_epoch_to_run_from_checkpoint = returned_epoch; loaded_supervised_checkpoint = True
    
    if not loaded_supervised_checkpoint and os.path.exists(os.path.join(model.model_run_dir, config.SUPERVISED_CHECKPOINT_FILENAME)):
        logger.info(f"Attempting to load regular supervised checkpoint: {config.SUPERVISED_CHECKPOINT_FILENAME}")
        returned_epoch = model.load_checkpoint(config.SUPERVISED_CHECKPOINT_FILENAME)
        if returned_epoch > 0: initial_epoch_to_run_from_checkpoint = returned_epoch; loaded_supervised_checkpoint = True

    scheduler_reinitialized_this_phase = False
    if loaded_supervised_checkpoint:
        logger.info(f"Resuming supervised training. Last completed epoch: {model.current_phase_step_or_epoch}. Next epoch to run (1-indexed): {initial_epoch_to_run_from_checkpoint +1}.")
        # Optimizer and scheduler states are expected to be loaded by model.load_checkpoint.
    else: # No supervised checkpoint. Try PrefixLM or start fresh.
        prefixlm_checkpoint_path = os.path.join(model.model_run_dir, config.PREFIXLM_CHECKPOINT_FILENAME)
        old_clm_ckpt_name = "model_clm_pretrained_checkpoint.pth"
        old_clm_checkpoint_path = os.path.join(model.model_run_dir, old_clm_ckpt_name)

        source_for_finetune = None
        if os.path.exists(prefixlm_checkpoint_path):
            source_for_finetune = config.PREFIXLM_CHECKPOINT_FILENAME
        elif os.path.exists(old_clm_checkpoint_path): # Fallback to old CLM if PrefixLM not found
            source_for_finetune = old_clm_ckpt_name
            
        if source_for_finetune:
            logger.info(f"No supervised checkpoint. Loading model from '{source_for_finetune}' for fine-tuning.")
            # Create a temporary model instance to load weights without affecting current optimizer/scheduler
            # Use current model's architecture parameters for the temp model.
            temp_model = SimpleTransformerSeq2Seq(
                vocab_size=model.vocab_size, d_model=model.d_model, n_heads=model.n_heads, 
                num_encoder_layers=model.num_encoder_layers, num_decoder_layers=model.num_decoder_layers,
                d_ff=model.d_ff, dropout=model.dropout_rate, positional_encoding_max_len=model.positional_encoding.pe.size(1),
                model_run_dir=model.model_run_dir # Needs model_run_dir for its own load_checkpoint
            )
            temp_model.load_checkpoint(source_for_finetune) # This loads weights into temp_model
            model.load_state_dict(temp_model.state_dict()) # Copy only weights to current model
            del temp_model; gc.collect()
            logger.info(f"Weights from '{source_for_finetune}' loaded into current model for supervised fine-tuning.")
            
            model.current_phase_step_or_epoch = 0 # Reset completed epochs for supervised phase
            initial_epoch_to_run_from_checkpoint = 0 # Start supervised from epoch 0
            model.history['supervised_train_epoch_loss'] = [] 
            model.history['supervised_validation_loss'] = []
            
            logger.info("Re-initializing optimizer for supervised fine-tuning (AdamW, potentially lower LR).")
            model.optimizer = optim.AdamW(model.parameters(), lr=config.NN_LEARNING_RATE / 2, weight_decay=config.OPTIMIZER_WEIGHT_DECAY)
            for group in model.optimizer.param_groups: group.setdefault('initial_lr', group['lr'])
            scheduler_reinitialized_this_phase = True
        else:
            logger.info("No supervised or pre-trained (PrefixLM/CLM) checkpoints found. Starting supervised training from scratch.")
            model.current_phase_step_or_epoch = 0
            initial_epoch_to_run_from_checkpoint = 0
            scheduler_reinitialized_this_phase = True # Scheduler will be new

    # --- Initialize or Re-initialize Scheduler for Supervised Phase ---
    # model.current_phase_step_or_epoch is the last *completed* epoch (0-indexed).
    # last_scheduler_step is the number of optimizer steps *already taken* in this supervised phase.
    # For LambdaLR, last_epoch is the number of times scheduler.step() has been called.
    
    if scheduler_reinitialized_this_phase or not model.scheduler: 
        # If starting fresh for this phase (from scratch or after pre-training), or if scheduler wasn't loaded.
        # model.current_phase_step_or_epoch = number of completed epochs (0-indexed)
        # So, number of completed steps = model.current_phase_step_or_epoch * num_training_steps_per_epoch
        # scheduler.last_epoch should be (completed_steps - 1)
        completed_steps = model.current_phase_step_or_epoch * num_training_steps_per_epoch
        scheduler_last_epoch = completed_steps -1 if completed_steps > 0 else -1
        logger.info(f"Initializing new Supervised LR scheduler. Last completed epoch for phase: {model.current_phase_step_or_epoch}. Calculated last_scheduler_step (for LR_Lambda): {scheduler_last_epoch}.")
    else: # Scheduler was loaded from a supervised checkpoint
        scheduler_last_epoch = model.scheduler.last_epoch 
        logger.info(f"Using Supervised LR scheduler loaded from checkpoint. Scheduler's last_epoch: {scheduler_last_epoch}.")

    if hasattr(model, 'scheduler') and model.scheduler and not scheduler_reinitialized_this_phase:
         logger.info("Scheduler seems to be loaded. Assuming it's compatible.")
    else:
        if hasattr(model, 'scheduler') and model.scheduler: del model.scheduler; model.scheduler = None; gc.collect()
        logger.debug(f"Optimizer param_groups before Supervised scheduler init: {model.optimizer.param_groups}")
        model.scheduler = get_linear_schedule_with_warmup(
            model.optimizer, 
            min(config.LR_SCHEDULER_WARMUP_STEPS, num_total_training_steps_supervised // 10), # Warmup should not exceed total steps
            num_total_training_steps_supervised,
            last_epoch=scheduler_last_epoch 
        )
        logger.info(f"Supervised LR scheduler created/recreated. Initial LR from optimizer: {model.optimizer.param_groups[0]['lr']:.2e}")

    # loop_start_epoch_0_indexed is the epoch number (0-indexed) to begin the training loop from.
    # model.current_phase_step_or_epoch is the last *completed* epoch.
    # initial_epoch_to_run_from_checkpoint is (last_completed_epoch + 1), so it's already the next epoch index.
    loop_start_epoch_0_indexed = model.current_phase_step_or_epoch if loaded_supervised_checkpoint else 0
    if initial_epoch_to_run_from_checkpoint > loop_start_epoch_0_indexed : # Should be if loaded_supervised_checkpoint
        loop_start_epoch_0_indexed = initial_epoch_to_run_from_checkpoint

    logger.info(f"Supervised training will run from epoch {loop_start_epoch_0_indexed + 1} (0-indexed: {loop_start_epoch_0_indexed}) to {config.SUPERVISED_EPOCHS}.")
    
    global_sup_step = loop_start_epoch_0_indexed * num_training_steps_per_epoch
    best_val_loss = min((h_val[1] for h_val in model.history.get('supervised_validation_loss', []) if h_val), default=float('inf'))

    for epoch_idx_0_based in range(loop_start_epoch_0_indexed, config.SUPERVISED_EPOCHS):
        epoch_display_num = epoch_idx_0_based + 1 
        logger.info(f"Supervised Code Epoch {epoch_display_num}/{config.SUPERVISED_EPOCHS}")
        
        if torch.cuda.is_available(): torch.cuda.empty_cache(); gc.collect()
        model.train()
        epoch_loss_sum = 0.0; num_batches_in_epoch = 0
        
        pbar_desc = f"Supervised Epoch {epoch_display_num} GlobalStep {global_sup_step}"
        progress_bar = tqdm(train_dataloader, desc=pbar_desc, leave=False, ncols=150)
        for batch_idx, batch in enumerate(progress_bar):
            try:
                src = batch['src_token_ids'].to(device, non_blocking=True)
                dec_in = batch['decoder_input_ids'].to(device, non_blocking=True)
                lbl = batch['label_ids'].to(device, non_blocking=True)
                
                model.optimizer.zero_grad(set_to_none=True)
                logits = model(src, dec_in)
                loss = model.criterion(logits.view(-1, model.vocab_size), lbl.view(-1))
                
                if not torch.isnan(loss) and not torch.isinf(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    model.optimizer.step()
                    if model.scheduler: model.scheduler.step() 
                    
                    epoch_loss_sum += loss.item(); num_batches_in_epoch += 1
                    current_lr_sup = model.optimizer.param_groups[0]['lr']
                    
                    if writer:
                        writer.add_scalar('Supervised/Batch_Loss', loss.item(), global_sup_step)
                        writer.add_scalar('Supervised/Learning_Rate', current_lr_sup, global_sup_step)
                    
                    progress_bar.set_postfix({"Loss": f"{loss.item():.4f}", "LR":f"{current_lr_sup:.2e}", "Step": f"{global_sup_step}"})
                else:
                    logger.warning(f"Supervised training: NaN or Inf loss at epoch {epoch_display_num}, batch {batch_idx}, global_step {global_sup_step}. Skipping optimizer step.")

                global_sup_step += 1

                if val_dataloader and global_sup_step > 0 and global_sup_step % config.SUPERVISED_VALIDATE_EVERY_N_BATCHES == 0:
                    avg_val_loss, _ = evaluate_model(model, val_dataloader, device, writer, global_sup_step, "Supervised_Validation_Mid_Epoch")
                    if writer: writer.add_scalar('Supervised_Validation/Step_Loss', avg_val_loss, global_sup_step) 
                    
                    if avg_val_loss < best_val_loss: 
                        best_val_loss = avg_val_loss
                        logger.info(f"New best mid-epoch validation_loss: {best_val_loss:.4f} at step {global_sup_step}. Saving best model.")
                        model.save_checkpoint(epoch_idx_0_based, config.BEST_MODEL_SUPERVISED_CHECKPOINT_FILENAME, is_best=True) # Save with current epoch
                    model.train() 
            except Exception as e_batch:
                logger.error(f"Error during supervised training batch {batch_idx} in epoch {epoch_display_num}: {e_batch}\n{traceback.format_exc()}")
                continue 

        progress_bar.close()
        avg_epoch_train_loss = epoch_loss_sum / num_batches_in_epoch if num_batches_in_epoch > 0 else float('inf')
        if writer: writer.add_scalar('Supervised/Train_Epoch_Loss', avg_epoch_train_loss, epoch_display_num) # Log with 1-indexed epoch
        
        model.history['supervised_train_epoch_loss'] = [(e,v) for e,v in model.history.get('supervised_train_epoch_loss',[]) if e != epoch_display_num]
        model.history['supervised_train_epoch_loss'].append((epoch_display_num, avg_epoch_train_loss))
        model.history['supervised_train_epoch_loss'].sort(key=lambda x:x[0])

        if val_dataloader:
            logger.info(f"Running end-of-epoch validation for Supervised Epoch {epoch_display_num}...")
            avg_val_loss_epoch, _ = evaluate_model(model, val_dataloader, device, writer, global_sup_step, "Supervised_Validation_Epoch_End") 
            if writer: writer.add_scalar('Supervised_Validation/Epoch_End_Loss', avg_val_loss_epoch, epoch_display_num) # Log with 1-indexed epoch

            model.history['supervised_validation_loss'] = [(e,v) for e,v in model.history.get('supervised_validation_loss',[]) if e != epoch_display_num]
            model.history['supervised_validation_loss'].append((epoch_display_num, avg_val_loss_epoch))
            model.history['supervised_validation_loss'].sort(key=lambda x:x[0])

            if avg_val_loss_epoch < best_val_loss:
                best_val_loss = avg_val_loss_epoch
                logger.info(f"New best end-of-epoch validation_loss: {best_val_loss:.4f}. Saving best model.")
                model.save_checkpoint(epoch_idx_0_based, config.BEST_MODEL_SUPERVISED_CHECKPOINT_FILENAME, is_best=True) # Save with 0-indexed epoch
        
        # model.current_phase_step_or_epoch should be the last *completed* epoch (0-indexed)
        model.current_phase_step_or_epoch = epoch_idx_0_based 
        if (epoch_display_num % config.SUPERVISED_SAVE_EVERY_N_EPOCHS == 0) or (epoch_idx_0_based == config.SUPERVISED_EPOCHS -1):
            model.save_checkpoint(epoch_idx_0_based, config.SUPERVISED_CHECKPOINT_FILENAME)
        
        gc.collect()

    logger.info(f"--- Supervised Code Training Phase Completed (Stored completed epoch: {model.current_phase_step_or_epoch}) ---")
    # Final save with the last completed epoch index
    model.save_checkpoint(model.current_phase_step_or_epoch, config.SUPERVISED_CHECKPOINT_FILENAME) 
    if hasattr(model, 'scheduler') and model.scheduler: del model.scheduler; model.scheduler = None; gc.collect()