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
from dataset_loader import get_dataloaders # FineWebPrefixLMDataset is used via get_dataloaders
from tokenizer_wrapper import global_tokenizer
import math
import gc
import traceback
import torch.optim as optim
import logging # Added logging

logger = logging.getLogger(__name__) # Added logger

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0,
    after a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
    """
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
        return float('inf'), 0.0 # Return neutral values

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
                # Optionally, re-raise or handle more gracefully
                continue # Skip this batch

    avg_val_loss = total_val_loss / len(val_dataloader) if len(val_dataloader) > 0 else float('inf')
    
    if writer: # Check if writer is provided
        writer.add_scalar(f'{phase_name}/Loss_Eval_Step', avg_val_loss, global_step)
    
    logger.info(f"{phase_name} Evaluation Complete. Avg Loss: {avg_val_loss:.4f} (at global step {global_step})")
    progress_bar.close()
    return avg_val_loss, 0.0 # Second value could be BLEU or other metric if implemented

def precalculate_total_prefix_lm_steps(files_to_cycle, docs_chunk_size, batch_size, tokenizer_path, model_run_name_for_cache):
    logger.info(f"Pre-calculating PrefixLM steps for run '{model_run_name_for_cache}' (this may build or check caches)...")
    total_estimated_steps_one_pass = 0
    
    # This base path should align with how FineWebPrefixLMDataset constructs its cache paths
    run_specific_cache_base = os.path.join(config.PROJECT_ROOT, "dataset_cache", model_run_name_for_cache)

    for clm_file_path in tqdm(files_to_cycle, desc="Pre-calculating steps (files)", unit="file", ncols=100):
        # Estimate total documents in this file. Parquet metadata is preferred.
        estimated_total_docs_in_file = docs_chunk_size # Fallback
        try:
            pf = pq.ParquetFile(clm_file_path)
            if pf.metadata: estimated_total_docs_in_file = pf.metadata.num_rows
            del pf; gc.collect()
        except Exception as e: 
            logger.warning(f"Could not get exact row count for {os.path.basename(clm_file_path)}: {e}. Using fallback estimation: {estimated_total_docs_in_file} docs.")

        num_chunks_in_file = math.ceil(estimated_total_docs_in_file / docs_chunk_size)
        current_doc_offset = 0
        
        for _ in tqdm(range(num_chunks_in_file), desc=f"Pre-calc Chunks for {os.path.basename(clm_file_path)}", unit="chunk", leave=False, ncols=100):
            # Instantiate dataset to trigger cache build/load and get length
            # Ensure parameters match those used in actual training loader
            dataset_chunk, _, num_examples_in_this_chunk = get_dataloaders(
                task_type="prefix_lm_pretrain",
                tokenizer=global_tokenizer, # Not strictly needed by get_dataloaders here, but FineWeb... uses it
                batch_size=batch_size, # Not used for len, but part of signature
                specific_file_path=clm_file_path,
                start_doc_offset=current_doc_offset,
                num_docs_in_chunk=docs_chunk_size,
                model_run_name_for_cache=model_run_name_for_cache # Crucial for correct cache path
            )

            if num_examples_in_this_chunk > 0:
                total_estimated_steps_one_pass += math.ceil(num_examples_in_this_chunk / batch_size)
            
            if dataset_chunk: del dataset_chunk; gc.collect()
            
            current_doc_offset += docs_chunk_size
            if current_doc_offset >= estimated_total_docs_in_file: # Stop if we've processed all estimated docs
                 break
                 
    logger.info(f"Pre-calculation complete. Estimated PrefixLM steps for ONE pass: {total_estimated_steps_one_pass}")
    return total_estimated_steps_one_pass


def run_prefix_lm_pretraining_phase(model, writer, model_run_name): # model_run_name is the unique dir name
    logger.info(f"--- PrefixLM Pre-training Phase Initiated for run: {model_run_name} ---")
    device = model.device
    batch_size = config.PREFIXLM_BATCH_SIZE
    total_passes = config.PREFIXLM_TOTAL_PASSES # Total passes over the dataset
    docs_chunk_size = config.DOCS_CHUNK_SIZE_PER_PREFIXLM_EPOCH # Docs per sub-dataset

    files = sorted(glob.glob(os.path.join(config.FINEWEB_DATA_DIR, "*.parquet")))
    files_to_cycle = files[:config.NUM_FINEWEB_FILES_TO_CYCLE]
    if not files_to_cycle:
        logger.warning("No FineWeb files found. Skipping PrefixLM pre-training.")
        return

    # Pre-calculate total steps for the LR scheduler accurately
    # model_run_name is used for consistent cache pathing
    steps_one_pass = precalculate_total_prefix_lm_steps(files_to_cycle, docs_chunk_size, batch_size, config.TOKENIZER_MODEL_PATH, model_run_name)
    total_training_steps = steps_one_pass * total_passes
    
    if total_training_steps == 0:
        logger.warning("Total estimated PrefixLM steps is 0. Skipping pre-training.")
        return
    logger.info(f"Total estimated PrefixLM steps for {total_passes} passes: {total_training_steps}")

    initial_step_from_checkpoint = 0
    # model.load_checkpoint returns (last_completed_epoch/step + 1) or 0 if no checkpoint
    # For PrefixLM, this "epoch" is actually the global_step
    resumed_from_step = model.load_checkpoint(config.CLM_PRETRAIN_CHECKPOINT_FILENAME)
    if resumed_from_step > 0:
        initial_step_from_checkpoint = resumed_from_step -1 # Scheduler's last_epoch is 0-indexed num of steps completed
        logger.info(f"Resuming PrefixLM from global_step: {initial_step_from_checkpoint + 1}")
    
    # Re-initialize optimizer (or ensure it's compatible if loaded)
    # For simplicity, let's assume the model's optimizer is either fresh or loaded correctly.
    # The scheduler needs to be fresh or its state loaded correctly.
    if model.scheduler: del model.scheduler # remove old one if any
    model.scheduler = get_linear_schedule_with_warmup(
        model.optimizer, 
        config.LR_SCHEDULER_WARMUP_STEPS, 
        total_training_steps, 
        last_epoch=initial_step_from_checkpoint # last_epoch is "number of steps already taken"
    )
    global_step_counter = initial_step_from_checkpoint # Number of steps already completed

    for pass_idx in range(total_passes):
        logger.info(f"PrefixLM Global Pass {pass_idx + 1}/{total_passes}")
        
        # Rough check to skip pass if already covered by resumed global_step_counter
        if steps_one_pass > 0 and global_step_counter >= (pass_idx + 1) * steps_one_pass:
            logger.info(f"  Skipping Pass {pass_idx + 1} as global_step_counter ({global_step_counter}) is beyond this pass's range.")
            continue

        if torch.cuda.is_available(): torch.cuda.empty_cache(); gc.collect()
        
        for file_idx, file_path in enumerate(files_to_cycle):
            logger.info(f"  PrefixLM File {file_idx + 1}/{len(files_to_cycle)} in Pass {pass_idx+1}: {os.path.basename(file_path)}")
            
            # Estimate total documents in this file for chunking progress
            estimated_total_docs_in_file = docs_chunk_size # Fallback
            try:
                pf = pq.ParquetFile(file_path)
                if pf.metadata: estimated_total_docs_in_file = pf.metadata.num_rows
                del pf
            except Exception: pass # Keep fallback
            
            num_chunks_in_file = math.ceil(estimated_total_docs_in_file / docs_chunk_size)
            current_doc_offset = 0
            
            for chunk_in_file_idx in range(num_chunks_in_file):
                if global_step_counter >= total_training_steps: break
                logger.info(f"    File {os.path.basename(file_path)}, Doc Chunk {chunk_in_file_idx + 1}/{num_chunks_in_file} (offset {current_doc_offset})")
                
                # Get dataloader for the current chunk of the current file
                # model_run_name is passed for unique cache path generation by get_dataloaders -> FineWebPrefixLMDataset
                train_dl, _, num_ex_chunk = get_dataloaders(
                    task_type="prefix_lm_pretrain", tokenizer=global_tokenizer, batch_size=batch_size,
                    specific_file_path=file_path, start_doc_offset=current_doc_offset,
                    num_docs_in_chunk=docs_chunk_size,
                    model_run_name_for_cache=model_run_name
                )
                
                current_doc_offset += docs_chunk_size # Prepare for next chunk

                if not train_dl or num_ex_chunk == 0:
                    logger.warning(f"      No examples in this chunk. Skipping.")
                    if current_doc_offset >= estimated_total_docs_in_file and estimated_total_docs_in_file > docs_chunk_size : break
                    continue
                
                model.train()
                pbar = tqdm(train_dl, desc=f"PrefixLM P{pass_idx+1} F{file_idx+1} Ck{chunk_in_file_idx+1}", leave=False, ncols=120)
                for batch in pbar:
                    if global_step_counter >= total_training_steps: break
                    
                    src = batch['src_token_ids'].to(device, non_blocking=True)
                    dec_in = batch['decoder_input_ids'].to(device, non_blocking=True)
                    lbl = batch['label_ids'].to(device, non_blocking=True)
                    
                    model.optimizer.zero_grad(set_to_none=True)
                    logits = model(src, dec_in)
                    loss = model.criterion(logits.view(-1, model.vocab_size), lbl.view(-1))
                    
                    if not torch.isnan(loss) and not torch.isinf(loss):
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradient clipping
                        model.optimizer.step()
                        if model.scheduler: model.scheduler.step() # Step scheduler
                        
                        if writer:
                            writer.add_scalar('PrefixLM/Batch_Loss', loss.item(), global_step_counter)
                            current_lr = model.optimizer.param_groups[0]['lr']
                            writer.add_scalar('PrefixLM/Learning_Rate', current_lr, global_step_counter)
                        
                        pbar.set_postfix({"Loss": f"{loss.item():.4f}", "LR":f"{current_lr:.2e}", "Step":f"{global_step_counter + 1}/{total_training_steps}"})
                    else:
                        logger.warning(f"PrefixLM training: NaN or Inf loss encountered at global_step {global_step_counter}. Skipping optimizer step.")

                    global_step_counter += 1
                
                pbar.close()
                del train_dl, batch, src, dec_in, lbl, logits, loss; gc.collect()
                
                # Save checkpoint using global_step_counter as the 'epoch' for this phase
                # model.current_phase_epoch within the model will be updated by load_checkpoint
                # and save_checkpoint uses the passed epoch_num.
                model.save_checkpoint(global_step_counter, config.CLM_PRETRAIN_CHECKPOINT_FILENAME) # Save progress
                if current_doc_offset >= estimated_total_docs_in_file and estimated_total_docs_in_file > docs_chunk_size : break # Ensure we don't loop beyond file contents
            if global_step_counter >= total_training_steps: break
        if global_step_counter >= total_training_steps: break
        
    logger.info(f"--- PrefixLM Pre-training Completed (Total Global Steps: {global_step_counter}) ---")
    model.save_checkpoint(global_step_counter, config.CLM_PRETRAIN_CHECKPOINT_FILENAME) # Final save
    if hasattr(model, 'scheduler') and model.scheduler: del model.scheduler; model.scheduler = None; gc.collect()


def run_supervised_training_phase(model, writer, model_run_name):
    logger.info(f"--- Supervised Code Training Phase Initiated for run: {model_run_name} ---")
    device = model.device
    
    result = get_dataloaders(
        task_type="supervised_code", 
        tokenizer=global_tokenizer, 
        batch_size=config.SUPERVISED_BATCH_SIZE, 
        val_split_ratio=config.SUPERVISED_VALIDATION_SPLIT_RATIO,
        model_run_name_for_cache=model_run_name # Though not strictly for cache, good for consistency
    )
    if result is None or result[0] is None:
        logger.error("ERROR: Failed to create supervised code DataLoaders. Aborting supervised phase.")
        return
    train_dataloader, val_dataloader = result

    if not train_dataloader or len(train_dataloader) == 0:
        logger.error("Supervised training dataloader is empty. Aborting supervised phase.")
        return

    num_training_steps_supervised = len(train_dataloader) * config.SUPERVISED_EPOCHS
    if num_training_steps_supervised == 0:
        logger.warning("No steps for supervised training (empty dataloader or 0 epochs). Skipping.")
        return

    # --- Checkpoint and Optimizer/Scheduler Initialization ---
    # Determine if resuming supervised, starting from CLM, or starting fresh
    start_epoch = 0 # 0-indexed for the loop (epoch to begin training)
    resumed_supervised_checkpoint = False

    # Try loading best supervised checkpoint first
    if os.path.exists(os.path.join(model.model_run_dir, config.BEST_MODEL_SUPERVISED_CHECKPOINT_FILENAME)):
        logger.info(f"Loading best supervised checkpoint: {config.BEST_MODEL_SUPERVISED_CHECKPOINT_FILENAME}")
        # load_checkpoint returns (last_completed_epoch + 1)
        start_epoch = model.load_checkpoint(config.BEST_MODEL_SUPERVISED_CHECKPOINT_FILENAME)
        # If optimizer/scheduler were loaded, model.optimizer and model.scheduler are now populated.
        resumed_supervised_checkpoint = True
    # Else, try loading regular supervised checkpoint
    elif os.path.exists(os.path.join(model.model_run_dir, config.SUPERVISED_CHECKPOINT_FILENAME)):
        logger.info(f"Loading regular supervised checkpoint: {config.SUPERVISED_CHECKPOINT_FILENAME}")
        start_epoch = model.load_checkpoint(config.SUPERVISED_CHECKPOINT_FILENAME)
        resumed_supervised_checkpoint = True
    
    # If not resuming a supervised checkpoint, decide if starting from CLM or entirely fresh
    if not resumed_supervised_checkpoint:
        clm_checkpoint_path = os.path.join(model.model_run_dir, config.CLM_PRETRAIN_CHECKPOINT_FILENAME)
        if os.path.exists(clm_checkpoint_path):
            logger.info(f"No supervised checkpoint found. Loading CLM pretrained model from {clm_checkpoint_path} as starting point for fine-tuning.")
            # Load CLM weights. This might also load CLM optimizer/scheduler state into model attributes.
            model.load_checkpoint(config.CLM_PRETRAIN_CHECKPOINT_FILENAME) 
            # We are starting supervised training fresh, so epochs start from 0 for this phase.
            start_epoch = 0 
            # Optimizer and Scheduler must be re-initialized for the supervised phase
            logger.info("Re-initializing optimizer and scheduler for supervised fine-tuning.")
            model.optimizer = optim.AdamW(model.parameters(), lr=config.NN_LEARNING_RATE / 2, weight_decay=config.OPTIMIZER_WEIGHT_DECAY) # Example: smaller LR for fine-tune
            if hasattr(model, 'scheduler') and model.scheduler: del model.scheduler
            model.scheduler = get_linear_schedule_with_warmup(
                model.optimizer, 
                min(config.LR_SCHEDULER_WARMUP_STEPS, num_training_steps_supervised // 10), # Warmup for a portion of supervised steps
                num_training_steps_supervised,
                last_epoch=-1 # Start scheduler fresh
            )
            # Clear any supervised-specific history from previous (unrelated) runs if model object is reused
            model.history['supervised_train_epoch_loss'] = []
            model.history['supervised_validation_loss'] = []
            model.history['supervised_validation_bleu'] = [] # Assuming BLEU is a metric
        else:
            logger.info("No supervised or CLM checkpoints found. Starting supervised training from scratch with fresh optimizer/scheduler.")
            start_epoch = 0
            model.optimizer = optim.AdamW(model.parameters(), lr=config.NN_LEARNING_RATE, weight_decay=config.OPTIMIZER_WEIGHT_DECAY)
            if hasattr(model, 'scheduler') and model.scheduler: del model.scheduler
            model.scheduler = get_linear_schedule_with_warmup(
                model.optimizer, 
                config.LR_SCHEDULER_WARMUP_STEPS, 
                num_training_steps_supervised,
                last_epoch=-1
            )
    else: # Resumed a supervised checkpoint
        logger.info(f"Resuming supervised training. Optimizer and Scheduler state loaded from checkpoint.")
        # Ensure scheduler's total steps are for this phase, might need re-init if total_steps changed
        # However, load_checkpoint should restore it. If not, re-init here using loaded optimizer.
        # For LambdaLR, if total_steps changed, it should be re-created.
        # For simplicity, let's assume load_checkpoint correctly restores scheduler or it's robust enough.
        # A more robust way if total_steps might change across resumption:
        # current_scheduler_step = model.scheduler.last_epoch if model.scheduler else -1 (if scheduler was loaded)
        # model.scheduler = get_linear_schedule_with_warmup(model.optimizer, ..., num_training_steps_supervised, last_epoch=current_scheduler_step)

    # `model.current_phase_epoch` is updated by `load_checkpoint` to last completed epoch.
    # The loop should start from `start_epoch` which is (last_completed_epoch + 1).
    # So, loop from `model.current_phase_epoch` if it was set by load_checkpoint, or `start_epoch` if fresh.
    # `load_checkpoint` sets `model.current_phase_epoch` to the epoch number *saved in the checkpoint*.
    # This is the last *completed* epoch. So, training starts from `model.current_phase_epoch`.
    # The loop should be: for epoch_idx_0_based in range(model.current_phase_epoch, config.SUPERVISED_EPOCHS):

    initial_completed_epochs = model.current_phase_epoch if resumed_supervised_checkpoint else 0
    if start_epoch > initial_completed_epochs : # If load_checkpoint returned a higher start_epoch due to some logic
        initial_completed_epochs = start_epoch -1

    logger.info(f"Starting supervised training from epoch {initial_completed_epochs +1}. Last completed epoch: {initial_completed_epochs}.")

    global_sup_step = initial_completed_epochs * len(train_dataloader)
    best_val_loss = min((h_val[1] for h_val in model.history.get('supervised_validation_loss', []) if h_val), default=float('inf'))

    for epoch_idx_0_based in range(initial_completed_epochs, config.SUPERVISED_EPOCHS):
        epoch_display_num = epoch_idx_0_based + 1 # For logging (1-indexed)
        logger.info(f"Supervised Code Epoch {epoch_display_num}/{config.SUPERVISED_EPOCHS}")
        
        if torch.cuda.is_available(): torch.cuda.empty_cache(); gc.collect()
        model.train()
        epoch_loss_sum = 0.0
        num_batches_in_epoch = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Supervised Epoch {epoch_display_num}", leave=False, ncols=120)
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
                    if model.scheduler: model.scheduler.step() # Step scheduler per batch
                    
                    epoch_loss_sum += loss.item()
                    num_batches_in_epoch += 1
                    current_lr_sup = model.optimizer.param_groups[0]['lr']
                    
                    if writer:
                        writer.add_scalar('Supervised/Batch_Loss', loss.item(), global_sup_step)
                        writer.add_scalar('Supervised/Learning_Rate', current_lr_sup, global_sup_step)
                    
                    progress_bar.set_postfix({"Loss": f"{loss.item():.4f}", "LR":f"{current_lr_sup:.2e}"})
                else:
                    logger.warning(f"Supervised training: NaN or Inf loss at epoch {epoch_display_num}, batch {batch_idx}, global_step {global_sup_step}. Skipping optimizer step.")

                global_sup_step += 1

                # Periodic Validation
                if val_dataloader and global_sup_step > 0 and global_sup_step % config.SUPERVISED_VALIDATE_EVERY_N_BATCHES == 0:
                    avg_val_loss, _ = evaluate_model(model, val_dataloader, device, writer, global_sup_step, "Supervised_Validation_Mid_Epoch")
                    if writer: writer.add_scalar('Supervised_Validation/Step_Loss', avg_val_loss, global_sup_step) # Log with global_step
                    
                    # Store validation loss with global_step for more granular history if needed
                    # model.history['supervised_validation_loss'].append((global_sup_step, avg_val_loss)) 
                    # For epoch-level val loss, do it after epoch.

                    if avg_val_loss < best_val_loss: 
                        best_val_loss = avg_val_loss
                        logger.info(f"New best mid-epoch validation_loss: {best_val_loss:.4f} at step {global_sup_step}. Saving best model.")
                        model.save_checkpoint(epoch_display_num, config.BEST_MODEL_SUPERVISED_CHECKPOINT_FILENAME) # Save with current epoch number
                    model.train() # Set back to train mode
            except Exception as e_batch:
                logger.error(f"Error during supervised training batch {batch_idx} in epoch {epoch_display_num}: {e_batch}\n{traceback.format_exc()}")
                continue # Skip to next batch

        progress_bar.close()
        avg_epoch_train_loss = epoch_loss_sum / num_batches_in_epoch if num_batches_in_epoch > 0 else float('inf')
        if writer: writer.add_scalar('Supervised/Train_Epoch_Loss', avg_epoch_train_loss, epoch_display_num)
        
        # Update training history for this epoch
        # Remove old entry for this epoch if resuming and re-running it
        model.history['supervised_train_epoch_loss'] = [(e,v) for e,v in model.history.get('supervised_train_epoch_loss',[]) if e != epoch_display_num]
        model.history['supervised_train_epoch_loss'].append((epoch_display_num, avg_epoch_train_loss))
        model.history['supervised_train_epoch_loss'].sort(key=lambda x:x[0])

        # End-of-Epoch Validation
        if val_dataloader:
            logger.info(f"Running end-of-epoch validation for Supervised Epoch {epoch_display_num}...")
            avg_val_loss_epoch, _ = evaluate_model(model, val_dataloader, device, writer, global_sup_step, "Supervised_Validation_Epoch_End") # Use global_step or epoch_display_num
            if writer: writer.add_scalar('Supervised_Validation/Epoch_End_Loss', avg_val_loss_epoch, epoch_display_num)

            model.history['supervised_validation_loss'] = [(e,v) for e,v in model.history.get('supervised_validation_loss',[]) if e != epoch_display_num]
            model.history['supervised_validation_loss'].append((epoch_display_num, avg_val_loss_epoch))
            model.history['supervised_validation_loss'].sort(key=lambda x:x[0])

            if avg_val_loss_epoch < best_val_loss:
                best_val_loss = avg_val_loss_epoch
                logger.info(f"New best end-of-epoch validation_loss: {best_val_loss:.4f}. Saving best model.")
                model.save_checkpoint(epoch_display_num, config.BEST_MODEL_SUPERVISED_CHECKPOINT_FILENAME)
        
        # Save regular checkpoint
        if epoch_display_num % config.SUPERVISED_SAVE_EVERY_N_EPOCHS == 0: 
            model.save_checkpoint(epoch_display_num, config.SUPERVISED_CHECKPOINT_FILENAME)
        
        model.current_phase_epoch = epoch_display_num # Update model's tracker of completed epochs for this phase
        gc.collect()

    logger.info("--- Supervised Code Training Phase Completed ---")
    model.save_checkpoint(model.current_phase_epoch, config.SUPERVISED_CHECKPOINT_FILENAME) # Save final model
    if hasattr(model, 'scheduler') and model.scheduler: del model.scheduler; model.scheduler = None; gc.collect()