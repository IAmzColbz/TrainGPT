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
            dataset_chunk, _, num_examples_in_this_chunk = get_dataloaders(
                task_type="prefix_lm_pretrain",
                tokenizer=global_tokenizer, 
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

    # Construct search pattern for FineWeb files
    fineweb_search_pattern = os.path.join(config.FINEWEB_DATA_DIR, "*.parquet")
    logger.info(f"Searching for FineWeb files in: {config.FINEWEB_DATA_DIR} with pattern: {fineweb_search_pattern}")
    all_found_files = sorted(glob.glob(fineweb_search_pattern))
    logger.info(f"Found {len(all_found_files)} FineWeb files initially: {all_found_files if len(all_found_files) < 10 else str(all_found_files[:5]) + '...'}")

    if not all_found_files:
        logger.warning(f"No FineWeb .parquet files found in directory '{config.FINEWEB_DATA_DIR}'. Skipping PrefixLM pre-training.")
        return

    # Determine files to cycle based on config.NUM_FINEWEB_FILES_TO_CYCLE
    if config.NUM_FINEWEB_FILES_TO_CYCLE <= 0: # Use all files if 0 or negative
        files_to_cycle = all_found_files
    else:
        files_to_cycle = all_found_files[:config.NUM_FINEWEB_FILES_TO_CYCLE]
    
    logger.info(f"Selected {len(files_to_cycle)} FineWeb files for this run based on NUM_FINEWEB_FILES_TO_CYCLE={config.NUM_FINEWEB_FILES_TO_CYCLE}.")

    if not files_to_cycle: # Should not happen if all_found_files was not empty and NUM_FINEWEB_FILES_TO_CYCLE wasn't misconfigured to slice to empty
        logger.warning("After applying NUM_FINEWEB_FILES_TO_CYCLE, no files selected. Skipping PrefixLM pre-training.")
        return

    steps_one_pass = precalculate_total_prefix_lm_steps(files_to_cycle, docs_chunk_size, batch_size, config.TOKENIZER_MODEL_PATH, model_run_name)
    total_training_steps = steps_one_pass * total_passes
    
    if total_training_steps == 0:
        logger.warning("Total estimated PrefixLM steps is 0 (possibly no data in selected files/chunks). Skipping pre-training.")
        return
    logger.info(f"Total estimated PrefixLM steps for {total_passes} passes: {total_training_steps}")

    initial_step_from_checkpoint = 0
    resumed_from_step = model.load_checkpoint(config.PREFIXLM_CHECKPOINT_FILENAME) # Use specific PrefixLM checkpoint
    if resumed_from_step > 0:
        initial_step_from_checkpoint = resumed_from_step -1 
        logger.info(f"Resuming PrefixLM from global_step: {initial_step_from_checkpoint + 1}")
    
    if model.scheduler: del model.scheduler 
    model.scheduler = get_linear_schedule_with_warmup(
        model.optimizer, 
        config.LR_SCHEDULER_WARMUP_STEPS, 
        total_training_steps, 
        last_epoch=initial_step_from_checkpoint 
    )
    global_step_counter = initial_step_from_checkpoint 

    for pass_idx in range(total_passes):
        logger.info(f"PrefixLM Global Pass {pass_idx + 1}/{total_passes}")
        
        if steps_one_pass > 0 and global_step_counter >= (pass_idx + 1) * steps_one_pass:
            logger.info(f"  Skipping Pass {pass_idx + 1} as global_step_counter ({global_step_counter}) is beyond this pass's range.")
            continue

        if torch.cuda.is_available(): torch.cuda.empty_cache(); gc.collect()
        
        for file_idx, file_path in enumerate(files_to_cycle):
            logger.info(f"  PrefixLM File {file_idx + 1}/{len(files_to_cycle)} in Pass {pass_idx+1}: {os.path.basename(file_path)}")
            
            estimated_total_docs_in_file = docs_chunk_size 
            try:
                pf = pq.ParquetFile(file_path)
                if pf.metadata: estimated_total_docs_in_file = pf.metadata.num_rows
                del pf
            except Exception: pass 
            
            num_chunks_in_file = math.ceil(estimated_total_docs_in_file / docs_chunk_size)
            current_doc_offset = 0
            
            for chunk_in_file_idx in range(num_chunks_in_file):
                if global_step_counter >= total_training_steps: break
                logger.info(f"    File {os.path.basename(file_path)}, Doc Chunk {chunk_in_file_idx + 1}/{num_chunks_in_file} (offset {current_doc_offset})")
                
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
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 
                        model.optimizer.step()
                        if model.scheduler: model.scheduler.step() 
                        
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
                
                model.save_checkpoint(global_step_counter, config.PREFIXLM_CHECKPOINT_FILENAME) 
                if current_doc_offset >= estimated_total_docs_in_file and estimated_total_docs_in_file > docs_chunk_size : break 
            if global_step_counter >= total_training_steps: break
        if global_step_counter >= total_training_steps: break
        
    logger.info(f"--- PrefixLM Pre-training Completed (Total Global Steps: {global_step_counter}) ---")
    model.save_checkpoint(global_step_counter, config.PREFIXLM_CHECKPOINT_FILENAME) 
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
    if result is None or result[0] is None: # get_dataloaders returns (train_dl, val_dl) for supervised
        logger.error("ERROR: Failed to create supervised code DataLoaders (train_dl is None). Aborting supervised phase.")
        return
    train_dataloader, val_dataloader = result

    if not train_dataloader or len(train_dataloader) == 0:
        logger.error("Supervised training dataloader is empty. Aborting supervised phase.")
        return

    num_training_steps_supervised = len(train_dataloader) * config.SUPERVISED_EPOCHS
    if num_training_steps_supervised == 0:
        logger.warning("No steps for supervised training (empty dataloader or 0 epochs). Skipping.")
        return

    start_epoch = 0 
    resumed_supervised_checkpoint = False
    optimizer_loaded_from_checkpoint = False
    scheduler_loaded_from_checkpoint = False


    best_supervised_ckpt_path = os.path.join(model.model_run_dir, config.BEST_MODEL_SUPERVISED_CHECKPOINT_FILENAME)
    regular_supervised_ckpt_path = os.path.join(model.model_run_dir, config.SUPERVISED_CHECKPOINT_FILENAME)

    if os.path.exists(best_supervised_ckpt_path):
        logger.info(f"Attempting to load best supervised checkpoint: {config.BEST_MODEL_SUPERVISED_CHECKPOINT_FILENAME}")
        start_epoch = model.load_checkpoint(config.BEST_MODEL_SUPERVISED_CHECKPOINT_FILENAME)
        resumed_supervised_checkpoint = True
        # Check if optimizer and scheduler were actually loaded by model.load_checkpoint
        # This requires model.load_checkpoint to indicate this, or we assume they were if checkpoint existed.
        # For simplicity, we assume if start_epoch > 0, they were likely loaded if present in checkpoint.
        if start_epoch > 0: optimizer_loaded_from_checkpoint = scheduler_loaded_from_checkpoint = True


    elif os.path.exists(regular_supervised_ckpt_path):
        logger.info(f"Attempting to load regular supervised checkpoint: {config.SUPERVISED_CHECKPOINT_FILENAME}")
        start_epoch = model.load_checkpoint(config.SUPERVISED_CHECKPOINT_FILENAME)
        resumed_supervised_checkpoint = True
        if start_epoch > 0: optimizer_loaded_from_checkpoint = scheduler_loaded_from_checkpoint = True
    
    if not resumed_supervised_checkpoint:
        prefixlm_checkpoint_path = os.path.join(model.model_run_dir, config.PREFIXLM_CHECKPOINT_FILENAME)
        if os.path.exists(prefixlm_checkpoint_path):
            logger.info(f"No supervised checkpoint found. Loading PrefixLM pretrained model from {prefixlm_checkpoint_path} as starting point for fine-tuning.")
            # Only loads model weights, does not affect optimizer/scheduler for the new phase
            model.load_checkpoint(config.PREFIXLM_CHECKPOINT_FILENAME) 
            start_epoch = 0 
            model.current_phase_step_or_epoch = 0 # Reset phase epoch for supervised
            # Clear supervised history if starting fine-tuning fresh from PrefixLM
            model.history['supervised_train_epoch_loss'] = []
            model.history['supervised_validation_loss'] = []
        else:
            logger.info("No supervised or PrefixLM checkpoints found. Starting supervised training from scratch.")
            start_epoch = 0
            model.current_phase_step_or_epoch = 0
    
    # Initialize or re-initialize optimizer and scheduler if not loaded from a dedicated supervised checkpoint
    if not optimizer_loaded_from_checkpoint:
        logger.info("Initializing new AdamW optimizer for supervised phase.")
        model.optimizer = optim.AdamW(model.parameters(), lr=config.NN_LEARNING_RATE / 2 if resumed_supervised_checkpoint else config.NN_LEARNING_RATE, weight_decay=config.OPTIMIZER_WEIGHT_DECAY)
    
    if not scheduler_loaded_from_checkpoint:
        logger.info("Initializing new LR scheduler for supervised phase.")
        if hasattr(model, 'scheduler') and model.scheduler: del model.scheduler
        # last_epoch for scheduler is (start_epoch_0_indexed * steps_per_epoch) - 1
        # However, get_linear_schedule_with_warmup's last_epoch is number of steps taken
        # If start_epoch is the *next* epoch to run (1-indexed from load_checkpoint),
        # then (start_epoch - 1) is num completed epochs.
        # For simplicity, if not loaded, start scheduler fresh (-1) or from where optim state suggests.
        # If optimizer was also fresh, last_epoch = -1. If optimizer was loaded but scheduler not, it's complex.
        # Safest is to re-init scheduler based on current completed supervised epochs.
        completed_epochs_for_scheduler = model.current_phase_step_or_epoch # This is last *completed* epoch
        last_scheduler_step = completed_epochs_for_scheduler * len(train_dataloader) -1 if completed_epochs_for_scheduler > 0 else -1

        model.scheduler = get_linear_schedule_with_warmup(
            model.optimizer, 
            min(config.LR_SCHEDULER_WARMUP_STEPS, num_training_steps_supervised // 10),
            num_training_steps_supervised,
            last_epoch=last_scheduler_step
        )

    # model.current_phase_step_or_epoch should be the last *completed* epoch (0-indexed)
    # The loop should go from this completed epoch up to total_epochs - 1.
    # So, if model.current_phase_step_or_epoch is 0 (start fresh or after PrefixLM), loop starts at 0.
    # If model.current_phase_step_or_epoch is N (resumed after N epochs), loop starts at N.
    
    loop_start_epoch_0_indexed = model.current_phase_step_or_epoch
    logger.info(f"Supervised training will run from epoch {loop_start_epoch_0_indexed + 1} to {config.SUPERVISED_EPOCHS}.")
    
    # This global_sup_step is for TensorBoard logging across all supervised epochs.
    # It should resume correctly based on completed epochs.
    global_sup_step = loop_start_epoch_0_indexed * len(train_dataloader)
    best_val_loss = min((h_val[1] for h_val in model.history.get('supervised_validation_loss', []) if h_val), default=float('inf'))

    for epoch_idx_0_based in range(loop_start_epoch_0_indexed, config.SUPERVISED_EPOCHS):
        epoch_display_num = epoch_idx_0_based + 1 
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
                    if model.scheduler: model.scheduler.step() 
                    
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

                if val_dataloader and global_sup_step > 0 and global_sup_step % config.SUPERVISED_VALIDATE_EVERY_N_BATCHES == 0:
                    avg_val_loss, _ = evaluate_model(model, val_dataloader, device, writer, global_sup_step, "Supervised_Validation_Mid_Epoch")
                    if writer: writer.add_scalar('Supervised_Validation/Step_Loss', avg_val_loss, global_sup_step) 
                    
                    if avg_val_loss < best_val_loss: 
                        best_val_loss = avg_val_loss
                        logger.info(f"New best mid-epoch validation_loss: {best_val_loss:.4f} at step {global_sup_step}. Saving best model.")
                        model.save_checkpoint(epoch_display_num, config.BEST_MODEL_SUPERVISED_CHECKPOINT_FILENAME, is_best=True) 
                    model.train() 
            except Exception as e_batch:
                logger.error(f"Error during supervised training batch {batch_idx} in epoch {epoch_display_num}: {e_batch}\n{traceback.format_exc()}")
                continue 

        progress_bar.close()
        avg_epoch_train_loss = epoch_loss_sum / num_batches_in_epoch if num_batches_in_epoch > 0 else float('inf')
        if writer: writer.add_scalar('Supervised/Train_Epoch_Loss', avg_epoch_train_loss, epoch_display_num)
        
        model.history['supervised_train_epoch_loss'] = [(e,v) for e,v in model.history.get('supervised_train_epoch_loss',[]) if e != epoch_display_num]
        model.history['supervised_train_epoch_loss'].append((epoch_display_num, avg_epoch_train_loss))
        model.history['supervised_train_epoch_loss'].sort(key=lambda x:x[0])

        if val_dataloader:
            logger.info(f"Running end-of-epoch validation for Supervised Epoch {epoch_display_num}...")
            avg_val_loss_epoch, _ = evaluate_model(model, val_dataloader, device, writer, global_sup_step, "Supervised_Validation_Epoch_End") 
            if writer: writer.add_scalar('Supervised_Validation/Epoch_End_Loss', avg_val_loss_epoch, epoch_display_num)

            model.history['supervised_validation_loss'] = [(e,v) for e,v in model.history.get('supervised_validation_loss',[]) if e != epoch_display_num]
            model.history['supervised_validation_loss'].append((epoch_display_num, avg_val_loss_epoch))
            model.history['supervised_validation_loss'].sort(key=lambda x:x[0])

            if avg_val_loss_epoch < best_val_loss:
                best_val_loss = avg_val_loss_epoch
                logger.info(f"New best end-of-epoch validation_loss: {best_val_loss:.4f}. Saving best model.")
                model.save_checkpoint(epoch_display_num, config.BEST_MODEL_SUPERVISED_CHECKPOINT_FILENAME, is_best=True)
        
        if epoch_display_num % config.SUPERVISED_SAVE_EVERY_N_EPOCHS == 0: 
            model.save_checkpoint(epoch_display_num, config.SUPERVISED_CHECKPOINT_FILENAME)
        
        model.current_phase_step_or_epoch = epoch_display_num 
        gc.collect()

    logger.info("--- Supervised Code Training Phase Completed ---")
    model.save_checkpoint(model.current_phase_step_or_epoch, config.SUPERVISED_CHECKPOINT_FILENAME) 
    if hasattr(model, 'scheduler') and model.scheduler: del model.scheduler; model.scheduler = None; gc.collect()
