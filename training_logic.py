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
from dataset_loader import FineWebPrefixLMDataset, get_dataloaders # FineWebPrefixLMDataset is used via get_dataloaders
from tokenizer_wrapper import global_tokenizer
import math
import gc
import traceback
import torch.optim as optim

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps: return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
    return LambdaLR(optimizer, lr_lambda, last_epoch)

def evaluate_model(model, val_dataloader, device, writer, global_step, phase_name="Validation"):
    model.eval(); total_val_loss = 0
    if not val_dataloader or len(val_dataloader) == 0: return float('inf'), 0.0
    progress_bar = tqdm(val_dataloader, desc=f"{phase_name} Evaluating", leave=False, ncols=100, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}')
    with torch.no_grad():
        for batch in progress_bar:
            # Ensure keys match what collate_fn for this phase provides
            src_ids = batch['src_token_ids'].to(device)
            dec_in_ids = batch['decoder_input_ids'].to(device)
            lbl_ids = batch['label_ids'].to(device)
            logits = model(src_ids, dec_in_ids)
            loss = model.criterion(logits.view(-1, model.vocab_size), lbl_ids.view(-1))
            total_val_loss += loss.item()
    avg_val_loss = total_val_loss / len(val_dataloader)
    if writer: writer.add_scalar(f'{phase_name}/Loss_Eval_Step', avg_val_loss, global_step)
    print(f"{phase_name} Complete. Avg Loss: {avg_val_loss:.4f} (step {global_step})"); progress_bar.close()
    return avg_val_loss, 0.0

def precalculate_total_prefix_lm_steps(files_to_cycle, docs_chunk_size, batch_size, tokenizer_path, model_run_name):
    print(f"Pre-calculating PrefixLM steps for run '{model_run_name}' (builds/checks caches)...")
    total_steps = 0
    # Construct the base cache directory path specific to this run
    cache_dir_base_for_run = config.CACHE_DIR_PREFIXLM_BASE.format(model_run_name=model_run_name)

    for clm_file_path in tqdm(files_to_cycle, desc="Pre-calc Files", unit="file", ncols=100):
        total_docs_in_this_file = docs_chunk_size 
        try:
            pf = pq.ParquetFile(clm_file_path)
            if pf.metadata: total_docs_in_this_file = pf.metadata.num_rows
            del pf; gc.collect()
        except Exception as e: 
            print(f"Warning: Could not get exact row count for {os.path.basename(clm_file_path)}: {e}. Using estimations.")


        num_chunks_in_file = math.ceil(total_docs_in_this_file / docs_chunk_size)
        current_doc_offset = 0
        for _ in tqdm(range(num_chunks_in_file), desc=f"Pre-calc Chunks {os.path.basename(clm_file_path)}", unit="chunk", leave=False, ncols=100):
            dataset_chunk = FineWebPrefixLMDataset( # Use FineWebPrefixLMDataset
                clm_file_path, 
                tokenizer_model_path_for_worker=tokenizer_path, # Pass path for workers
                max_prefix_suffix_len=config.MAX_TEXT_PRETRAIN_SEQ_LEN,
                start_doc_offset=current_doc_offset,
                num_docs_to_process_in_chunk=docs_chunk_size,
                cache_dir_base_with_run_name=cache_dir_base_for_run # Pass run-specific base
            )
            num_examples_in_this_chunk = len(dataset_chunk)
            if num_examples_in_this_chunk > 0:
                total_steps += math.ceil(num_examples_in_this_chunk / batch_size)
            del dataset_chunk; gc.collect()
            current_doc_offset += docs_chunk_size
            # Check to prevent processing beyond the actual file content if total_docs_in_this_file was estimated
            if current_doc_offset >= total_docs_in_this_file and total_docs_in_this_file > 0 : # If we know total_docs accurately
                 break
    print(f"Pre-calculation complete. Estimated PrefixLM steps for ONE pass: {total_steps}"); return total_steps


def run_prefix_lm_pretraining_phase(model, writer, model_run_name):
    print("--- PrefixLM Pre-training Phase Initiated ---"); device = model.device
    batch_size = config.PREFIXLM_BATCH_SIZE
    total_passes = config.PREFIXLM_TOTAL_PASSES
    docs_chunk_size = config.DOCS_CHUNK_SIZE_PER_PREFIXLM_EPOCH

    files = sorted(glob.glob(os.path.join(config.FINEWEB_DATA_DIR, "*.parquet")))
    files_to_cycle = files[:config.NUM_FINEWEB_FILES_TO_CYCLE]
    if not files_to_cycle: print("No FineWeb files. Skipping PrefixLM."); return

    steps_one_pass = precalculate_total_prefix_lm_steps(files_to_cycle, docs_chunk_size, batch_size, config.TOKENIZER_MODEL_PATH, model_run_name)
    total_training_steps = steps_one_pass * total_passes
    if total_training_steps == 0: print("No PrefixLM steps. Skipping."); return
    print(f"Total estimated PrefixLM steps for {total_passes} passes: {total_training_steps}")

    initial_step_from_checkpoint = 0
    _old_scheduler = model.scheduler; model.scheduler = None # Temporarily remove to avoid state load issues if re-initing
    loaded_epoch_equiv_step = model.load_checkpoint(config.CLM_PRETRAIN_CHECKPOINT_FILENAME) # This now refers to global_step
    if loaded_epoch_equiv_step > 0 : initial_step_from_checkpoint = loaded_epoch_equiv_step
    
    model.scheduler = get_linear_schedule_with_warmup(model.optimizer, config.LR_SCHEDULER_WARMUP_STEPS, total_training_steps, last_epoch=initial_step_from_checkpoint -1)
    global_step_counter = initial_step_from_checkpoint

    for pass_idx in range(total_passes):
        print(f"PrefixLM Global Pass {pass_idx + 1}/{total_passes}")
        # Skip pass if already covered by resumed global_step_counter
        # Rough check: if current global step is beyond what this pass would start with.
        # A pass has `steps_one_pass` steps.
        if global_step_counter >= (pass_idx + 1) * steps_one_pass and steps_one_pass > 0 :
            print(f"  Skipping Pass {pass_idx + 1} as global_step_counter ({global_step_counter}) is beyond this pass's range.")
            continue

        if torch.cuda.is_available(): torch.cuda.empty_cache(); gc.collect()
        for file_idx, file_path in enumerate(files_to_cycle):
            print(f"  PrefixLM File {file_idx + 1}/{len(files_to_cycle)} in Pass {pass_idx+1}: {os.path.basename(file_path)}")
            total_docs_in_this_file = docs_chunk_size 
            try: pf = pq.ParquetFile(file_path); total_docs_in_this_file = pf.metadata.num_rows if pf.metadata else docs_chunk_size; del pf
            except: pass
            num_chunks_in_file = math.ceil(total_docs_in_this_file / docs_chunk_size)
            current_doc_offset = 0; chunk_in_file_idx = 0
            
            while current_doc_offset < total_docs_in_this_file:
                chunk_in_file_idx += 1; gc.collect()
                # Skip chunks if already processed according to global_step_counter
                # This estimation is tricky. A simpler way is just to run and let scheduler manage LR.
                # For now, we just run based on pass_idx.
                
                print(f"    File {os.path.basename(file_path)}, Doc Chunk {chunk_in_file_idx}/{num_chunks_in_file} (offset {current_doc_offset})")
                run_specific_cache_base = config.CACHE_DIR_PREFIXLM_BASE.format(model_run_name=model_run_name)
                train_dl, _, num_ex_chunk = get_dataloaders(
                    task_type="prefix_lm_pretrain", tokenizer=global_tokenizer, batch_size=batch_size,
                    specific_file_path=file_path, start_doc_offset=current_doc_offset,
                    num_docs_in_chunk=docs_chunk_size,
                    model_run_name_for_cache=model_run_name # Pass for unique cache path
                )
                if not train_dl or num_ex_chunk == 0:
                    current_doc_offset += docs_chunk_size
                    if current_doc_offset >= total_docs_in_this_file and docs_chunk_size < total_docs_in_this_file: break
                    continue
                
                model.train()
                pbar = tqdm(train_dl, desc=f"PrefixLM P{pass_idx+1} F{file_idx+1} C{chunk_in_file_idx}", leave=False, ncols=120)
                for batch in pbar:
                    if global_step_counter >= total_training_steps: break
                    # In PrefixLM, 'src_token_ids' is the prefix, 'decoder_input_ids' is shifted suffix
                    src, dec_in, lbl = batch['src_token_ids'].to(device,non_blocking=True), batch['decoder_input_ids'].to(device,non_blocking=True), batch['label_ids'].to(device,non_blocking=True)
                    model.optimizer.zero_grad(set_to_none=True)
                    logits = model(src, dec_in) # Pass both src (prefix) and dec_in (shifted suffix)
                    loss = model.criterion(logits.view(-1, model.vocab_size), lbl.view(-1)) # lbl is suffix
                    if not torch.isnan(loss): loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); model.optimizer.step()
                    if model.scheduler: model.scheduler.step()
                    writer.add_scalar('PrefixLM/Batch_Loss', loss.item(), global_step_counter)
                    lr = model.optimizer.param_groups[0]['lr']
                    writer.add_scalar('PrefixLM/Learning_Rate', lr, global_step_counter)
                    global_step_counter += 1
                    pbar.set_postfix({"Loss": f"{loss.item():.4f}", "LR":f"{lr:.2e}", "Step":f"{global_step_counter}/{total_training_steps}"})
                pbar.close(); del train_dl, batch, src, dec_in, lbl, logits, loss; gc.collect()
                current_doc_offset += docs_chunk_size
                model.save_checkpoint(global_step_counter, config.CLM_PRETRAIN_CHECKPOINT_FILENAME)
                if global_step_counter >= total_training_steps: break
            if global_step_counter >= total_training_steps: break
        if global_step_counter >= total_training_steps: break
    print(f"--- PrefixLM Pre-training Completed (Global Steps: {global_step_counter}) ---")
    model.save_checkpoint(global_step_counter, config.CLM_PRETRAIN_CHECKPOINT_FILENAME)
    if model.scheduler: del model.scheduler; model.scheduler = None; gc.collect()

# run_supervised_training_phase (ensure it correctly initializes its own scheduler and handles checkpoints)
def run_supervised_training_phase(model, writer, model_run_name):
    # ... (This function remains largely the same as the last full version I provided for it)
    # Key things to ensure:
    # 1. It calls get_dataloaders with task_type="supervised_code".
    # 2. It initializes its own LR scheduler using get_linear_schedule_with_warmup,
    #    calculating its own num_training_steps_supervised based on its dataloader and SUPERVISED_EPOCHS.
    # 3. It loads checkpoints using config.BEST_MODEL_SUPERVISED_CHECKPOINT_FILENAME or config.SUPERVISED_CHECKPOINT_FILENAME.
    #    If neither exists, it can optionally try to load config.CLM_PRETRAIN_CHECKPOINT_FILENAME as a base.
    # 4. It saves checkpoints using config.SUPERVISED_CHECKPOINT_FILENAME and config.BEST_MODEL_SUPERVISED_CHECKPOINT_FILENAME.
    # 5. `model.current_phase_epoch` should track epochs *for this supervised phase*.
    print("--- Supervised Code Training Phase Initiated ---"); device = model.device
    result = get_dataloaders(task_type="supervised_code", tokenizer=global_tokenizer, batch_size=config.SUPERVISED_BATCH_SIZE, val_split_ratio=config.SUPERVISED_VALIDATION_SPLIT_RATIO, model_run_name_for_cache=model_run_name)
    if result is None or result[0] is None: print("ERROR: Failed to create supervised code DataLoader. Aborting."); return
    train_dataloader, val_dataloader = result # Expects 2 return values

    num_steps_sup = len(train_dataloader) * config.SUPERVISED_EPOCHS
    if num_steps_sup == 0: print("No steps for supervised training (empty dataloader or 0 epochs). Skipping."); return

    # Reset optimizer and scheduler for fine-tuning phase IF desired, or adjust LR
    # For true fine-tuning, a new optimizer with smaller LR is common.
    # Here, we re-initialize the scheduler for the current optimizer.
    model.optimizer = optim.AdamW(model.parameters(), lr=config.NN_LEARNING_RATE / 2, weight_decay=config.OPTIMIZER_WEIGHT_DECAY) # Example: smaller LR for fine-tune
    model.scheduler = get_linear_schedule_with_warmup(model.optimizer, min(config.LR_SCHEDULER_WARMUP_STEPS, num_steps_sup // 10), num_steps_sup, -1)
    
    start_epoch = 0 # 0-indexed for the loop
    # Try loading best supervised, then regular supervised.
    # If none, and CLM checkpoint exists, it means we are fine-tuning from CLM.
    best_sup_ckpt = os.path.join(model.model_run_dir, config.BEST_MODEL_SUPERVISED_CHECKPOINT_FILENAME)
    reg_sup_ckpt = os.path.join(model.model_run_dir, config.SUPERVISED_CHECKPOINT_FILENAME)
    clm_ckpt = os.path.join(model.model_run_dir, config.CLM_PRETRAIN_CHECKPOINT_FILENAME)

    if os.path.exists(best_sup_ckpt):
        print(f"Loading best supervised checkpoint: {best_sup_ckpt}")
        start_epoch = model.load_checkpoint(config.BEST_MODEL_SUPERVISED_CHECKPOINT_FILENAME)
    elif os.path.exists(reg_sup_ckpt):
        print(f"Loading regular supervised checkpoint: {reg_sup_ckpt}")
        start_epoch = model.load_checkpoint(config.SUPERVISED_CHECKPOINT_FILENAME)
    elif os.path.exists(clm_ckpt): # Fallback: if no supervised, but CLM exists, start fine-tuning
        print(f"No supervised checkpoint found. Loading CLM pretrained model from {clm_ckpt} as starting point for fine-tuning.")
        model.load_checkpoint(config.CLM_PRETRAIN_CHECKPOINT_FILENAME) 
        start_epoch = 0 # Reset epoch count for the supervised phase
        # Optimizer/Scheduler already re-initialized above for fine-tuning.
        # Clear supervised-specific history in the model if starting fresh fine-tune
        model.history['supervised_train_epoch_loss'] = []
        model.history['supervised_validation_loss'] = []
        model.history['supervised_validation_bleu'] = []
    else: # No checkpoints at all, model is fresh (should not happen if CLM ran)
        print("No checkpoints found at all. Starting supervised training from scratch (or current model state if CLM didn't run).")
        start_epoch = 0


    if start_epoch == 0: 
        print("Starting supervised training from epoch 0."); model.current_phase_epoch = 0
        # Ensure history is clean for this phase if truly starting from epoch 0 of supervised
        if not model.history.get('supervised_train_epoch_loss'): model.history['supervised_train_epoch_loss'] = []
        if not model.history.get('supervised_validation_loss'): model.history['supervised_validation_loss'] = []
        if not model.history.get('supervised_validation_bleu'): model.history['supervised_validation_bleu'] = []

    else: 
        # load_checkpoint returns next epoch to run, so last completed is start_epoch - 1
        # model.current_phase_epoch should already be set by load_checkpoint correctly
        print(f"Resuming supervised training. Last completed epoch: {model.current_phase_epoch}. Next epoch: {start_epoch}.")


    global_sup_step = model.current_phase_epoch * len(train_dataloader) # global_step for this phase
    best_val_loss = min((h[1] for h in model.history.get('supervised_validation_loss', []) if h), default=float('inf'))

    # The loop should go from the actual 0-indexed epoch to start from, up to total epochs
    for epoch_idx_0_based in range(model.current_phase_epoch, config.SUPERVISED_EPOCHS):
        epoch_display = epoch_idx_0_based + 1 # For logging (1-indexed)
        print(f"Supervised Code Epoch {epoch_display}/{config.SUPERVISED_EPOCHS}");
        if torch.cuda.is_available(): torch.cuda.empty_cache(); gc.collect()
        model.train(); epoch_loss_sum = 0; num_batches = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Supervised Epoch {epoch_display}", leave=False, ncols=120)
        for batch in progress_bar:
            src, dec_in, lbl = batch['src_token_ids'].to(device,non_blocking=True), batch['decoder_input_ids'].to(device,non_blocking=True), batch['label_ids'].to(device,non_blocking=True)
            model.optimizer.zero_grad(set_to_none=True)
            logits = model(src, dec_in)
            loss = model.criterion(logits.view(-1, model.vocab_size), lbl.view(-1))
            if not torch.isnan(loss): 
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); model.optimizer.step()
                if model.scheduler: model.scheduler.step()
                epoch_loss_sum += loss.item(); num_batches +=1
                current_lr_sup = model.optimizer.param_groups[0]['lr']
                writer.add_scalar('Supervised/Batch_Loss', loss.item(), global_sup_step)
                writer.add_scalar('Supervised/Learning_Rate', current_lr_sup, global_sup_step)
            global_sup_step += 1
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}", "LR":f"{current_lr_sup:.2e}"})

            if val_dataloader and global_sup_step > 0 and global_sup_step % config.SUPERVISED_VALIDATE_EVERY_N_BATCHES == 0:
                avg_val_loss, _ = evaluate_model(model, val_dataloader, device, writer, global_sup_step, "Supervised_Validation")
                writer.add_scalar('Supervised_Validation/Epoch_Loss', avg_val_loss, epoch_display) # Log with 1-indexed epoch
                
                # Update history ensuring no duplicate epoch entries
                model.history['supervised_validation_loss'] = [(e,v) for e,v in model.history.get('supervised_validation_loss',[]) if e != epoch_display]
                model.history['supervised_validation_loss'].append((epoch_display, avg_val_loss))
                model.history['supervised_validation_loss'].sort(key=lambda x:x[0])

                if avg_val_loss < best_val_loss: 
                    best_val_loss = avg_val_loss; print(f"New best val_loss: {best_val_loss:.4f}. Saving.")
                    model.save_checkpoint(epoch_display, config.BEST_MODEL_SUPERVISED_CHECKPOINT_FILENAME)
                model.train()
        progress_bar.close()
        avg_epoch_train_loss = epoch_loss_sum / num_batches if num_batches > 0 else float('inf')
        writer.add_scalar('Supervised/Train_Epoch_Loss', avg_epoch_train_loss, epoch_display)
        
        model.history['supervised_train_epoch_loss'] = [(e,v) for e,v in model.history.get('supervised_train_epoch_loss',[]) if e != epoch_display]
        model.history['supervised_train_epoch_loss'].append((epoch_display, avg_epoch_train_loss))
        model.history['supervised_train_epoch_loss'].sort(key=lambda x:x[0])

        if epoch_display % config.SUPERVISED_SAVE_EVERY_N_EPOCHS == 0: 
            model.save_checkpoint(epoch_display, config.SUPERVISED_CHECKPOINT_FILENAME)
        
        model.current_phase_epoch = epoch_idx_0_based + 1 # Update model's internal tracker of completed epochs for this phase
        gc.collect()

    print("--- Supervised Code Training Phase Completed ---")
    # Save final model based on the last completed epoch for this phase
    model.save_checkpoint(model.current_phase_epoch, config.SUPERVISED_CHECKPOINT_FILENAME)