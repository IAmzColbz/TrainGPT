# dataset_loader.py
import pandas as pd
import pyarrow.parquet as pq
import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from tokenizer_wrapper import global_tokenizer, TokenizerWrapper # Ensure TokenizerWrapper is importable for workers
import config
from utils import extract_python_code_from_markdown
from tqdm import tqdm
import gc
import struct
import traceback
import random
from multiprocessing import Pool, cpu_count

# --- Worker function for multiprocessing ---
WORKER_TOKENIZER_INSTANCE = None # Use a more descriptive name
WORKER_MAX_SEQ_LENGTH_FOR_PREFIXLM = None
WORKER_MIN_LEN_FOR_SPLIT_PREFIXLM = None
WORKER_MIN_PART_LEN_PREFIXLM = None

def init_worker_for_prefix_lm(tokenizer_path_str, max_seq_len, min_len_split, min_part_len):
    """Initializes tokenizer and params for each worker process for PrefixLM."""
    global WORKER_TOKENIZER_INSTANCE, WORKER_MAX_SEQ_LENGTH_FOR_PREFIXLM, \
           WORKER_MIN_LEN_FOR_SPLIT_PREFIXLM, WORKER_MIN_PART_LEN_PREFIXLM
    
    # print(f"Worker {os.getpid()} initializing tokenizer from: {tokenizer_path_str}") # For debugging
    try:
        WORKER_TOKENIZER_INSTANCE = TokenizerWrapper(model_path=tokenizer_path_str)
        WORKER_MAX_SEQ_LENGTH_FOR_PREFIXLM = max_seq_len
        WORKER_MIN_LEN_FOR_SPLIT_PREFIXLM = min_len_split
        WORKER_MIN_PART_LEN_PREFIXLM = min_part_len
    except Exception as e:
        print(f"WORKER ERROR: Failed to initialize tokenizer in worker {os.getpid()}: {e}")
        # WORKER_TOKENIZER_INSTANCE will remain None, process_document_for_prefix_lm should handle this
        # Or, raise an error to stop the pool if tokenizer is critical
        raise e # Reraise to stop the pool if tokenizer init fails

def process_document_for_prefix_lm(doc_str):
    """Processes a single document string to generate prefix/suffix sub-chunks."""
    if not doc_str or not isinstance(doc_str, str) or WORKER_TOKENIZER_INSTANCE is None:
        return [] 

    doc_tokens = WORKER_TOKENIZER_INSTANCE.encode(doc_str, add_bos=False, add_eos=False)
    
    if len(doc_tokens) < WORKER_MIN_LEN_FOR_SPLIT_PREFIXLM:
        return []

    # Ensure min_part_len is sensible
    if len(doc_tokens) - WORKER_MIN_PART_LEN_PREFIXLM <= WORKER_MIN_PART_LEN_PREFIXLM :
        return []
    split_point = random.randint(WORKER_MIN_PART_LEN_PREFIXLM, len(doc_tokens) - WORKER_MIN_PART_LEN_PREFIXLM)
    
    prefix_toks = doc_tokens[:split_point]
    suffix_toks = doc_tokens[split_point:]

    prefix_toks = prefix_toks[-WORKER_MAX_SEQ_LENGTH_FOR_PREFIXLM:]
    suffix_toks = suffix_toks[:WORKER_MAX_SEQ_LENGTH_FOR_PREFIXLM]
    
    if not prefix_toks or not suffix_toks or len(suffix_toks) < 2:
        return []
        
    return [(len(prefix_toks), prefix_toks, len(suffix_toks), suffix_toks)]


# --- ProblemSolutionDataset and collate_problem_solution_batch (Unchanged) ---
class ProblemSolutionDataset(Dataset):
    def __init__(self, problem_statements, gold_solutions, tokenizer):
        self.problems = problem_statements; self.solutions = gold_solutions; self.tokenizer = tokenizer
    def __len__(self): return len(self.problems)
    def __getitem__(self, idx):
        problem, solution = self.problems[idx], self.solutions[idx]
        src_token_ids = self.tokenizer.encode(problem, add_bos=True, add_eos=True, max_length=config.MAX_PROBLEM_STATEMENT_TOKENS)
        tgt_token_ids_raw = self.tokenizer.encode(solution, add_bos=True, add_eos=True, max_length=config.MAX_GOLD_SOLUTION_TOKENS)
        decoder_input_ids, label_ids = tgt_token_ids_raw[:-1], tgt_token_ids_raw[1:]
        return {"src_token_ids": torch.tensor(src_token_ids, dtype=torch.long), 
                "decoder_input_ids": torch.tensor(decoder_input_ids, dtype=torch.long), 
                "label_ids": torch.tensor(label_ids, dtype=torch.long)}

def collate_problem_solution_batch(batch):
    src_ids_list = [item['src_token_ids'] for item in batch]
    decoder_input_ids_list = [item['decoder_input_ids'] for item in batch]
    label_ids_list = [item['label_ids'] for item in batch]
    padded_src_ids = torch.nn.utils.rnn.pad_sequence(src_ids_list, batch_first=True, padding_value=config.PAD_TOKEN_ID)
    padded_decoder_input_ids = torch.nn.utils.rnn.pad_sequence(decoder_input_ids_list, batch_first=True, padding_value=config.PAD_TOKEN_ID)
    padded_label_ids = torch.nn.utils.rnn.pad_sequence(label_ids_list, batch_first=True, padding_value=config.PAD_TOKEN_ID)
    max_len_dec = padded_decoder_input_ids.size(1); max_len_lbl = padded_label_ids.size(1); target_len = max(max_len_dec, max_len_lbl)
    if max_len_dec < target_len: padded_decoder_input_ids = torch.cat([padded_decoder_input_ids, torch.full((padded_decoder_input_ids.size(0), target_len - max_len_dec), config.PAD_TOKEN_ID, dtype=torch.long, device=padded_decoder_input_ids.device)], dim=1)
    elif max_len_dec > target_len: padded_decoder_input_ids = padded_decoder_input_ids[:, :target_len]
    if max_len_lbl < target_len: padded_label_ids = torch.cat([padded_label_ids, torch.full((padded_label_ids.size(0), target_len - max_len_lbl), config.PAD_TOKEN_ID, dtype=torch.long, device=padded_label_ids.device)], dim=1)
    elif max_len_lbl > target_len: padded_label_ids = padded_label_ids[:, :target_len]
    return {"src_token_ids": padded_src_ids, "decoder_input_ids": padded_decoder_input_ids, "label_ids": padded_label_ids}


class FineWebPrefixLMDataset(Dataset):
    def __init__(self, parquet_file_path, 
                 tokenizer_model_path_for_worker, # Explicitly named for clarity
                 max_prefix_suffix_len=config.MAX_TEXT_PRETRAIN_SEQ_LEN,
                 start_doc_offset=0,
                 num_docs_to_process_in_chunk=config.DOCS_CHUNK_SIZE_PER_PREFIXLM_EPOCH,
                 cache_dir_base_with_run_name="./dataset_cache/default_run/fineweb_prefix_lm",
                 min_len_for_split=40,
                 num_workers_caching=max(1, cpu_count() // 2)):
        
        self.parquet_file_path = parquet_file_path
        self.main_process_tokenizer = global_tokenizer # For __getitem__ if needed after pickling
        self.tokenizer_model_path_for_worker = tokenizer_model_path_for_worker
        self.max_part_len = max_prefix_suffix_len
        self.start_doc_offset = start_doc_offset
        self.num_docs_to_process_in_chunk = num_docs_to_process_in_chunk
        self.min_len_for_split = min_len_for_split
        self.num_workers_caching = num_workers_caching if num_workers_caching > 0 else 1 # Ensure at least 1 worker
        
        file_basename = os.path.basename(self.parquet_file_path).replace('.parquet', '')
        self.cache_dir_for_file = os.path.join(cache_dir_base_with_run_name, file_basename) 
        os.makedirs(self.cache_dir_for_file, exist_ok=True)
        
        cache_chunk_id = f"prefixlm_s{start_doc_offset}_n{num_docs_to_process_in_chunk}_mpl{max_prefix_suffix_len}"
        self.cache_data_file = os.path.join(self.cache_dir_for_file, f"{cache_chunk_id}.dat")
        self.cache_index_file = os.path.join(self.cache_dir_for_file, f"{cache_chunk_id}.idx")

        self.num_examples = 0
        self.example_metadata = []
        self.data_file_handle = None

        if not self._load_index_from_cache():
            print(f"PrefixLM Cache not found for {os.path.basename(parquet_file_path)} chunk {cache_chunk_id}. Building with {self.num_workers_caching} workers...")
            self._build_and_write_cache_prefix_lm_mp()
            if not self._load_index_from_cache() or self.num_examples == 0:
                print(f"Warning: No PrefixLM examples built/loaded for {os.path.basename(parquet_file_path)} chunk {cache_chunk_id}.")
        
        if self.num_examples > 0:
            print(f"FineWebPrefixLMDataset: {os.path.basename(parquet_file_path)} chunk {cache_chunk_id}. Found {self.num_examples} examples.")

    def _build_and_write_cache_prefix_lm_mp(self):
        print(f"Building PrefixLM cache (MP): {os.path.basename(self.cache_data_file)}")
        current_offset_bytes = 0; temp_metadata = []
        processed_docs_count_total = 0; docs_skipped_before_chunk = 0
        min_part_len_for_worker = max(1, self.min_len_for_split // 2)

        try:
            # Use context manager for Pool
            with Pool(processes=self.num_workers_caching, 
                      initializer=init_worker_for_prefix_lm, # Corrected initializer name
                      initargs=(self.tokenizer_model_path_for_worker, # Pass string path
                                self.max_part_len, 
                                self.min_len_for_split, 
                                min_part_len_for_worker)) as pool, \
                 open(self.cache_data_file, 'wb') as data_f: # Open file here

                parquet_file = pq.ParquetFile(self.parquet_file_path)
                pbar_rg = tqdm(range(parquet_file.num_row_groups), desc="Caching PrefixLM RGs (MP)", unit="rg", ncols=100, smoothing=0.1)
                
                for rg_idx in pbar_rg:
                    if processed_docs_count_total >= self.num_docs_to_process_in_chunk: break
                    rg_meta = parquet_file.metadata.row_group(rg_idx); rows_in_rg = rg_meta.num_rows
                    if docs_skipped_before_chunk + rows_in_rg <= self.start_doc_offset:
                        docs_skipped_before_chunk += rows_in_rg; continue
                    
                    table = parquet_file.read_row_group(rg_idx, columns=['text']); text_col = table.column('text')
                    docs_in_rg_to_process_py_strings = [] # Collect Python strings for this row group
                    
                    for doc_idx_rg in range(len(text_col)):
                        global_doc_idx = docs_skipped_before_chunk + doc_idx_rg
                        if global_doc_idx < self.start_doc_offset: continue
                        if processed_docs_count_total >= self.num_docs_to_process_in_chunk: break
                        
                        arrow_scalar = text_col[doc_idx_rg]
                        if not arrow_scalar.is_valid: continue
                        doc_str = arrow_scalar.as_py()
                        if not doc_str: continue
                        docs_in_rg_to_process_py_strings.append(doc_str)
                        # We increment processed_docs_count_total only after successful processing by workers
                        # Or, if we decide to process all strings sent to workers, increment here.
                        # Let's count it as "sent to worker" here for tqdm display.
                        # The actual number of *examples* comes from worker results.
                        # processed_docs_count_total +=1 # This was causing overcounting if worker returned []

                    if not docs_in_rg_to_process_py_strings:
                        del table, text_col; gc.collect(); continue

                    # Process this batch of Python strings in parallel
                    # map_async might be better for tqdm updates, but map is simpler for now
                    list_of_worker_results = pool.map(process_document_for_prefix_lm, docs_in_rg_to_process_py_strings)
                    
                    processed_docs_count_total += len(docs_in_rg_to_process_py_strings) # Update after sending to pool

                    for result_list_for_doc in list_of_worker_results:
                        for len_p, prefix_toks, len_s, suffix_toks in result_list_for_doc:
                            data_f.write(struct.pack('<I', len_p))
                            if len_p > 0: data_f.write(struct.pack(f'<{len_p}I', *prefix_toks))
                            data_f.write(struct.pack('<I', len_s))
                            if len_s > 0: data_f.write(struct.pack(f'<{len_s}I', *suffix_toks))
                            temp_metadata.append((current_offset_bytes, len_p, len_s))
                            current_offset_bytes += 4 + (len_p * 4) + 4 + (len_s * 4)
                    
                    del table, text_col, docs_in_rg_to_process_py_strings, list_of_worker_results; gc.collect()
                    pbar_rg.set_postfix_str(f"{processed_docs_count_total}/{self.num_docs_to_process_in_chunk} docs, {len(temp_metadata)} ex.")
                    if processed_docs_count_total >= self.num_docs_to_process_in_chunk: break
                pbar_rg.close()

            with open(self.cache_index_file, 'wb') as idx_f:
                idx_f.write(struct.pack('<Q', len(temp_metadata)))
                for offset, lp, ls in temp_metadata: idx_f.write(struct.pack('<QII', offset, lp, ls))
            print(f"PrefixLM Cache (MP): {len(temp_metadata)} examples from {processed_docs_count_total} docs.")
        except Exception as e: print(f"Err PrefixLM Cache (MP): {e}"); traceback.print_exc(); self._cleanup_failed_cache()
        finally: 
            if 'parquet_file' in locals() and parquet_file is not None: del parquet_file
            gc.collect()

    def _cleanup_failed_cache(self):
        if os.path.exists(self.cache_data_file): os.remove(self.cache_data_file)
        if os.path.exists(self.cache_index_file): os.remove(self.cache_index_file)
        print(f"Cleaned up cache files for {os.path.basename(self.cache_data_file)}")

    def _load_index_from_cache(self):
        if not os.path.exists(self.cache_data_file) or not os.path.exists(self.cache_index_file): return False
        try:
            with open(self.cache_index_file, 'rb') as index_f:
                self.num_examples = struct.unpack('<Q', index_f.read(8))[0]
                for _ in range(self.num_examples): self.example_metadata.append(struct.unpack('<QII', index_f.read(16))) 
            if self.num_examples > 0: self.data_file_handle = open(self.cache_data_file, 'rb')
            return True
        except Exception as e: print(f"Error loading PrefixLM cache index: {e}"); self._cleanup_failed_cache(); return False

    def __len__(self): return self.num_examples

    def __getitem__(self, idx):
        if not self.data_file_handle or self.data_file_handle.closed:
            try: self.data_file_handle = open(self.cache_data_file, 'rb')
            except Exception as e: raise RuntimeError(f"Failed to re-open cache data file in __getitem__: {e}")
        if idx >= self.num_examples: raise IndexError("Index out of bounds.")
        
        offset, len_prefix, len_suffix = self.example_metadata[idx]
        self.data_file_handle.seek(offset)
        stored_len_p = struct.unpack('<I', self.data_file_handle.read(4))[0]
        if stored_len_p != len_prefix: raise RuntimeError(f"PrefixLM Cache integrity: prefix len mismatch. Index: {len_prefix}, Stored: {stored_len_p} at offset {offset}")
        prefix_tokens = list(struct.unpack(f'<{len_prefix}I', self.data_file_handle.read(len_prefix * 4))) if len_prefix > 0 else []
        stored_len_s = struct.unpack('<I', self.data_file_handle.read(4))[0]
        if stored_len_s != len_suffix: raise RuntimeError(f"PrefixLM Cache integrity: suffix len mismatch. Index: {len_suffix}, Stored: {stored_len_s}")
        suffix_tokens = list(struct.unpack(f'<{len_suffix}I', self.data_file_handle.read(len_suffix * 4))) if len_suffix > 0 else []

        if not suffix_tokens or len(suffix_tokens) < 1:
             return {"src_token_ids": torch.tensor([self.main_process_tokenizer.bos_id, config.PAD_TOKEN_ID, self.main_process_tokenizer.eos_id], dtype=torch.long), 
                     "decoder_input_ids": torch.tensor([self.main_process_tokenizer.bos_id, self.main_process_tokenizer.eos_id], dtype=torch.long), 
                     "label_ids": torch.tensor([self.main_process_tokenizer.eos_id, config.PAD_TOKEN_ID], dtype=torch.long)}
        
        src_ids = [self.main_process_tokenizer.bos_id] + prefix_tokens + [self.main_process_tokenizer.eos_id]
        if len(suffix_tokens) == 1: 
             decoder_input_ids = [self.main_process_tokenizer.bos_id]
             label_ids = suffix_tokens 
        else: 
             decoder_input_ids = [self.main_process_tokenizer.bos_id] + suffix_tokens[:-1]
             label_ids = suffix_tokens
        return {"src_token_ids": torch.tensor(src_ids, dtype=torch.long), 
                "decoder_input_ids": torch.tensor(decoder_input_ids, dtype=torch.long), 
                "label_ids": torch.tensor(label_ids, dtype=torch.long)}
    
    def __del__(self):
        if hasattr(self, 'data_file_handle') and self.data_file_handle:
            try: self.data_file_handle.close()
            except: pass


def collate_prefix_lm_batch(batch):
    # ... (This function remains the same as provided in the previous "full code" response) ...
    src_ids_list = [item['src_token_ids'] for item in batch]; decoder_input_ids_list = [item['decoder_input_ids'] for item in batch]; label_ids_list = [item['label_ids'] for item in batch]
    padded_src_ids = torch.nn.utils.rnn.pad_sequence(src_ids_list, batch_first=True, padding_value=config.PAD_TOKEN_ID)
    padded_decoder_input_ids = torch.nn.utils.rnn.pad_sequence(decoder_input_ids_list, batch_first=True, padding_value=config.PAD_TOKEN_ID)
    padded_label_ids = torch.nn.utils.rnn.pad_sequence(label_ids_list, batch_first=True, padding_value=config.PAD_TOKEN_ID)
    max_len_decoder_side = padded_decoder_input_ids.size(1)
    if padded_label_ids.size(1) < max_len_decoder_side: padded_label_ids = torch.cat([padded_label_ids, torch.full((padded_label_ids.size(0), max_len_decoder_side - padded_label_ids.size(1)), config.PAD_TOKEN_ID, dtype=torch.long, device=padded_label_ids.device)], dim=1)
    elif padded_label_ids.size(1) > max_len_decoder_side: padded_label_ids = padded_label_ids[:, :max_len_decoder_side]
    return {"src_token_ids": padded_src_ids, "decoder_input_ids": padded_decoder_input_ids, "label_ids": padded_label_ids}


def get_dataloaders(task_type: str, tokenizer, batch_size, 
                    val_split_ratio=config.SUPERVISED_VALIDATION_SPLIT_RATIO, 
                    specific_file_path: str = None,
                    start_doc_offset: int = 0, 
                    num_docs_in_chunk: int = config.DOCS_CHUNK_SIZE_PER_PREFIXLM_EPOCH,
                    model_run_name_for_cache: str = "default_run"):
    # ... (Supervised code part unchanged) ...
    full_dataset = None; collate_fn = None; num_examples_from_chunk = 0
    if task_type == "supervised_code":
        problems_list, solutions_list = [], []
        dataset_abs_dir = config.CODE_DATASET_DIR; search_pattern = os.path.join(dataset_abs_dir, "train-*-of-*.parquet"); file_paths = sorted(glob.glob(search_pattern))
        if not file_paths: print(f"No Problem-Solution Parquet files found in {dataset_abs_dir}.")
        else:
            files_to_load = file_paths[:config.NUM_CODE_DATASET_FILES_TO_LOAD] if hasattr(config, 'NUM_CODE_DATASET_FILES_TO_LOAD') and config.NUM_CODE_DATASET_FILES_TO_LOAD > 0 else file_paths
            print(f"Loading up to {len(files_to_load)} Problem-Solution dataset files.")
            for filepath_idx, filepath in enumerate(tqdm(files_to_load, desc="Loading Code Datasets", unit="file")):
                try:
                    df = pd.read_parquet(filepath);
                    for _, row in df.iterrows():
                        problem, solution_raw = row.get('problem_statement', ''), row.get('gold_standard_solution', '')
                        solution = extract_python_code_from_markdown(solution_raw)
                        if problem and solution: problems_list.append(problem.strip()); solutions_list.append(solution.strip())
                except Exception as e: print(f"Error loading {filepath}: {e}")
            if not problems_list: print("No problem-solution pairs loaded.")
            else: print(f"Loaded {len(problems_list)} problem-solution pairs."); full_dataset = ProblemSolutionDataset(problems_list, solutions_list, tokenizer)
        collate_fn = collate_problem_solution_batch
    elif task_type == "prefix_lm_pretrain":
        if not specific_file_path or not os.path.exists(specific_file_path):
            print(f"ERROR: File path invalid for PrefixLM: {specific_file_path}"); return None, None, 0
        if not model_run_name_for_cache: 
             raise ValueError("model_run_name_for_cache must be provided for PrefixLM task type for caching.")
        
        cache_base = config.CACHE_DIR_PREFIXLM_BASE.format(model_run_name=model_run_name_for_cache)
        full_dataset = FineWebPrefixLMDataset(
            specific_file_path, 
            tokenizer_model_path_for_worker=config.TOKENIZER_MODEL_PATH,
            max_prefix_suffix_len=config.MAX_TEXT_PRETRAIN_SEQ_LEN,
            start_doc_offset=start_doc_offset,
            num_docs_to_process_in_chunk=num_docs_in_chunk,
            cache_dir_base_with_run_name=cache_base
        )
        num_examples_from_chunk = len(full_dataset)
        if num_examples_from_chunk == 0: return None, None, 0
        collate_fn = collate_prefix_lm_batch; val_split_ratio = 0
    else: raise ValueError(f"Unknown task_type: {task_type}")

    if not full_dataset: return None, None, 0 if task_type == "prefix_lm_pretrain" else None # Adjusted for 3-tuple return
    
    train_dataset, val_dataset = None, None
    if val_split_ratio > 0 and len(full_dataset) > 1:
        val_size = int(len(full_dataset) * val_split_ratio); train_size = len(full_dataset) - val_size
        if train_size <= 0 or val_size <= 0 : print(f"Dataset too small. Using all for training."); train_dataset = full_dataset
        else: train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size]); print(f"Train: {len(train_dataset)}, Val: {len(val_dataset) if val_dataset else 0}")
    else:
        train_dataset = full_dataset
        if len(full_dataset)>0: print(f"Using all {len(full_dataset)} samples for training.")
    
    if not train_dataset or len(train_dataset) == 0 : print("ERROR: Training dataset empty."); return None, None, 0 if task_type == "prefix_lm_pretrain" else None

    dl_num_workers = 0 # Start with 0 for stability, increase later if needed
    # if torch.cuda.is_available() and not sys.platform == 'win32': dl_num_workers = min(4, cpu_count() // 2)


    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=dl_num_workers, pin_memory=torch.cuda.is_available())
    val_dl = None
    if val_dataset and len(val_dataset) > 0: val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=dl_num_workers, pin_memory=torch.cuda.is_available())
    
    if task_type == "prefix_lm_pretrain": return train_dl, val_dl, num_examples_from_chunk
    else: return train_dl, val_dl