# dataset_loader.py
import pandas as pd
import pyarrow.parquet as pq
import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from tokenizer_wrapper import global_tokenizer, TokenizerWrapper 
import config
from utils import extract_python_code_from_markdown
from tqdm import tqdm
import gc
import struct
import traceback
import random
from multiprocessing import Pool, cpu_count
import logging

logger = logging.getLogger(__name__)

# --- Worker function for multiprocessing (init_worker_for_prefix_lm, process_document_for_prefix_lm) ---
# These should be okay from the previous refactor, ensure WORKER_TOKENIZER_INSTANCE uses TokenizerWrapper
WORKER_TOKENIZER_INSTANCE = None
WORKER_MAX_SEQ_LENGTH_FOR_PREFIXLM = None
WORKER_MIN_LEN_FOR_SPLIT_PREFIXLM = None
WORKER_MIN_PART_LEN_PREFIXLM = None

def init_worker_for_prefix_lm(tokenizer_path_str, max_seq_len, min_len_split, min_part_len):
    global WORKER_TOKENIZER_INSTANCE, WORKER_MAX_SEQ_LENGTH_FOR_PREFIXLM, \
           WORKER_MIN_LEN_FOR_SPLIT_PREFIXLM, WORKER_MIN_PART_LEN_PREFIXLM
    logger.debug(f"Worker {os.getpid()} initializing tokenizer from: {tokenizer_path_str}")
    try:
        WORKER_TOKENIZER_INSTANCE = TokenizerWrapper(model_path=tokenizer_path_str)
        WORKER_MAX_SEQ_LENGTH_FOR_PREFIXLM = max_seq_len
        WORKER_MIN_LEN_FOR_SPLIT_PREFIXLM = min_len_split
        WORKER_MIN_PART_LEN_PREFIXLM = min_part_len
    except Exception as e:
        logger.error(f"WORKER ERROR: Failed to initialize tokenizer in worker {os.getpid()}: {e}")
        raise 

def process_document_for_prefix_lm(doc_str):
    if not doc_str or not isinstance(doc_str, str) or WORKER_TOKENIZER_INSTANCE is None:
        return [] 
    doc_tokens = WORKER_TOKENIZER_INSTANCE.encode(doc_str, add_bos=False, add_eos=False)
    if len(doc_tokens) < WORKER_MIN_LEN_FOR_SPLIT_PREFIXLM or \
       len(doc_tokens) < 2 * WORKER_MIN_PART_LEN_PREFIXLM: 
        return []
    if WORKER_MIN_PART_LEN_PREFIXLM > len(doc_tokens) - WORKER_MIN_PART_LEN_PREFIXLM:
        return [] 
    split_point = random.randint(WORKER_MIN_PART_LEN_PREFIXLM, len(doc_tokens) - WORKER_MIN_PART_LEN_PREFIXLM)
    prefix_toks = doc_tokens[:split_point][-WORKER_MAX_SEQ_LENGTH_FOR_PREFIXLM:]
    suffix_toks = doc_tokens[split_point:][:WORKER_MAX_SEQ_LENGTH_FOR_PREFIXLM]
    if not prefix_toks or not suffix_toks: return []
    return [(len(prefix_toks), prefix_toks, len(suffix_toks), suffix_toks)]

# --- ProblemSolutionDataset and collate_problem_solution_batch ---
# (Assumed okay from previous refactor, ensure __getitem__ and collate_fn are robust)
class ProblemSolutionDataset(Dataset):
    def __init__(self, problem_statements, gold_solutions, tokenizer):
        self.problems = problem_statements
        self.solutions = gold_solutions
        self.tokenizer = tokenizer
    def __len__(self): return len(self.problems)
    def __getitem__(self, idx):
        problem, solution = self.problems[idx], self.solutions[idx]
        src_token_ids = self.tokenizer.encode(problem, add_bos=True, add_eos=True, max_length=config.MAX_PROBLEM_STATEMENT_TOKENS)
        tgt_token_ids_raw = self.tokenizer.encode(solution, add_bos=True, add_eos=True, max_length=config.MAX_GOLD_SOLUTION_TOKENS)
        decoder_input_ids = tgt_token_ids_raw[:-1] 
        label_ids = tgt_token_ids_raw[1:]
        return {
            "src_token_ids": torch.tensor(src_token_ids, dtype=torch.long), 
            "decoder_input_ids": torch.tensor(decoder_input_ids, dtype=torch.long), 
            "label_ids": torch.tensor(label_ids, dtype=torch.long)
        }

def collate_problem_solution_batch(batch):
    src_ids_list = [item['src_token_ids'] for item in batch]
    decoder_input_ids_list = [item['decoder_input_ids'] for item in batch]
    label_ids_list = [item['label_ids'] for item in batch]
    padded_src_ids = torch.nn.utils.rnn.pad_sequence(src_ids_list, batch_first=True, padding_value=config.PAD_TOKEN_ID)
    padded_decoder_input_ids = torch.nn.utils.rnn.pad_sequence(decoder_input_ids_list, batch_first=True, padding_value=config.PAD_TOKEN_ID)
    padded_label_ids = torch.nn.utils.rnn.pad_sequence(label_ids_list, batch_first=True, padding_value=config.PAD_TOKEN_ID)
    max_len_dec = padded_decoder_input_ids.size(1); max_len_lbl = padded_label_ids.size(1); target_len = max(max_len_dec, max_len_lbl)
    if max_len_dec < target_len:
        padding_tensor = torch.full((padded_decoder_input_ids.size(0), target_len - max_len_dec), config.PAD_TOKEN_ID, dtype=torch.long, device=padded_decoder_input_ids.device)
        padded_decoder_input_ids = torch.cat([padded_decoder_input_ids, padding_tensor], dim=1)
    elif max_len_dec > target_len: 
        padded_decoder_input_ids = padded_decoder_input_ids[:, :target_len]
    if max_len_lbl < target_len:
        padding_tensor = torch.full((padded_label_ids.size(0), target_len - max_len_lbl), config.PAD_TOKEN_ID, dtype=torch.long, device=padded_label_ids.device)
        padded_label_ids = torch.cat([padded_label_ids, padding_tensor], dim=1)
    elif max_len_lbl > target_len: 
        padded_label_ids = padded_label_ids[:, :target_len]
    return {"src_token_ids": padded_src_ids, "decoder_input_ids": padded_decoder_input_ids, "label_ids": padded_label_ids}


# --- FineWebPrefixLMDataset ---
# (Assumed okay from previous refactor, ensure cache_dir_base_with_run_name uses template correctly)
class FineWebPrefixLMDataset(Dataset):
    def __init__(self, parquet_file_path, 
                 tokenizer_model_path_for_worker,
                 max_prefix_suffix_len=config.MAX_TEXT_PRETRAIN_SEQ_LEN,
                 start_doc_offset=0,
                 num_docs_to_process_in_chunk=config.DOCS_CHUNK_SIZE_PER_PREFIXLM_EPOCH,
                 # cache_dir_base_with_run_name is now the full path to the run's cache base
                 cache_dir_base_for_run: str ="./dataset_cache/default_run", # Changed name for clarity
                 min_len_for_split=config.MIN_LEN_FOR_PREFIXLM_SPLIT,
                 min_part_len=config.MIN_PART_LEN_FOR_PREFIXLM,
                 num_workers_caching=0):
        
        self.parquet_file_path = parquet_file_path
        self.main_process_tokenizer = global_tokenizer 
        self.tokenizer_model_path_for_worker = tokenizer_model_path_for_worker
        self.max_part_len = max_prefix_suffix_len
        self.start_doc_offset = start_doc_offset
        self.num_docs_to_process_in_chunk = num_docs_to_process_in_chunk
        self.min_len_for_split = min_len_for_split
        self.min_part_len = min_part_len
        
        if num_workers_caching <= 0: self.num_workers_caching = max(1, cpu_count() // 2)
        else: self.num_workers_caching = num_workers_caching
        
        file_basename = os.path.basename(self.parquet_file_path).replace('.parquet', '')
        # cache_dir_base_for_run is now expected to be like ".../dataset_cache/my_run_YYYYMMDD-HHMMSS"
        self.cache_dir_for_file = os.path.join(cache_dir_base_for_run, "fineweb_prefix_lm", file_basename) # Added "fineweb_prefix_lm" subfolder
        os.makedirs(self.cache_dir_for_file, exist_ok=True)
        
        cache_chunk_id = f"prefixlm_s{start_doc_offset}_n{num_docs_to_process_in_chunk}_mpl{max_prefix_suffix_len}_mls{min_len_for_split}_mp{min_part_len}"
        self.cache_data_file = os.path.join(self.cache_dir_for_file, f"{cache_chunk_id}.dat")
        self.cache_index_file = os.path.join(self.cache_dir_for_file, f"{cache_chunk_id}.idx")

        self.num_examples = 0
        self.example_metadata = [] 
        self.data_file_handle = None

        if not self._load_index_from_cache():
            logger.info(f"PrefixLM Cache not found or invalid for {os.path.basename(parquet_file_path)} chunk {cache_chunk_id}. Building with {self.num_workers_caching} workers...")
            self._build_and_write_cache_prefix_lm_mp()
            if not self._load_index_from_cache() or self.num_examples == 0:
                logger.warning(f"No PrefixLM examples built/loaded for {os.path.basename(parquet_file_path)} chunk {cache_chunk_id}.")
        
        if self.num_examples > 0:
            logger.info(f"FineWebPrefixLMDataset: {os.path.basename(parquet_file_path)} chunk {cache_chunk_id}. Found {self.num_examples} examples.")

    def _build_and_write_cache_prefix_lm_mp(self):
        logger.info(f"Building PrefixLM cache (MP): {os.path.basename(self.cache_data_file)}")
        current_offset_bytes = 0; temp_metadata = []; processed_docs_count_for_chunk = 0; docs_skipped_before_offset = 0
        pool_initargs = (self.tokenizer_model_path_for_worker, self.max_part_len, self.min_len_for_split, self.min_part_len)
        try:
            with Pool(processes=self.num_workers_caching, initializer=init_worker_for_prefix_lm, initargs=pool_initargs) as pool, \
                 open(self.cache_data_file, 'wb') as data_f:
                parquet_file = pq.ParquetFile(self.parquet_file_path)
                pbar_rg = tqdm(range(parquet_file.num_row_groups), desc=f"Caching PrefixLM RGs for {os.path.basename(self.parquet_file_path)}", unit="rg", ncols=100, smoothing=0.1)
                for rg_idx in pbar_rg:
                    if processed_docs_count_for_chunk >= self.num_docs_to_process_in_chunk: break 
                    rg_meta = parquet_file.metadata.row_group(rg_idx); rows_in_rg = rg_meta.num_rows
                    if docs_skipped_before_offset + rows_in_rg <= self.start_doc_offset:
                        docs_skipped_before_offset += rows_in_rg; continue
                    table = parquet_file.read_row_group(rg_idx, columns=['text']); text_column = table.column('text')
                    docs_to_process_in_rg_py_strings = []
                    for doc_idx_in_rg in range(len(text_column)):
                        current_global_doc_index = docs_skipped_before_offset + doc_idx_in_rg
                        if current_global_doc_index < self.start_doc_offset: continue 
                        if processed_docs_count_for_chunk >= self.num_docs_to_process_in_chunk: break 
                        arrow_scalar = text_column[doc_idx_in_rg]
                        if arrow_scalar.is_valid:
                            doc_str = arrow_scalar.as_py()
                            if doc_str and isinstance(doc_str, str):
                                docs_to_process_in_rg_py_strings.append(doc_str); processed_docs_count_for_chunk += 1 
                    del table, text_column; gc.collect()
                    if not docs_to_process_in_rg_py_strings: continue
                    list_of_worker_results = pool.map(process_document_for_prefix_lm, docs_to_process_in_rg_py_strings)
                    for single_doc_results in list_of_worker_results: 
                        for len_p, prefix_toks, len_s, suffix_toks in single_doc_results: 
                            data_f.write(struct.pack('<I', len_p))
                            if len_p > 0: data_f.write(struct.pack(f'<{len_p}I', *prefix_toks)) 
                            data_f.write(struct.pack('<I', len_s)) 
                            if len_s > 0: data_f.write(struct.pack(f'<{len_s}I', *suffix_toks)) 
                            temp_metadata.append((current_offset_bytes, len_p, len_s))
                            current_offset_bytes += 4 + (len_p * 4) + 4 + (len_s * 4) 
                    del list_of_worker_results; gc.collect()
                    pbar_rg.set_postfix_str(f"{processed_docs_count_for_chunk}/{self.num_docs_to_process_in_chunk} docs, {len(temp_metadata)} ex.")
                pbar_rg.close()
            with open(self.cache_index_file, 'wb') as idx_f:
                idx_f.write(struct.pack('<Q', len(temp_metadata))) 
                for offset, lp, ls in temp_metadata: idx_f.write(struct.pack('<QII', offset, lp, ls)) 
            logger.info(f"PrefixLM Cache (MP) for chunk completed: {len(temp_metadata)} examples from {processed_docs_count_for_chunk} documents.")
        except Exception as e: logger.error(f"Error during PrefixLM Cache build (MP): {e}\n{traceback.format_exc()}"); self._cleanup_failed_cache()
        finally: 
            if 'parquet_file' in locals() and parquet_file is not None: del parquet_file
            gc.collect()

    def _cleanup_failed_cache(self):
        try:
            if os.path.exists(self.cache_data_file): os.remove(self.cache_data_file)
            if os.path.exists(self.cache_index_file): os.remove(self.cache_index_file)
            logger.info(f"Cleaned up potentially corrupted cache files for {os.path.basename(self.cache_data_file)}")
        except OSError as e: logger.error(f"Error cleaning up cache files: {e}")

    def _load_index_from_cache(self):
        if not os.path.exists(self.cache_data_file) or not os.path.exists(self.cache_index_file): return False
        try:
            with open(self.cache_index_file, 'rb') as index_f:
                self.num_examples = struct.unpack('<Q', index_f.read(8))[0]; self.example_metadata = [] 
                for _ in range(self.num_examples): self.example_metadata.append(struct.unpack('<QII', index_f.read(16))) 
            if self.num_examples > 0:
                if self.data_file_handle and not self.data_file_handle.closed: self.data_file_handle.close()
                self.data_file_handle = open(self.cache_data_file, 'rb')
            return True
        except Exception as e: logger.error(f"Error loading PrefixLM cache index for {os.path.basename(self.cache_index_file)}: {e}"); self._cleanup_failed_cache(); return False

    def __len__(self): return self.num_examples
    def __getitem__(self, idx):
        if idx >= self.num_examples or idx < 0: raise IndexError(f"Index {idx} out of bounds for {self.num_examples} examples.")
        if not self.data_file_handle or self.data_file_handle.closed:
            try: logger.warning(f"Re-opening data_file_handle in __getitem__ for {os.path.basename(self.cache_data_file)}."); self.data_file_handle = open(self.cache_data_file, 'rb')
            except Exception as e: raise RuntimeError(f"Failed to re-open cache data file: {e}")
        offset, len_prefix, len_suffix = self.example_metadata[idx]; self.data_file_handle.seek(offset)
        stored_len_p_bytes = self.data_file_handle.read(4)
        if len(stored_len_p_bytes) < 4: raise RuntimeError(f"Cache integrity: Unexpected EOF reading prefix length.")
        stored_len_p = struct.unpack('<I', stored_len_p_bytes)[0]
        if stored_len_p != len_prefix: raise RuntimeError(f"Cache integrity: prefix length mismatch. Index: {len_prefix}, File: {stored_len_p}")
        prefix_tokens = list(struct.unpack(f'<{len_prefix}I', self.data_file_handle.read(len_prefix * 4))) if len_prefix > 0 else []
        stored_len_s_bytes = self.data_file_handle.read(4)
        if len(stored_len_s_bytes) < 4: raise RuntimeError(f"Cache integrity: Unexpected EOF reading suffix length.")
        stored_len_s = struct.unpack('<I', stored_len_s_bytes)[0]
        if stored_len_s != len_suffix: raise RuntimeError(f"Cache integrity: suffix length mismatch. Index: {len_suffix}, File: {stored_len_s}")
        suffix_tokens = list(struct.unpack(f'<{len_suffix}I', self.data_file_handle.read(len_suffix * 4))) if len_suffix > 0 else []
        if not suffix_tokens:
             logger.warning(f"Empty suffix for index {idx} in {os.path.basename(self.cache_data_file)}. Returning PAD example.")
             return {"src_token_ids": torch.tensor([self.main_process_tokenizer.bos_id, config.PAD_TOKEN_ID, self.main_process_tokenizer.eos_id], dtype=torch.long), 
                     "decoder_input_ids": torch.tensor([self.main_process_tokenizer.bos_id, self.main_process_tokenizer.eos_id], dtype=torch.long), 
                     "label_ids": torch.tensor([self.main_process_tokenizer.eos_id, config.PAD_TOKEN_ID], dtype=torch.long)}
        src_ids = [self.main_process_tokenizer.bos_id] + prefix_tokens + [self.main_process_tokenizer.eos_id]
        if len(suffix_tokens) == 1: decoder_input_ids = [self.main_process_tokenizer.bos_id]; label_ids = suffix_tokens 
        else: decoder_input_ids = [self.main_process_tokenizer.bos_id] + suffix_tokens[:-1]; label_ids = suffix_tokens
        return {"src_token_ids": torch.tensor(src_ids, dtype=torch.long), "decoder_input_ids": torch.tensor(decoder_input_ids, dtype=torch.long), "label_ids": torch.tensor(label_ids, dtype=torch.long)}
    def __del__(self):
        if hasattr(self, 'data_file_handle') and self.data_file_handle and not self.data_file_handle.closed:
            try: self.data_file_handle.close()
            except Exception as e: logger.error(f"Error closing data file handle for {getattr(self, 'cache_data_file', 'N/A')}: {e}")

# --- collate_prefix_lm_batch ---
# (Assumed okay from previous refactor)
def collate_prefix_lm_batch(batch):
    src_ids_list = [item['src_token_ids'] for item in batch]; decoder_input_ids_list = [item['decoder_input_ids'] for item in batch]; label_ids_list = [item['label_ids'] for item in batch]
    padded_src_ids = torch.nn.utils.rnn.pad_sequence(src_ids_list, batch_first=True, padding_value=config.PAD_TOKEN_ID)
    padded_decoder_input_ids = torch.nn.utils.rnn.pad_sequence(decoder_input_ids_list, batch_first=True, padding_value=config.PAD_TOKEN_ID)
    padded_label_ids = torch.nn.utils.rnn.pad_sequence(label_ids_list, batch_first=True, padding_value=config.PAD_TOKEN_ID)
    max_len_decoder_side = max(padded_decoder_input_ids.size(1), padded_label_ids.size(1))
    if padded_decoder_input_ids.size(1) < max_len_decoder_side:
        padding_tensor = torch.full((padded_decoder_input_ids.size(0), max_len_decoder_side - padded_decoder_input_ids.size(1)), config.PAD_TOKEN_ID, dtype=torch.long, device=padded_decoder_input_ids.device)
        padded_decoder_input_ids = torch.cat([padded_decoder_input_ids, padding_tensor], dim=1)
    if padded_label_ids.size(1) < max_len_decoder_side:
        padding_tensor = torch.full((padded_label_ids.size(0), max_len_decoder_side - padded_label_ids.size(1)), config.PAD_TOKEN_ID, dtype=torch.long, device=padded_label_ids.device)
        padded_label_ids = torch.cat([padded_label_ids, padding_tensor], dim=1)
    return {"src_token_ids": padded_src_ids, "decoder_input_ids": padded_decoder_input_ids, "label_ids": padded_label_ids}


def get_dataloaders(task_type: str, tokenizer, batch_size, 
                    val_split_ratio=config.SUPERVISED_VALIDATION_SPLIT_RATIO, 
                    specific_file_path: str = None, 
                    start_doc_offset: int = 0,      
                    num_docs_in_chunk: int = config.DOCS_CHUNK_SIZE_PER_PREFIXLM_EPOCH, 
                    model_run_name_for_cache: str = "default_run"):
    
    full_dataset = None; collate_fn = None; num_examples_from_chunk = 0

    if task_type == "supervised_code":
        logger.info("Loading supervised code dataset.")
        problems_list, solutions_list = [], []
        dataset_abs_dir = config.CODE_DATASET_DIR
        search_pattern = os.path.join(dataset_abs_dir, "train-*-of-*.parquet") # Standard pattern
        logger.info(f"Searching for supervised code dataset files in: {dataset_abs_dir} with pattern: {search_pattern}")
        all_file_paths = sorted(glob.glob(search_pattern))
        logger.info(f"Found {len(all_file_paths)} supervised code files initially: {all_file_paths if len(all_file_paths) < 10 else str(all_file_paths[:5])+'...'}")
        
        if not all_file_paths:
            logger.warning(f"No Problem-Solution Parquet files found in {dataset_abs_dir} matching pattern '{os.path.basename(search_pattern)}'.")
        else:
            files_to_load_count = config.NUM_CODE_DATASET_FILES_TO_LOAD
            if files_to_load_count <= 0: # Load all if 0 or negative
                files_to_load = all_file_paths
            else:
                files_to_load = all_file_paths[:files_to_load_count]
            
            logger.info(f"Attempting to load {len(files_to_load)} Problem-Solution dataset files based on NUM_CODE_DATASET_FILES_TO_LOAD={config.NUM_CODE_DATASET_FILES_TO_LOAD}.")
            
            for filepath in tqdm(files_to_load, desc="Loading Code Datasets", unit="file", ncols=100):
                try:
                    df = pd.read_parquet(filepath)
                    for _, row in df.iterrows():
                        problem, solution_raw = row.get('problem_statement', ''), row.get('gold_standard_solution', '')
                        solution = extract_python_code_from_markdown(solution_raw) 
                        if problem and solution: problems_list.append(problem.strip()); solutions_list.append(solution.strip())
                    del df; gc.collect()
                except Exception as e: logger.error(f"Error loading or processing supervised code file {filepath}: {e}")
            
            if not problems_list: logger.warning("No problem-solution pairs were loaded from supervised dataset files.")
            else: logger.info(f"Loaded {len(problems_list)} problem-solution pairs."); full_dataset = ProblemSolutionDataset(problems_list, solutions_list, tokenizer)
        collate_fn = collate_problem_solution_batch

    elif task_type == "prefix_lm_pretrain":
        logger.info(f"Loading PrefixLM dataset for file: {specific_file_path}, model_run_name for cache: {model_run_name_for_cache}")
        if not specific_file_path or not os.path.exists(specific_file_path):
            logger.error(f"ERROR: File path invalid or does not exist for PrefixLM: {specific_file_path}"); return None, None, 0
        if not model_run_name_for_cache or model_run_name_for_cache == "default_run": 
             logger.warning("model_run_name_for_cache is default or not set; cache might not be unique if multiple runs use default.")
        
        # Construct the base directory for caches related to this specific model run
        # config.CACHE_DIR_BASE_TEMPLATE is like ".../dataset_cache/{model_run_name}"
        run_specific_cache_base = config.CACHE_DIR_BASE_TEMPLATE.format(model_run_name=model_run_name_for_cache)
        logger.info(f"  PrefixLM cache base for this run: {run_specific_cache_base}")
        
        full_dataset = FineWebPrefixLMDataset(
            parquet_file_path=specific_file_path, 
            tokenizer_model_path_for_worker=config.TOKENIZER_MODEL_PATH,
            max_prefix_suffix_len=config.MAX_TEXT_PRETRAIN_SEQ_LEN,
            start_doc_offset=start_doc_offset,
            num_docs_to_process_in_chunk=num_docs_in_chunk,
            cache_dir_base_for_run=run_specific_cache_base, 
            min_len_for_split=config.MIN_LEN_FOR_PREFIXLM_SPLIT,
            min_part_len=config.MIN_PART_LEN_FOR_PREFIXLM
        )
        num_examples_from_chunk = len(full_dataset)
        if num_examples_from_chunk == 0: logger.warning(f"PrefixLM dataset for {specific_file_path} (chunk starting {start_doc_offset}) is empty.")
        collate_fn = collate_prefix_lm_batch; val_split_ratio = 0 
    else: raise ValueError(f"Unknown task_type for get_dataloaders: {task_type}")

    if not full_dataset or len(full_dataset) == 0:
        logger.warning(f"Full dataset for task_type '{task_type}' is empty or None.")
        return (None, None, 0) if task_type == "prefix_lm_pretrain" else (None, None)

    train_dataset, val_dataset = None, None
    if val_split_ratio > 0 and len(full_dataset) > 1: 
        val_size = int(len(full_dataset) * val_split_ratio); train_size = len(full_dataset) - val_size
        if train_size > 0 and val_size > 0:
            train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
            logger.info(f"Dataset split: Train: {len(train_dataset)}, Val: {len(val_dataset)}")
        else: logger.warning(f"Dataset too small for validation split. Using all for training."); train_dataset = full_dataset
    else:
        train_dataset = full_dataset
        if len(full_dataset) > 0 : logger.info(f"Using all {len(full_dataset)} samples for training (no validation split for this part).")
    
    if not train_dataset or len(train_dataset) == 0 :
        logger.error("Training dataset is empty after split or initial load.")
        return (None, None, 0) if task_type == "prefix_lm_pretrain" else (None, None)

    dl_num_workers = 0 
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=dl_num_workers, pin_memory=torch.cuda.is_available())
    val_dl = None
    if val_dataset and len(val_dataset) > 0: val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=dl_num_workers, pin_memory=torch.cuda.is_available())
    
    return (train_dl, val_dl, num_examples_from_chunk) if task_type == "prefix_lm_pretrain" else (train_dl, val_dl)
