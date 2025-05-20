# --- START OF FILE dataset_loader.py ---

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
from multiprocessing import Pool, cpu_count, Manager
import logging

logger = logging.getLogger(__name__)

# --- Worker function for multiprocessing (init_worker_for_prefix_lm, process_document_for_prefix_lm) ---
WORKER_TOKENIZER_INSTANCE = None
WORKER_MAX_SEQ_LENGTH_FOR_PREFIXLM = None
WORKER_MIN_LEN_FOR_SPLIT_PREFIXLM = None
WORKER_MIN_PART_LEN_PREFIXLM = None

def init_worker_for_prefix_lm(tokenizer_path_str, max_seq_len, min_len_split, min_part_len):
    global WORKER_TOKENIZER_INSTANCE, WORKER_MAX_SEQ_LENGTH_FOR_PREFIXLM, \
           WORKER_MIN_LEN_FOR_SPLIT_PREFIXLM, WORKER_MIN_PART_LEN_PREFIXLM
    # Each worker needs its own tokenizer instance
    logger.debug(f"Worker {os.getpid()} initializing tokenizer from: {tokenizer_path_str}")
    try:
        WORKER_TOKENIZER_INSTANCE = TokenizerWrapper(model_path=tokenizer_path_str)
        WORKER_MAX_SEQ_LENGTH_FOR_PREFIXLM = max_seq_len
        WORKER_MIN_LEN_FOR_SPLIT_PREFIXLM = min_len_split
        WORKER_MIN_PART_LEN_PREFIXLM = min_part_len
    except Exception as e:
        logger.error(f"WORKER ERROR: Failed to initialize tokenizer in worker {os.getpid()}: {e}")
        # Propagate exception to ensure main process knows worker failed init
        raise 

def process_document_for_prefix_lm(doc_str):
    if not doc_str or not isinstance(doc_str, str) or WORKER_TOKENIZER_INSTANCE is None:
        return [] # Return empty list if no doc, wrong type, or tokenizer not init
    
    # Encode without BOS/EOS for splitting, they will be added later per segment
    doc_tokens = WORKER_TOKENIZER_INSTANCE.encode(doc_str, add_bos=False, add_eos=False)

    # Basic length checks
    if len(doc_tokens) < WORKER_MIN_LEN_FOR_SPLIT_PREFIXLM: return []
    # Ensure there's enough length for two meaningful parts
    if len(doc_tokens) < 2 * WORKER_MIN_PART_LEN_PREFIXLM: return []

    # Determine valid split range. Split point is *after* prefix, *before* suffix.
    # Min split_point index: WORKER_MIN_PART_LEN_PREFIXLM
    # Max split_point index: len(doc_tokens) - WORKER_MIN_PART_LEN_PREFIXLM
    if WORKER_MIN_PART_LEN_PREFIXLM > len(doc_tokens) - WORKER_MIN_PART_LEN_PREFIXLM:
        return [] # Not enough room to make a valid split

    split_point = random.randint(WORKER_MIN_PART_LEN_PREFIXLM, len(doc_tokens) - WORKER_MIN_PART_LEN_PREFIXLM)
    
    prefix_toks = doc_tokens[:split_point]
    suffix_toks = doc_tokens[split_point:]

    # Truncate if parts are too long (from the left for prefix, from the right for suffix is implicit by slicing)
    # This means for prefix, we take the *end* of it if it's too long.
    # For suffix, we take the *beginning* of it.
    prefix_toks = prefix_toks[-WORKER_MAX_SEQ_LENGTH_FOR_PREFIXLM:]
    suffix_toks = suffix_toks[:WORKER_MAX_SEQ_LENGTH_FOR_PREFIXLM]
    
    if not prefix_toks or not suffix_toks: return [] # If either part becomes empty after truncation

    # Return tuple: (len_prefix, prefix_tokens, len_suffix, suffix_tokens)
    return [(len(prefix_toks), prefix_toks, len(suffix_toks), suffix_toks)]


class ProblemSolutionDataset(Dataset):
    def __init__(self, problem_statements, gold_solutions, tokenizer: TokenizerWrapper):
        self.problems = problem_statements
        self.solutions = gold_solutions
        self.tokenizer = tokenizer
        if self.tokenizer is None:
            raise ValueError("ProblemSolutionDataset: Tokenizer cannot be None.")

    def __len__(self):
        return len(self.problems)

    def __getitem__(self, idx):
        problem, solution = self.problems[idx], self.solutions[idx]
        
        src_token_ids = self.tokenizer.encode(
            problem, 
            add_bos=True, add_eos=True, 
            max_length=config.MAX_PROBLEM_STATEMENT_TOKENS
        )
        # For Seq2Seq, target needs BOS at start for decoder_input, and EOS at end for labels
        tgt_token_ids_raw = self.tokenizer.encode(
            solution, 
            add_bos=True, add_eos=True, # Add both initially
            max_length=config.MAX_GOLD_SOLUTION_TOKENS # Max length for the whole target sequence
        )
        
        # Decoder input: BOS + solution_tokens (no final EOS here)
        # Labels: solution_tokens + EOS (no initial BOS here)
        # Ensure they are same length after processing.
        # If tgt_token_ids_raw is [BOS, t1, t2, EOS]
        # decoder_input_ids = [BOS, t1, t2]
        # label_ids = [t1, t2, EOS]

        if len(tgt_token_ids_raw) < 2 : # Must have at least BOS + EOS or BOS + token
            # This can happen if solution is empty and max_length is small.
            # Handle gracefully, e.g., by creating minimal valid sequences.
            # Example: if raw is [BOS], then input is [BOS], label is [PAD] (or EOS)
            # If raw is [BOS, EOS], then input is [BOS], label is [EOS]
            decoder_input_ids = [self.tokenizer.bos_id, self.tokenizer.pad_id] # Pad to min length 2
            label_ids = [self.tokenizer.pad_id, self.tokenizer.eos_id]
            if len(tgt_token_ids_raw) == 1 and tgt_token_ids_raw[0] == self.tokenizer.bos_id:
                 decoder_input_ids = [self.tokenizer.bos_id, self.tokenizer.eos_id] # Make it learn to output EOS after BOS for empty
                 label_ids = [self.tokenizer.eos_id, self.tokenizer.pad_id] # PAD
            elif len(tgt_token_ids_raw) > 0: # e.g. just [BOS] or some tokens
                 decoder_input_ids = tgt_token_ids_raw[:-1] if len(tgt_token_ids_raw) > 1 else [tgt_token_ids_raw[0], self.tokenizer.eos_id]
                 label_ids = tgt_token_ids_raw[1:] if len(tgt_token_ids_raw) > 1 else [self.tokenizer.eos_id, self.tokenizer.pad_id]


        else:
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

    # Pad sequences
    padded_src_ids = torch.nn.utils.rnn.pad_sequence(
        src_ids_list, batch_first=True, padding_value=config.PAD_TOKEN_ID
    )
    padded_decoder_input_ids = torch.nn.utils.rnn.pad_sequence(
        decoder_input_ids_list, batch_first=True, padding_value=config.PAD_TOKEN_ID
    )
    padded_label_ids = torch.nn.utils.rnn.pad_sequence(
        label_ids_list, batch_first=True, padding_value=config.PAD_TOKEN_ID
    )
    
    # Ensure decoder_input and labels have the same sequence length (max of the two)
    # This is important because TransformerDecoder expects them to align for loss calculation shifted.
    # If one is shorter (e.g. due to truncation of an empty solution differently), pad it.
    max_len_dec_tgt = max(padded_decoder_input_ids.size(1), padded_label_ids.size(1))

    if padded_decoder_input_ids.size(1) < max_len_dec_tgt:
        padding_needed = max_len_dec_tgt - padded_decoder_input_ids.size(1)
        padding_tensor = torch.full((padded_decoder_input_ids.size(0), padding_needed), config.PAD_TOKEN_ID, 
                                    dtype=torch.long, device=padded_decoder_input_ids.device)
        padded_decoder_input_ids = torch.cat([padded_decoder_input_ids, padding_tensor], dim=1)
    
    if padded_label_ids.size(1) < max_len_dec_tgt:
        padding_needed = max_len_dec_tgt - padded_label_ids.size(1)
        padding_tensor = torch.full((padded_label_ids.size(0), padding_needed), config.PAD_TOKEN_ID, 
                                    dtype=torch.long, device=padded_label_ids.device)
        padded_label_ids = torch.cat([padded_label_ids, padding_tensor], dim=1)

    return {
        "src_token_ids": padded_src_ids, 
        "decoder_input_ids": padded_decoder_input_ids, 
        "label_ids": padded_label_ids
    }


class FineWebPrefixLMDataset(Dataset):
    def __init__(self, parquet_file_path, 
                 tokenizer_model_path_for_worker, # Path for worker processes
                 max_prefix_suffix_len=config.MAX_TEXT_PRETRAIN_SEQ_LEN,
                 start_doc_offset=0,
                 num_docs_to_process_in_chunk=config.DOCS_CHUNK_SIZE_PER_PREFIXLM_EPOCH,
                 cache_dir_base_for_run: str ="./dataset_cache/default_run", 
                 min_len_for_split=config.MIN_LEN_FOR_PREFIXLM_SPLIT,
                 min_part_len=config.MIN_PART_LEN_FOR_PREFIXLM,
                 num_workers_caching=0): # 0 means auto-detect based on CPU
        
        self.parquet_file_path = parquet_file_path
        # Main process tokenizer is for __getitem__ (adding BOS/EOS)
        self.main_process_tokenizer = global_tokenizer 
        if self.main_process_tokenizer is None:
            raise ValueError("FineWebPrefixLMDataset: Global tokenizer not available for main process.")
            
        self.tokenizer_model_path_for_worker = tokenizer_model_path_for_worker
        self.max_part_len = max_prefix_suffix_len
        self.start_doc_offset = start_doc_offset
        self.num_docs_to_process_in_chunk = num_docs_to_process_in_chunk
        self.min_len_for_split = min_len_for_split
        self.min_part_len = min_part_len
        
        if num_workers_caching <= 0: self.num_workers_caching = max(1, cpu_count() // 2 if cpu_count() > 1 else 1)
        else: self.num_workers_caching = num_workers_caching
        
        file_basename = os.path.basename(self.parquet_file_path).replace('.parquet', '')
        self.cache_dir_for_file = os.path.join(cache_dir_base_for_run, "fineweb_prefix_lm", file_basename)
        os.makedirs(self.cache_dir_for_file, exist_ok=True)
        
        # More descriptive cache chunk ID
        cache_chunk_id = (f"prefixlm_s{start_doc_offset}_n{num_docs_to_process_in_chunk}_"
                          f"mpl{max_prefix_suffix_len}_mls{min_len_for_split}_mp{min_part_len}_"
                          f"tokV{self.main_process_tokenizer.sp.vocab_size()}") # Add tokenizer vocab size to vary cache with tokenizer
        
        self.cache_data_file = os.path.join(self.cache_dir_for_file, f"{cache_chunk_id}.dat")
        self.cache_index_file = os.path.join(self.cache_dir_for_file, f"{cache_chunk_id}.idx")

        self.num_examples = 0
        self.example_metadata = [] 
        self.data_file_handle = None

        if not self._load_index_from_cache():
            logger.info(f"PrefixLM Cache not found or invalid for {os.path.basename(parquet_file_path)} chunk {cache_chunk_id}. Building with {self.num_workers_caching} workers...")
            self._build_and_write_cache_prefix_lm_mp()
            if not self._load_index_from_cache() or self.num_examples == 0:
                logger.warning(f"No PrefixLM examples built/loaded for {os.path.basename(parquet_file_path)} chunk {cache_chunk_id}. Dataset will be empty.")
        
        if self.num_examples > 0:
            logger.info(f"FineWebPrefixLMDataset: {os.path.basename(parquet_file_path)} chunk {cache_chunk_id}. Found {self.num_examples} examples.")

    def _build_and_write_cache_prefix_lm_mp(self):
        logger.info(f"Building PrefixLM cache (MP): {os.path.basename(self.cache_data_file)}")
        current_offset_bytes = 0; temp_metadata = []; processed_docs_count_for_chunk = 0; docs_skipped_before_offset = 0
        
        # Check if worker tokenizer path exists, critical for Pool
        if not os.path.exists(self.tokenizer_model_path_for_worker):
            logger.error(f"WORKER TOKENIZER MODEL PATH NOT FOUND: {self.tokenizer_model_path_for_worker}. Cannot build cache.")
            self._cleanup_failed_cache()
            return

        pool_initargs = (self.tokenizer_model_path_for_worker, self.max_part_len, self.min_len_for_split, self.min_part_len)
        
        try:
            # Use try-finally for Pool to ensure it's closed
            pool = Pool(processes=self.num_workers_caching, initializer=init_worker_for_prefix_lm, initargs=pool_initargs)
            try:
                with open(self.cache_data_file, 'wb') as data_f:
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
                                    docs_to_process_in_rg_py_strings.append(doc_str); 
                                    processed_docs_count_for_chunk += 1 
                        
                        del table, text_column; gc.collect()
                        if not docs_to_process_in_rg_py_strings: continue
                        
                        # Process this batch of documents with the pool
                        list_of_worker_results = pool.map(process_document_for_prefix_lm, docs_to_process_in_rg_py_strings)
                        
                        for single_doc_results in list_of_worker_results: # Each item is a list (usually one item) from a worker
                            for len_p, prefix_toks, len_s, suffix_toks in single_doc_results: 
                                data_f.write(struct.pack('<I', len_p)) # Prefix length
                                if len_p > 0: data_f.write(struct.pack(f'<{len_p}I', *prefix_toks)) # Prefix tokens
                                data_f.write(struct.pack('<I', len_s)) # Suffix length
                                if len_s > 0: data_f.write(struct.pack(f'<{len_s}I', *suffix_toks)) # Suffix tokens
                                
                                temp_metadata.append((current_offset_bytes, len_p, len_s))
                                current_offset_bytes += 4 + (len_p * 4) + 4 + (len_s * 4) # Update byte offset
                        
                        del list_of_worker_results; gc.collect()
                        pbar_rg.set_postfix_str(f"{processed_docs_count_for_chunk}/{self.num_docs_to_process_in_chunk} docs, {len(temp_metadata)} ex.")
                    pbar_rg.close()
            finally:
                pool.close()
                pool.join() # Wait for all worker processes to finish

            with open(self.cache_index_file, 'wb') as idx_f:
                idx_f.write(struct.pack('<Q', len(temp_metadata))) # Number of examples
                for offset, lp, ls in temp_metadata: idx_f.write(struct.pack('<QII', offset, lp, ls)) # Offset, len_prefix, len_suffix
            
            logger.info(f"PrefixLM Cache (MP) for chunk {os.path.basename(self.cache_data_file)} completed: {len(temp_metadata)} examples from {processed_docs_count_for_chunk} documents.")
        
        except Exception as e: 
            logger.error(f"Error during PrefixLM Cache build (MP): {e}\n{traceback.format_exc()}")
            self._cleanup_failed_cache() # Clean up partial/corrupt files
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
                for _ in range(self.num_examples): self.example_metadata.append(struct.unpack('<QII', index_f.read(16))) # (offset, len_prefix, len_suffix)
            
            if self.num_examples > 0:
                # Close existing handle if open, then re-open. Important for multi-epoch/re-init scenarios.
                if self.data_file_handle and not self.data_file_handle.closed: self.data_file_handle.close()
                self.data_file_handle = open(self.cache_data_file, 'rb')
            return True
        except Exception as e: 
            logger.error(f"Error loading PrefixLM cache index for {os.path.basename(self.cache_index_file)}: {e}")
            self._cleanup_failed_cache(); return False

    def __len__(self): return self.num_examples

    def __getitem__(self, idx):
        if idx >= self.num_examples or idx < 0: raise IndexError(f"Index {idx} out of bounds for {self.num_examples} examples.")
        if not self.data_file_handle or self.data_file_handle.closed:
            try: 
                logger.warning(f"Re-opening data_file_handle in __getitem__ for {os.path.basename(self.cache_data_file)} (index {idx}). This might happen if DataLoader workers are re-initialized or after pickling/unpickling dataset object across processes without proper handle re-establishment.")
                self.data_file_handle = open(self.cache_data_file, 'rb')
            except Exception as e: 
                raise RuntimeError(f"Failed to re-open cache data file in __getitem__ for {os.path.basename(self.cache_data_file)}: {e}")
        
        offset, len_prefix, len_suffix = self.example_metadata[idx]; self.data_file_handle.seek(offset)
        
        stored_len_p_bytes = self.data_file_handle.read(4)
        if len(stored_len_p_bytes) < 4: raise RuntimeError(f"Cache integrity error: Unexpected EOF reading prefix length at offset {offset} for index {idx}.")
        stored_len_p = struct.unpack('<I', stored_len_p_bytes)[0]
        if stored_len_p != len_prefix: raise RuntimeError(f"Cache integrity error: prefix length mismatch for index {idx}. Index says: {len_prefix}, File says: {stored_len_p}")
        
        prefix_tokens = list(struct.unpack(f'<{len_prefix}I', self.data_file_handle.read(len_prefix * 4))) if len_prefix > 0 else []
        
        stored_len_s_bytes = self.data_file_handle.read(4)
        if len(stored_len_s_bytes) < 4: raise RuntimeError(f"Cache integrity error: Unexpected EOF reading suffix length for index {idx}.")
        stored_len_s = struct.unpack('<I', stored_len_s_bytes)[0]
        if stored_len_s != len_suffix: raise RuntimeError(f"Cache integrity error: suffix length mismatch for index {idx}. Index says: {len_suffix}, File says: {stored_len_s}")
        
        suffix_tokens = list(struct.unpack(f'<{len_suffix}I', self.data_file_handle.read(len_suffix * 4))) if len_suffix > 0 else []

        # Handle cases where suffix_tokens might be empty (e.g., if min_part_len allows it or due to truncation)
        if not suffix_tokens:
             # This shouldn't happen if min_part_len > 0 and processing is correct. Log as warning.
             logger.warning(f"Empty suffix_tokens encountered for index {idx} in {os.path.basename(self.cache_data_file)}. This might indicate an issue in data processing or too short segments. Returning a PAD example.")
             # Create a minimal padded example to avoid crashing the collate_fn
             # This should be a rare case.
             pad_tok = self.main_process_tokenizer.pad_id
             bos_tok = self.main_process_tokenizer.bos_id
             eos_tok = self.main_process_tokenizer.eos_id
             # Dummy prefix, learn to output EOS from BOS
             src_ids = [bos_tok, pad_tok, eos_tok] # BOS + PAD + EOS for src
             decoder_input_ids = [bos_tok, eos_tok]    # BOS + EOS for decoder input
             label_ids =         [eos_tok, pad_tok]    # EOS + PAD for label
        else:
            # Prefix becomes src, Suffix becomes target
            # Add BOS/EOS to prefix for encoder input
            src_ids = [self.main_process_tokenizer.bos_id] + prefix_tokens + [self.main_process_tokenizer.eos_id]
            
            # Prepare decoder_input_ids (BOS + suffix_tokens[:-1]) and label_ids (suffix_tokens)
            if len(suffix_tokens) == 1: # e.g. suffix is just one token [t1]
                decoder_input_ids = [self.main_process_tokenizer.bos_id] # Input BOS, expect t1
                label_ids = suffix_tokens                                # Label t1
            else: # e.g. suffix is [t1, t2, t3]
                decoder_input_ids = [self.main_process_tokenizer.bos_id] + suffix_tokens[:-1] # BOS, t1, t2
                label_ids = suffix_tokens # t1, t2, t3
        
        return {
            "src_token_ids": torch.tensor(src_ids, dtype=torch.long), 
            "decoder_input_ids": torch.tensor(decoder_input_ids, dtype=torch.long), 
            "label_ids": torch.tensor(label_ids, dtype=torch.long)
        }

    def __del__(self):
        if hasattr(self, 'data_file_handle') and self.data_file_handle and not self.data_file_handle.closed:
            try: self.data_file_handle.close()
            except Exception as e: logger.error(f"Error closing data file handle for {getattr(self, 'cache_data_file', 'N/A')} on deletion: {e}")

def collate_prefix_lm_batch(batch):
    src_ids_list = [item['src_token_ids'] for item in batch]
    decoder_input_ids_list = [item['decoder_input_ids'] for item in batch]
    label_ids_list = [item['label_ids'] for item in batch]

    padded_src_ids = torch.nn.utils.rnn.pad_sequence(src_ids_list, batch_first=True, padding_value=config.PAD_TOKEN_ID)
    
    # Pad decoder_input_ids and label_ids to the max length between them in the batch
    # This is crucial because the decoder input and labels must align shifted by one.
    max_len_tgt_side = 0
    for dec_in_ids, lbl_ids in zip(decoder_input_ids_list, label_ids_list):
        max_len_tgt_side = max(max_len_tgt_side, len(dec_in_ids), len(lbl_ids))
    
    padded_decoder_input_ids_list = []
    for ids in decoder_input_ids_list:
        padding = [config.PAD_TOKEN_ID] * (max_len_tgt_side - len(ids))
        padded_decoder_input_ids_list.append(ids + torch.tensor(padding, dtype=torch.long))
    
    padded_label_ids_list = []
    for ids in label_ids_list:
        padding = [config.PAD_TOKEN_ID] * (max_len_tgt_side - len(ids))
        padded_label_ids_list.append(ids + torch.tensor(padding, dtype=torch.long))

    padded_decoder_input_ids = torch.stack(padded_decoder_input_ids_list)
    padded_label_ids = torch.stack(padded_label_ids_list)
    
    return {
        "src_token_ids": padded_src_ids, 
        "decoder_input_ids": padded_decoder_input_ids, 
        "label_ids": padded_label_ids
    }


def get_dataloaders(task_type: str, tokenizer: TokenizerWrapper, batch_size: int, 
                    val_split_ratio: float = config.SUPERVISED_VALIDATION_SPLIT_RATIO, 
                    specific_file_path: str = None, # For PrefixLM, path to one .parquet file
                    start_doc_offset: int = 0,      # For PrefixLM, where to start in the .parquet file
                    num_docs_in_chunk: int = config.DOCS_CHUNK_SIZE_PER_PREFIXLM_EPOCH, # For PrefixLM
                    model_run_name_for_cache: str = "default_run"): # For unique cache location
    
    if tokenizer is None:
        logger.error(f"Tokenizer is None for get_dataloaders (task: {task_type}). Cannot proceed.")
        return (None, None, 0) if task_type == "prefix_lm_pretrain" else (None, None)

    full_dataset = None; collate_fn = None; num_examples_from_chunk = 0 # For PrefixLM

    if task_type == "supervised_code":
        logger.info("Loading supervised code dataset.")
        problems_list, solutions_list = [], []
        dataset_abs_dir = config.CODE_DATASET_DIR
        search_pattern = os.path.join(dataset_abs_dir, "train-*-of-*.parquet")
        logger.info(f"Searching for supervised code dataset files in: {dataset_abs_dir} with pattern: {search_pattern}")
        all_file_paths = sorted(glob.glob(search_pattern))
        
        if not all_file_paths:
            logger.warning(f"No Problem-Solution Parquet files found in {dataset_abs_dir} matching pattern '{os.path.basename(search_pattern)}'.")
        else:
            files_to_load_count = config.NUM_CODE_DATASET_FILES_TO_LOAD
            files_to_load = all_file_paths
            if files_to_load_count > 0 and files_to_load_count < len(all_file_paths):
                files_to_load = all_file_paths[:files_to_load_count]
            
            logger.info(f"Attempting to load {len(files_to_load)} Problem-Solution dataset files (NUM_CODE_DATASET_FILES_TO_LOAD={config.NUM_CODE_DATASET_FILES_TO_LOAD}).")
            
            for filepath in tqdm(files_to_load, desc="Loading Code Datasets", unit="file", ncols=100):
                try:
                    df = pd.read_parquet(filepath)
                    for _, row in df.iterrows():
                        problem, solution_raw = row.get('problem_statement', ''), row.get('gold_standard_solution', '')
                        solution = extract_python_code_from_markdown(solution_raw) 
                        if problem and solution: 
                            problems_list.append(problem.strip())
                            solutions_list.append(solution.strip())
                    del df; gc.collect()
                except Exception as e: logger.error(f"Error loading or processing supervised code file {filepath}: {e}")
            
            if not problems_list: logger.warning("No problem-solution pairs were loaded from supervised dataset files.")
            else: 
                logger.info(f"Loaded {len(problems_list)} problem-solution pairs.")
                full_dataset = ProblemSolutionDataset(problems_list, solutions_list, tokenizer)
        collate_fn = collate_problem_solution_batch

    elif task_type == "prefix_lm_pretrain":
        logger.info(f"Loading PrefixLM dataset for file: {specific_file_path}, model_run_name for cache: {model_run_name_for_cache}")
        if not specific_file_path or not os.path.exists(specific_file_path):
            logger.error(f"ERROR: File path invalid or does not exist for PrefixLM: {specific_file_path}"); return None, None, 0
        if not model_run_name_for_cache or model_run_name_for_cache == "default_run": 
             logger.warning("model_run_name_for_cache is 'default_run' or not set; cache might not be unique if multiple runs use default.")
        
        run_specific_cache_base = config.CACHE_DIR_BASE_TEMPLATE.format(model_run_name=model_run_name_for_cache)
        logger.info(f"  PrefixLM cache base for this run: {run_specific_cache_base}")
        
        full_dataset = FineWebPrefixLMDataset(
            parquet_file_path=specific_file_path, 
            tokenizer_model_path_for_worker=config.TOKENIZER_MODEL_PATH, # Pass path for workers
            max_prefix_suffix_len=config.MAX_TEXT_PRETRAIN_SEQ_LEN,
            start_doc_offset=start_doc_offset,
            num_docs_to_process_in_chunk=num_docs_in_chunk,
            cache_dir_base_for_run=run_specific_cache_base, 
            min_len_for_split=config.MIN_LEN_FOR_PREFIXLM_SPLIT,
            min_part_len=config.MIN_PART_LEN_FOR_PREFIXLM
        )
        num_examples_from_chunk = len(full_dataset) if full_dataset else 0
        if num_examples_from_chunk == 0: logger.warning(f"PrefixLM dataset for {specific_file_path} (chunk starting {start_doc_offset}) is empty.")
        
        collate_fn = collate_prefix_lm_batch
        val_split_ratio = 0 # No validation split for PrefixLM chunks by default
    else: 
        raise ValueError(f"Unknown task_type for get_dataloaders: {task_type}")

    if not full_dataset or len(full_dataset) == 0:
        logger.warning(f"Full dataset for task_type '{task_type}' is empty or None after loading.")
        return (None, None, 0) if task_type == "prefix_lm_pretrain" else (None, None)

    train_dataset, val_dataset = None, None
    if val_split_ratio > 0 and len(full_dataset) > 1: 
        val_size = int(len(full_dataset) * val_split_ratio)
        train_size = len(full_dataset) - val_size
        if train_size > 0 and val_size > 0:
            try:
                train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
                logger.info(f"Dataset split: Train: {len(train_dataset)}, Val: {len(val_dataset)}")
            except Exception as e_split:
                logger.error(f"Error during dataset split: {e_split}. Using all data for training.")
                train_dataset = full_dataset
        else: 
            logger.warning(f"Dataset too small ({len(full_dataset)} samples) for validation split with ratio {val_split_ratio}. Using all for training.")
            train_dataset = full_dataset
    else:
        train_dataset = full_dataset
        if len(full_dataset) > 0 : logger.info(f"Using all {len(full_dataset)} samples for training (no validation split for this part).")
    
    if not train_dataset or len(train_dataset) == 0 :
        logger.error("Training dataset is empty after split or initial load. Cannot create DataLoader.")
        return (None, None, 0) if task_type == "prefix_lm_pretrain" else (None, None)

    # For DataLoader, num_workers > 0 can be problematic with SentencePiece in worker_init_fn
    # unless tokenizer is re-initialized carefully in each worker.
    # For FineWebPrefixLMDataset, the heavy lifting (tokenization) is done during caching.
    # So, __getitem__ is mostly I/O and tensor conversion, num_workers=0 might be fine or even preferred
    # to avoid inter-process communication overhead for already processed data.
    # If caching is fast and __getitem__ is light, num_workers=0 is good.
    # If __getitem__ were heavy, num_workers > 0 would be beneficial but need careful setup.
    dl_num_workers = 0 # Defaulting to 0 for simplicity and stability with current caching.
    
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, 
                          num_workers=dl_num_workers, pin_memory=torch.cuda.is_available())
    val_dl = None
    if val_dataset and len(val_dataset) > 0: 
        val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, 
                            num_workers=dl_num_workers, pin_memory=torch.cuda.is_available())
    
    if task_type == "prefix_lm_pretrain":
        return train_dl, val_dl, num_examples_from_chunk
    else:
        return train_dl, val_dl