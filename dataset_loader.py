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
import logging

logger = logging.getLogger(__name__)

# --- Worker function for multiprocessing ---
WORKER_TOKENIZER_INSTANCE = None
WORKER_MAX_SEQ_LENGTH_FOR_PREFIXLM = None
WORKER_MIN_LEN_FOR_SPLIT_PREFIXLM = None
WORKER_MIN_PART_LEN_PREFIXLM = None

def init_worker_for_prefix_lm(tokenizer_path_str, max_seq_len, min_len_split, min_part_len):
    """Initializes tokenizer and params for each worker process for PrefixLM."""
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
        raise # Reraise to stop the pool if tokenizer init fails

def process_document_for_prefix_lm(doc_str):
    """Processes a single document string to generate prefix/suffix sub-chunks."""
    if not doc_str or not isinstance(doc_str, str) or WORKER_TOKENIZER_INSTANCE is None:
        return [] 

    doc_tokens = WORKER_TOKENIZER_INSTANCE.encode(doc_str, add_bos=False, add_eos=False)
    
    # Ensure document is long enough for a meaningful split
    if len(doc_tokens) < WORKER_MIN_LEN_FOR_SPLIT_PREFIXLM or \
       len(doc_tokens) < 2 * WORKER_MIN_PART_LEN_PREFIXLM: # Simplified condition
        return []

    # Ensure split_point allows for min_part_len on both sides
    # random.randint is inclusive for both ends.
    # Max split point is len(doc_tokens) - WORKER_MIN_PART_LEN_PREFIXLM
    # Min split point is WORKER_MIN_PART_LEN_PREFIXLM
    if WORKER_MIN_PART_LEN_PREFIXLM > len(doc_tokens) - WORKER_MIN_PART_LEN_PREFIXLM:
        return [] # Not enough room to make a valid split

    split_point = random.randint(WORKER_MIN_PART_LEN_PREFIXLM, len(doc_tokens) - WORKER_MIN_PART_LEN_PREFIXLM)
    
    prefix_toks = doc_tokens[:split_point]
    suffix_toks = doc_tokens[split_point:]

    # Truncate to max sequence length (from the end for prefix, from the start for suffix)
    prefix_toks = prefix_toks[-WORKER_MAX_SEQ_LENGTH_FOR_PREFIXLM:]
    suffix_toks = suffix_toks[:WORKER_MAX_SEQ_LENGTH_FOR_PREFIXLM]
    
    # Ensure parts are not empty and suffix has at least one token for label
    if not prefix_toks or not suffix_toks:
        return []
        
    return [(len(prefix_toks), prefix_toks, len(suffix_toks), suffix_toks)]


# --- ProblemSolutionDataset and collate_problem_solution_batch (Assumed Unchanged from original) ---
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
        # For decoder input, remove last token (EOS). For labels, remove first token (BOS).
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
    
    # Ensure decoder_input and labels have the same length due to padding differences if one is all PAD.
    # This typically means padding them to the max length observed between them.
    # However, standard practice is that label_ids should correspond to the predictions for decoder_input_ids.
    # If decoder_input_ids is [BOS, t1, t2] and label_ids is [t1, t2, EOS], padding should align them.
    # The existing padding logic in ProblemSolutionDataset might need adjustment if this causes issues.
    # For now, assume the padding logic in collate_fn is handling this by padding to max length among their type.
    # The original had a more complex padding adjustment; if that's critical, it should be reinstated.
    # My __getitem__ for ProblemSolutionDataset prepares them to be of potentially different lengths before collation.
    # Standard collate_fn just pads each list of tensors independently.
    # Let's revert to the user's original complex padding for this collate_fn to be safe.
    max_len_dec = padded_decoder_input_ids.size(1)
    max_len_lbl = padded_label_ids.size(1)
    target_len = max(max_len_dec, max_len_lbl)

    if max_len_dec < target_len:
        padding_tensor = torch.full((padded_decoder_input_ids.size(0), target_len - max_len_dec), config.PAD_TOKEN_ID, dtype=torch.long, device=padded_decoder_input_ids.device)
        padded_decoder_input_ids = torch.cat([padded_decoder_input_ids, padding_tensor], dim=1)
    elif max_len_dec > target_len: # Should not happen if target_len is max
        padded_decoder_input_ids = padded_decoder_input_ids[:, :target_len]

    if max_len_lbl < target_len:
        padding_tensor = torch.full((padded_label_ids.size(0), target_len - max_len_lbl), config.PAD_TOKEN_ID, dtype=torch.long, device=padded_label_ids.device)
        padded_label_ids = torch.cat([padded_label_ids, padding_tensor], dim=1)
    elif max_len_lbl > target_len: # Should not happen
        padded_label_ids = padded_label_ids[:, :target_len]

    return {
        "src_token_ids": padded_src_ids, 
        "decoder_input_ids": padded_decoder_input_ids, 
        "label_ids": padded_label_ids
    }

class FineWebPrefixLMDataset(Dataset):
    def __init__(self, parquet_file_path, 
                 tokenizer_model_path_for_worker,
                 max_prefix_suffix_len=config.MAX_TEXT_PRETRAIN_SEQ_LEN,
                 start_doc_offset=0,
                 num_docs_to_process_in_chunk=config.DOCS_CHUNK_SIZE_PER_PREFIXLM_EPOCH,
                 cache_dir_base_with_run_name="./dataset_cache/default_run/fineweb_prefix_lm",
                 min_len_for_split=40, # Min total tokens in a doc to consider splitting
                 min_part_len=10,      # Min tokens for prefix and suffix after split, before truncation
                 num_workers_caching=0): # Default to 0, let user specify or use auto
        
        self.parquet_file_path = parquet_file_path
        self.main_process_tokenizer = global_tokenizer 
        self.tokenizer_model_path_for_worker = tokenizer_model_path_for_worker
        self.max_part_len = max_prefix_suffix_len
        self.start_doc_offset = start_doc_offset
        self.num_docs_to_process_in_chunk = num_docs_to_process_in_chunk
        self.min_len_for_split = min_len_for_split
        self.min_part_len = min_part_len # Store this for worker initargs
        
        if num_workers_caching <= 0:
            self.num_workers_caching = max(1, cpu_count() // 2)
        else:
            self.num_workers_caching = num_workers_caching
        
        file_basename = os.path.basename(self.parquet_file_path).replace('.parquet', '')
        # Ensure cache_dir_base_with_run_name is a valid path component
        safe_run_name_component = os.path.basename(cache_dir_base_with_run_name.rstrip('/\\')) # Get last part if it's a path
        
        self.cache_dir_for_file = os.path.join(config.PROJECT_ROOT, "dataset_cache", safe_run_name_component, "fineweb_prefix_lm", file_basename)
        os.makedirs(self.cache_dir_for_file, exist_ok=True)
        
        cache_chunk_id = f"prefixlm_s{start_doc_offset}_n{num_docs_to_process_in_chunk}_mpl{max_prefix_suffix_len}_mls{min_len_for_split}_mp{min_part_len}"
        self.cache_data_file = os.path.join(self.cache_dir_for_file, f"{cache_chunk_id}.dat")
        self.cache_index_file = os.path.join(self.cache_dir_for_file, f"{cache_chunk_id}.idx")

        self.num_examples = 0
        self.example_metadata = [] # List of (offset_in_data_file, len_prefix_tokens, len_suffix_tokens)
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
        current_offset_bytes = 0
        temp_metadata = []
        processed_docs_count_for_chunk = 0 # Docs contributing to this specific chunk
        docs_skipped_before_offset = 0

        try:
            # initargs for Pool's initializer
            pool_initargs = (
                self.tokenizer_model_path_for_worker,
                self.max_part_len,
                self.min_len_for_split,
                self.min_part_len # Pass min_part_len to worker
            )
            with Pool(processes=self.num_workers_caching, initializer=init_worker_for_prefix_lm, initargs=pool_initargs) as pool, \
                 open(self.cache_data_file, 'wb') as data_f:

                parquet_file = pq.ParquetFile(self.parquet_file_path)
                # Iterate through row groups for memory efficiency
                pbar_rg = tqdm(range(parquet_file.num_row_groups), desc=f"Caching PrefixLM RGs for {os.path.basename(self.parquet_file_path)}", unit="rg", ncols=100, smoothing=0.1)
                
                for rg_idx in pbar_rg:
                    if processed_docs_count_for_chunk >= self.num_docs_to_process_in_chunk:
                        break 
                    
                    rg_meta = parquet_file.metadata.row_group(rg_idx)
                    rows_in_rg = rg_meta.num_rows

                    # Efficiently skip entire row groups if they are before the start_doc_offset
                    if docs_skipped_before_offset + rows_in_rg <= self.start_doc_offset:
                        docs_skipped_before_offset += rows_in_rg
                        continue
                    
                    table = parquet_file.read_row_group(rg_idx, columns=['text'])
                    text_column = table.column('text')
                    
                    docs_to_process_in_rg_py_strings = []
                    
                    # Iterate within the row group
                    for doc_idx_in_rg in range(len(text_column)):
                        current_global_doc_index = docs_skipped_before_offset + doc_idx_in_rg
                        
                        if current_global_doc_index < self.start_doc_offset:
                            continue # Skip docs before the designated start offset

                        if processed_docs_count_for_chunk >= self.num_docs_to_process_in_chunk:
                            break # Stop if we've gathered enough documents for this chunk

                        arrow_scalar = text_column[doc_idx_in_rg]
                        if arrow_scalar.is_valid:
                            doc_str = arrow_scalar.as_py()
                            if doc_str and isinstance(doc_str, str):
                                docs_to_process_in_rg_py_strings.append(doc_str)
                                processed_docs_count_for_chunk += 1 
                    
                    del table, text_column # Free memory
                    gc.collect()

                    if not docs_to_process_in_rg_py_strings:
                        continue

                    # Process this batch of documents in parallel
                    # Each item in list_of_worker_results is a list of tuples (or empty list)
                    list_of_worker_results = pool.map(process_document_for_prefix_lm, docs_to_process_in_rg_py_strings)
                    
                    for single_doc_results in list_of_worker_results: # Results for one original document
                        for len_p, prefix_toks, len_s, suffix_toks in single_doc_results: # One example (prefix, suffix pair)
                            data_f.write(struct.pack('<I', len_p)) # Store length of prefix
                            if len_p > 0: data_f.write(struct.pack(f'<{len_p}I', *prefix_toks)) # Store prefix tokens
                            data_f.write(struct.pack('<I', len_s)) # Store length of suffix
                            if len_s > 0: data_f.write(struct.pack(f'<{len_s}I', *suffix_toks)) # Store suffix tokens
                            
                            temp_metadata.append((current_offset_bytes, len_p, len_s))
                            current_offset_bytes += 4 + (len_p * 4) + 4 + (len_s * 4) # 4 bytes per int (I)
                    
                    del list_of_worker_results
                    gc.collect()
                    pbar_rg.set_postfix_str(f"{processed_docs_count_for_chunk}/{self.num_docs_to_process_in_chunk} docs considered, {len(temp_metadata)} ex. found")

                pbar_rg.close()

            # After processing all row groups (or enough documents), write the index file
            with open(self.cache_index_file, 'wb') as idx_f:
                idx_f.write(struct.pack('<Q', len(temp_metadata))) # Store number of examples (Q: unsigned long long)
                for offset, lp, ls in temp_metadata:
                    idx_f.write(struct.pack('<QII', offset, lp, ls)) # Store offset, len_prefix, len_suffix
            
            logger.info(f"PrefixLM Cache (MP) for chunk completed: {len(temp_metadata)} examples from {processed_docs_count_for_chunk} documents processed for this chunk.")

        except Exception as e:
            logger.error(f"Error during PrefixLM Cache build (MP): {e}\n{traceback.format_exc()}")
            self._cleanup_failed_cache() # Remove potentially corrupted cache files
        finally:
            if 'parquet_file' in locals() and parquet_file is not None: del parquet_file
            gc.collect()


    def _cleanup_failed_cache(self):
        try:
            if os.path.exists(self.cache_data_file): os.remove(self.cache_data_file)
            if os.path.exists(self.cache_index_file): os.remove(self.cache_index_file)
            logger.info(f"Cleaned up potentially corrupted cache files for {os.path.basename(self.cache_data_file)}")
        except OSError as e:
            logger.error(f"Error cleaning up cache files: {e}")


    def _load_index_from_cache(self):
        if not os.path.exists(self.cache_data_file) or not os.path.exists(self.cache_index_file):
            return False
        try:
            with open(self.cache_index_file, 'rb') as index_f:
                self.num_examples = struct.unpack('<Q', index_f.read(8))[0]
                self.example_metadata = [] # Reset if loading
                for _ in range(self.num_examples):
                    # Q (offset: 8 bytes), I (len_prefix: 4 bytes), I (len_suffix: 4 bytes) = 16 bytes
                    offset, len_p, len_s = struct.unpack('<QII', index_f.read(16))
                    self.example_metadata.append((offset, len_p, len_s))
            
            if self.num_examples > 0:
                # Open data file handle only if examples exist, close if previously open
                if self.data_file_handle and not self.data_file_handle.closed:
                    self.data_file_handle.close()
                self.data_file_handle = open(self.cache_data_file, 'rb')
            return True
        except Exception as e:
            logger.error(f"Error loading PrefixLM cache index for {os.path.basename(self.cache_index_file)}: {e}")
            self._cleanup_failed_cache() # Clean up if index is corrupted
            return False

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        if idx >= self.num_examples or idx < 0:
            raise IndexError(f"Index {idx} out of bounds for {self.num_examples} examples.")
        
        if not self.data_file_handle or self.data_file_handle.closed:
            # This should ideally not happen if _load_index_from_cache was successful and num_examples > 0
            try:
                logger.warning(f"Re-opening data_file_handle in __getitem__ for {os.path.basename(self.cache_data_file)}. This might indicate an issue.")
                self.data_file_handle = open(self.cache_data_file, 'rb')
            except Exception as e:
                raise RuntimeError(f"Failed to re-open cache data file in __getitem__ for {os.path.basename(self.cache_data_file)}: {e}")
        
        offset, len_prefix, len_suffix = self.example_metadata[idx]
        self.data_file_handle.seek(offset)
        
        # Read and verify prefix length
        stored_len_p_bytes = self.data_file_handle.read(4)
        if len(stored_len_p_bytes) < 4: raise RuntimeError(f"PrefixLM Cache integrity: Unexpected EOF reading prefix length.")
        stored_len_p = struct.unpack('<I', stored_len_p_bytes)[0]
        if stored_len_p != len_prefix:
            raise RuntimeError(f"PrefixLM Cache integrity: prefix length mismatch. Index says {len_prefix}, file says {stored_len_p} at offset {offset}.")
        
        prefix_tokens = []
        if len_prefix > 0:
            prefix_bytes = self.data_file_handle.read(len_prefix * 4)
            if len(prefix_bytes) < len_prefix * 4: raise RuntimeError(f"PrefixLM Cache integrity: Unexpected EOF reading prefix tokens.")
            prefix_tokens = list(struct.unpack(f'<{len_prefix}I', prefix_bytes))

        # Read and verify suffix length
        stored_len_s_bytes = self.data_file_handle.read(4)
        if len(stored_len_s_bytes) < 4: raise RuntimeError(f"PrefixLM Cache integrity: Unexpected EOF reading suffix length.")
        stored_len_s = struct.unpack('<I', stored_len_s_bytes)[0]
        if stored_len_s != len_suffix:
            raise RuntimeError(f"PrefixLM Cache integrity: suffix length mismatch. Index says {len_suffix}, file says {stored_len_s}.")

        suffix_tokens = []
        if len_suffix > 0:
            suffix_bytes = self.data_file_handle.read(len_suffix * 4)
            if len(suffix_bytes) < len_suffix * 4: raise RuntimeError(f"PrefixLM Cache integrity: Unexpected EOF reading suffix tokens.")
            suffix_tokens = list(struct.unpack(f'<{len_suffix}I', suffix_bytes))

        # Fallback for problematic data (e.g., suffix too short)
        # The model expects decoder_input_ids (shifted suffix) and label_ids (original suffix)
        if not suffix_tokens: # Or len(suffix_tokens) < 1, depending on strictness
             logger.warning(f"Found an empty suffix for index {idx} in {os.path.basename(self.cache_data_file)}. Returning PAD example.")
             # Return a minimal padded example that the collate_fn can handle
             # This structure should match what the model expects after BOS/EOS addition.
             return {
                 "src_token_ids": torch.tensor([self.main_process_tokenizer.bos_id, config.PAD_TOKEN_ID, self.main_process_tokenizer.eos_id], dtype=torch.long), 
                 "decoder_input_ids": torch.tensor([self.main_process_tokenizer.bos_id, self.main_process_tokenizer.eos_id], dtype=torch.long), 
                 "label_ids": torch.tensor([self.main_process_tokenizer.eos_id, config.PAD_TOKEN_ID], dtype=torch.long)
             }
        
        # Prepare for Seq2Seq:
        # src_token_ids: [BOS] + prefix + [EOS]
        # decoder_input_ids: [BOS] + suffix[:-1]  (shifted right)
        # label_ids: suffix (target)
        src_ids = [self.main_process_tokenizer.bos_id] + prefix_tokens + [self.main_process_tokenizer.eos_id]
        
        # Suffix is the target. Decoder input is BOS + suffix shifted. Labels are suffix.
        if len(suffix_tokens) == 1: # e.g., suffix is just [tokenA]
            decoder_input_ids = [self.main_process_tokenizer.bos_id] # Input to predict tokenA
            label_ids = suffix_tokens # Target is [tokenA]
        else: # e.g., suffix is [tokenA, tokenB, EOS_from_doc_if_present]
            decoder_input_ids = [self.main_process_tokenizer.bos_id] + suffix_tokens[:-1]
            label_ids = suffix_tokens # The model's loss function will ignore PADs in labels

        return {
            "src_token_ids": torch.tensor(src_ids, dtype=torch.long), 
            "decoder_input_ids": torch.tensor(decoder_input_ids, dtype=torch.long), 
            "label_ids": torch.tensor(label_ids, dtype=torch.long)
        }
    
    def __del__(self):
        if hasattr(self, 'data_file_handle') and self.data_file_handle and not self.data_file_handle.closed:
            try:
                self.data_file_handle.close()
            except Exception as e:
                logger.error(f"Error closing data file handle for {getattr(self, 'cache_data_file', 'N/A')}: {e}")


def collate_prefix_lm_batch(batch):
    """Collates a batch of prefix_lm examples, padding them to the max length in the batch."""
    src_ids_list = [item['src_token_ids'] for item in batch]
    decoder_input_ids_list = [item['decoder_input_ids'] for item in batch]
    label_ids_list = [item['label_ids'] for item in batch]

    padded_src_ids = torch.nn.utils.rnn.pad_sequence(src_ids_list, batch_first=True, padding_value=config.PAD_TOKEN_ID)
    padded_decoder_input_ids = torch.nn.utils.rnn.pad_sequence(decoder_input_ids_list, batch_first=True, padding_value=config.PAD_TOKEN_ID)
    padded_label_ids = torch.nn.utils.rnn.pad_sequence(label_ids_list, batch_first=True, padding_value=config.PAD_TOKEN_ID)

    # Ensure decoder_input_ids and label_ids have the same sequence length after padding,
    # as the transformer decoder expects this for calculating loss.
    # The CrossEntropyLoss will ignore PAD_TOKEN_ID in labels.
    # Max length on the decoder side (decoder_input vs labels)
    max_len_decoder_side = max(padded_decoder_input_ids.size(1), padded_label_ids.size(1))

    if padded_decoder_input_ids.size(1) < max_len_decoder_side:
        padding_tensor = torch.full(
            (padded_decoder_input_ids.size(0), max_len_decoder_side - padded_decoder_input_ids.size(1)),
            config.PAD_TOKEN_ID, dtype=torch.long, device=padded_decoder_input_ids.device
        )
        padded_decoder_input_ids = torch.cat([padded_decoder_input_ids, padding_tensor], dim=1)
    
    if padded_label_ids.size(1) < max_len_decoder_side:
        padding_tensor = torch.full(
            (padded_label_ids.size(0), max_len_decoder_side - padded_label_ids.size(1)),
            config.PAD_TOKEN_ID, dtype=torch.long, device=padded_label_ids.device
        )
        padded_label_ids = torch.cat([padded_label_ids, padding_tensor], dim=1)
        
    return {
        "src_token_ids": padded_src_ids, 
        "decoder_input_ids": padded_decoder_input_ids, 
        "label_ids": padded_label_ids
    }


def get_dataloaders(task_type: str, tokenizer, batch_size, 
                    val_split_ratio=config.SUPERVISED_VALIDATION_SPLIT_RATIO, 
                    specific_file_path: str = None, # For PrefixLM, path to a single Parquet file
                    start_doc_offset: int = 0,      # For PrefixLM, offset within that file
                    num_docs_in_chunk: int = config.DOCS_CHUNK_SIZE_PER_PREFIXLM_EPOCH, # For PrefixLM
                    model_run_name_for_cache: str = "default_run"): # Crucial for unique cache paths
    
    full_dataset = None
    collate_fn = None
    num_examples_from_chunk = 0 # Specific to PrefixLM

    if task_type == "supervised_code":
        logger.info("Loading supervised code dataset.")
        problems_list, solutions_list = [], []
        dataset_abs_dir = config.CODE_DATASET_DIR
        search_pattern = os.path.join(dataset_abs_dir, "train-*-of-*.parquet")
        file_paths = sorted(glob.glob(search_pattern))
        
        if not file_paths:
            logger.warning(f"No Problem-Solution Parquet files found in {dataset_abs_dir}.")
        else:
            files_to_load_count = config.NUM_CODE_DATASET_FILES_TO_LOAD if hasattr(config, 'NUM_CODE_DATASET_FILES_TO_LOAD') and config.NUM_CODE_DATASET_FILES_TO_LOAD > 0 else len(file_paths)
            files_to_load = file_paths[:files_to_load_count]
            logger.info(f"Loading up to {len(files_to_load)} Problem-Solution dataset files.")
            
            for filepath in tqdm(files_to_load, desc="Loading Code Datasets", unit="file", ncols=100):
                try:
                    df = pd.read_parquet(filepath)
                    for _, row in df.iterrows():
                        problem = row.get('problem_statement', '')
                        solution_raw = row.get('gold_standard_solution', '')
                        solution = extract_python_code_from_markdown(solution_raw) # Assuming this util is robust
                        if problem and solution:
                            problems_list.append(problem.strip())
                            solutions_list.append(solution.strip())
                    del df; gc.collect()
                except Exception as e:
                    logger.error(f"Error loading or processing supervised code file {filepath}: {e}")
            
            if not problems_list:
                logger.warning("No problem-solution pairs were loaded from supervised dataset files.")
            else:
                logger.info(f"Loaded {len(problems_list)} problem-solution pairs.")
                full_dataset = ProblemSolutionDataset(problems_list, solutions_list, tokenizer)
        
        collate_fn = collate_problem_solution_batch

    elif task_type == "prefix_lm_pretrain":
        logger.info(f"Loading PrefixLM dataset for file: {specific_file_path}, model_run_name for cache: {model_run_name_for_cache}")
        if not specific_file_path or not os.path.exists(specific_file_path):
            logger.error(f"ERROR: File path invalid or does not exist for PrefixLM: {specific_file_path}")
            return None, None, 0 # Match expected tuple return
        if not model_run_name_for_cache or model_run_name_for_cache == "default_run": 
             logger.warning("model_run_name_for_cache is default or not set; cache might not be unique if multiple runs use default.")
        
        # Construct the base directory for caches related to this specific model run
        run_specific_cache_base = os.path.join(config.PROJECT_ROOT, "dataset_cache", model_run_name_for_cache)
        
        full_dataset = FineWebPrefixLMDataset(
            parquet_file_path=specific_file_path, 
            tokenizer_model_path_for_worker=config.TOKENIZER_MODEL_PATH, # Pass tokenizer model path for workers
            max_prefix_suffix_len=config.MAX_TEXT_PRETRAIN_SEQ_LEN,
            start_doc_offset=start_doc_offset,
            num_docs_to_process_in_chunk=num_docs_in_chunk,
            cache_dir_base_with_run_name=run_specific_cache_base, # Pass the run-specific base path
            min_len_for_split=config.MIN_LEN_FOR_PREFIXLM_SPLIT if hasattr(config, 'MIN_LEN_FOR_PREFIXLM_SPLIT') else 40,
            min_part_len=config.MIN_PART_LEN_FOR_PREFIXLM if hasattr(config, 'MIN_PART_LEN_FOR_PREFIXLM') else 10
        )
        num_examples_from_chunk = len(full_dataset)
        if num_examples_from_chunk == 0:
            logger.warning(f"PrefixLM dataset for {specific_file_path} (chunk starting {start_doc_offset}) is empty.")
            # Still return empty DataLoaders and 0 examples, phase logic should handle this.
        collate_fn = collate_prefix_lm_batch
        val_split_ratio = 0 # No validation split for individual PrefixLM chunks by default
    
    else:
        raise ValueError(f"Unknown task_type for get_dataloaders: {task_type}")

    if not full_dataset or len(full_dataset) == 0:
        logger.warning(f"Full dataset for task_type '{task_type}' is empty or None.")
        if task_type == "prefix_lm_pretrain":
            return None, None, 0
        else:
            return None, None # For supervised

    train_dataset, val_dataset = None, None
    if val_split_ratio > 0 and len(full_dataset) > 1: # Ensure there's enough data for a split
        val_size = int(len(full_dataset) * val_split_ratio)
        train_size = len(full_dataset) - val_size
        if train_size > 0 and val_size > 0:
            train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
            logger.info(f"Dataset split: Train: {len(train_dataset)}, Val: {len(val_dataset)}")
        else:
            logger.warning(f"Dataset too small for validation split (total {len(full_dataset)}). Using all for training.")
            train_dataset = full_dataset
    else:
        train_dataset = full_dataset
        if len(full_dataset) > 0 : logger.info(f"Using all {len(full_dataset)} samples for training (no validation split for this part).")
    
    if not train_dataset or len(train_dataset) == 0 :
        logger.error("Training dataset is empty after split or initial load.")
        if task_type == "prefix_lm_pretrain":
            return None, None, 0
        else:
            return None, None

    # DataLoader num_workers: 0 is often safer for debugging and on some platforms.
    # Can be increased for performance if I/O is a bottleneck and multiprocessing is stable.
    dl_num_workers = 0 # As per original file's preference for stability
    # if torch.cuda.is_available() and not sys.platform == 'win32':
    #     dl_num_workers = min(4, cpu_count() // 2) # Example of enabling workers

    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
                          num_workers=dl_num_workers, pin_memory=torch.cuda.is_available())
    val_dl = None
    if val_dataset and len(val_dataset) > 0:
        val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
                            num_workers=dl_num_workers, pin_memory=torch.cuda.is_available())
    
    if task_type == "prefix_lm_pretrain":
        return train_dl, val_dl, num_examples_from_chunk # val_dl will be None here
    else: # supervised_code
        return train_dl, val_dl