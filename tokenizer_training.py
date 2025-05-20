# --- START OF FILE tokenizer_training.py ---

# tokenizer_training.py
import sentencepiece as spm
import glob
import os
import pandas as pd
from utils import extract_python_code_from_markdown
import config # Import the main config file
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths from config for consistency
OUTPUT_TOKENIZER_DIR = os.path.dirname(config.TOKENIZER_MODEL_PATH)
TOKENIZER_MODEL_PREFIX = os.path.splitext(os.path.basename(config.TOKENIZER_MODEL_PATH))[0] # e.g. "custom_tokenizer"
TRAIN_TEXT_FILE = os.path.join(OUTPUT_TOKENIZER_DIR, "tokenizer_train_data.txt")

# Configuration for how much data to use for tokenizer training
NUM_CODE_DATASET_FILES_FOR_TOKENIZER = 5
NUM_FINEWEB_FILES_FOR_TOKENIZER = 3 # Number of FineWeb files to process
# FineWeb files can be large, consider processing only a portion of rows from each
MAX_ROWS_PER_FINEWEB_FILE_FOR_TOKENIZER = 100000 # Limit rows per FineWeb file to save memory further

TARGET_VOCAB_SIZE_TOKENIZER = config.FALLBACK_VOCAB_SIZE # Default, can be overridden


def stream_text_for_tokenizer(
    output_train_file_path, 
    num_code_files_to_process=NUM_CODE_DATASET_FILES_FOR_TOKENIZER,
    num_fineweb_files_to_process=NUM_FINEWEB_FILES_FOR_TOKENIZER,
    max_rows_per_fineweb_file=MAX_ROWS_PER_FINEWEB_FILE_FOR_TOKENIZER
    ):
    os.makedirs(os.path.dirname(output_train_file_path), exist_ok=True)
    total_segments_written = 0

    try:
        with open(output_train_file_path, 'w', encoding='utf-8') as outfile:
            # --- 1. Stream from Code Problem/Solution Dataset ---
            code_dataset_search_pattern = os.path.join(config.CODE_DATASET_DIR, "train-*-of-*.parquet")
            code_file_paths = sorted(glob.glob(code_dataset_search_pattern))

            if not code_file_paths:
                logger.warning(f"No Code Problem/Solution Parquet files found in {config.CODE_DATASET_DIR}.")
            else:
                logger.info(f"Streaming text from up to {num_code_files_to_process} Code Problem/Solution Parquet files...")
                files_processed_code = 0
                for filepath in code_file_paths:
                    if files_processed_code >= num_code_files_to_process and num_code_files_to_process > 0 : # only limit if >0
                        break
                    try:
                        df = pd.read_parquet(filepath) 
                        for _, row in df.iterrows():
                            problem = row.get('problem_statement', '')
                            solution_raw = row.get('gold_standard_solution', '')
                            solution = extract_python_code_from_markdown(solution_raw)
                            if problem:
                                outfile.write(problem.strip().replace('\n', ' ') + '\n')
                                total_segments_written += 1
                            if solution:
                                outfile.write(solution.strip().replace('\n', ' ') + '\n')
                                total_segments_written += 1
                        del df; import gc; gc.collect()
                        files_processed_code += 1
                        logger.info(f"  Streamed Code Dataset: {os.path.basename(filepath)}")
                    except Exception as e:
                        logger.error(f"Error processing Code Dataset file {filepath}: {e}")

            # --- 2. Stream from FineWeb Text Dataset ---
            fineweb_search_pattern = os.path.join(config.FINEWEB_DATA_DIR, "*.parquet")
            fineweb_file_paths = sorted(glob.glob(fineweb_search_pattern))

            if not fineweb_file_paths:
                logger.warning(f"No FineWeb Parquet files found in {config.FINEWEB_DATA_DIR}.")
            else:
                logger.info(f"Streaming text from up to {num_fineweb_files_to_process} FineWeb Parquet files (max {max_rows_per_fineweb_file} rows per file)...")
                files_processed_fineweb = 0
                for filepath in fineweb_file_paths:
                    if files_processed_fineweb >= num_fineweb_files_to_process and num_fineweb_files_to_process > 0: # only limit if > 0
                        break
                    try:
                        df = pd.read_parquet(filepath) 
                        if 'text' in df.columns:
                            rows_processed_in_file = 0
                            for text_doc in df['text'].dropna(): 
                                if rows_processed_in_file >= max_rows_per_fineweb_file and max_rows_per_fineweb_file > 0 : # only limit if > 0
                                    logger.info(f"    Reached max row limit ({max_rows_per_fineweb_file}) for {os.path.basename(filepath)}")
                                    break
                                if text_doc and isinstance(text_doc, str):
                                    outfile.write(text_doc.strip().replace('\n', ' ') + '\n')
                                    total_segments_written += 1
                                    rows_processed_in_file += 1
                        else:
                            logger.warning(f"  Warning: Column 'text' not found in FineWeb file {os.path.basename(filepath)}. Skipping.")
                        del df; import gc; gc.collect()
                        files_processed_fineweb += 1
                        logger.info(f"  Streamed FineWeb Dataset: {os.path.basename(filepath)}")
                    except Exception as e:
                        logger.error(f"Error processing FineWeb file {filepath}: {e}")

    except Exception as e_outer:
        logger.error(f"Error opening or writing to output file {output_train_file_path}: {e_outer}")
        return False, 0 

    if total_segments_written == 0:
        logger.warning("No text data streamed from any source. Tokenizer training cannot proceed.")
        return False, 0

    logger.info(f"\nTotal text segments streamed to {output_train_file_path}: {total_segments_written}")
    return True, total_segments_written


def train_sentencepiece_tokenizer(vocab_size=TARGET_VOCAB_SIZE_TOKENIZER):
    success, num_segments = stream_text_for_tokenizer(TRAIN_TEXT_FILE)
    if not success or num_segments == 0:
        logger.error("Failed to prepare training data for tokenizer.")
        return

    model_prefix_path = os.path.join(OUTPUT_TOKENIZER_DIR, TOKENIZER_MODEL_PREFIX)

    logger.info(f"\nStarting SentencePiece tokenizer training...")
    logger.info(f"  Input text file: {TRAIN_TEXT_FILE} ({num_segments} lines)")
    logger.info(f"  Output model prefix: {model_prefix_path}")
    logger.info(f"  Target vocabulary size: {vocab_size}")
    logger.info(f"  Using Token IDs -> UNK: {config.UNK_TOKEN_ID}, BOS: {config.BOS_TOKEN_ID}, EOS: {config.EOS_TOKEN_ID}, PAD: (SentencePiece uses -1 for PAD during training if specified, wrapper handles explicit PAD_ID)")

    # SentencePiece requires these to be strings. pad_id=-1 means it's not treated as a special token *during training in the same way as others*.
    # The TokenizerWrapper will handle the config.PAD_TOKEN_ID for actual padding.
    spm_unk_id = str(config.UNK_TOKEN_ID)
    spm_bos_id = str(config.BOS_TOKEN_ID)
    spm_eos_id = str(config.EOS_TOKEN_ID)
    # spm_pad_id = str(config.PAD_TOKEN_ID) # Typically, SentencePiece is trained without a PAD token in vocabulary.
                                         # If you need it in the vocab (e.g. if it's very frequent character not covered)
                                         # then you might add it as a user_defined_symbol.
                                         # Using pad_id=-1 in train call tells SP to not assign it as a piece type.
                                         # Our TokenizerWrapper will assign config.PAD_TOKEN_ID for padding.

    train_command = (
        f"--input={TRAIN_TEXT_FILE} "
        f"--model_prefix={model_prefix_path} "
        f"--vocab_size={vocab_size} "
        f"--model_type=bpe "
        f"--character_coverage=0.9995 " # Try to cover most characters
        f"--input_sentence_size=0 " # 0 means use all sentences from input.
        f"--shuffle_input_sentence=true "
        f"--unk_id={spm_unk_id} "
        f"--bos_id={spm_bos_id} "
        f"--eos_id={spm_eos_id} "
        f"--pad_id=-1 " #  Tell SentencePiece not to use a PAD piece from its types. We use config.PAD_TOKEN_ID.
        # Consider user_defined_symbols if you want to ensure certain tokens like <PAD> are definitely in vocab:
        # f"--user_defined_symbols=<PAD>" (but then ensure PAD_TOKEN_ID matches its assigned ID).
        # For now, this setup is common.
    )
    
    # Check if training file size justifies train_extremely_large_corpus
    # Threshold e.g. 2GB
    if os.path.exists(TRAIN_TEXT_FILE) and os.path.getsize(TRAIN_TEXT_FILE) > (2 * 1024**3) :
        train_command += " --train_extremely_large_corpus=true"
        logger.info("  Enabled train_extremely_large_corpus for large input file.")
    else:
        train_command += " --train_extremely_large_corpus=false"


    try:
        logger.info(f"Executing SentencePiece Trainer with: {train_command.replace(model_prefix_path, os.path.basename(model_prefix_path))}") # Log cmd without full path for brevity
        spm.SentencePieceTrainer.train(train_command)
        logger.info(f"SentencePiece tokenizer training complete.")
        logger.info(f"Model saved with prefix: {model_prefix_path} (e.g., {model_prefix_path}.model)")
    except Exception as e:
        logger.error(f"Error during SentencePiece training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists(TRAIN_TEXT_FILE):
            try:
                # os.remove(TRAIN_TEXT_FILE)
                # logger.info(f"Temporary training data file {TRAIN_TEXT_FILE} removed.")
                logger.info(f"Temporary training data file {TRAIN_TEXT_FILE} was NOT removed (for inspection).")
            except OSError as e_rem:
                logger.error(f"Error removing temporary file {TRAIN_TEXT_FILE}: {e_rem}")


if __name__ == '__main__':
    # Example: train_sentencepiece_tokenizer(vocab_size=config.FALLBACK_VOCAB_SIZE)
    # Or use a specific vocab size:
    train_sentencepiece_tokenizer(vocab_size=8000)
    logger.info("If training was successful, a new tokenizer model should be in "
                f"{os.path.join(OUTPUT_TOKENIZER_DIR, TOKENIZER_MODEL_PREFIX + '.model')}")