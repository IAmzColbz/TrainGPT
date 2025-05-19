# tokenizer_training.py
import sentencepiece as spm
import glob
import os
import pandas as pd
from utils import extract_python_code_from_markdown
import config # Import the main config file

# Paths from config for consistency
OUTPUT_TOKENIZER_DIR = os.path.dirname(config.TOKENIZER_MODEL_PATH)
TOKENIZER_MODEL_PREFIX = 'custom_tokenizer' # Filename prefix for .model and .vocab
TRAIN_TEXT_FILE = os.path.join(OUTPUT_TOKENIZER_DIR, "tokenizer_train_data.txt")

# Configuration for how much data to use for tokenizer training
NUM_CODE_DATASET_FILES_FOR_TOKENIZER = 5
NUM_FINEWEB_FILES_FOR_TOKENIZER = 3 # Number of FineWeb files to process
# FineWeb files can be large, consider processing only a portion of rows from each
MAX_ROWS_PER_FINEWEB_FILE_FOR_TOKENIZER = 100000 # Limit rows per FineWeb file to save memory further

TARGET_VOCAB_SIZE_TOKENIZER = config.FALLBACK_VOCAB_SIZE


def stream_text_for_tokenizer(
    output_train_file_path, # Path to write the combined text
    num_code_files_to_process=NUM_CODE_DATASET_FILES_FOR_TOKENIZER,
    num_fineweb_files_to_process=NUM_FINEWEB_FILES_FOR_TOKENIZER,
    max_rows_per_fineweb_file=MAX_ROWS_PER_FINEWEB_FILE_FOR_TOKENIZER
    ):
    os.makedirs(os.path.dirname(output_train_file_path), exist_ok=True)
    total_segments_written = 0

    try:
        with open(output_train_file_path, 'w', encoding='utf-8') as outfile:
            # --- 1. Stream from Code Problem/Solution Dataset ---
            code_dataset_search_pattern = os.path.join(config.DATASET_DIR, "train-*-of-*.parquet")
            code_file_paths = sorted(glob.glob(code_dataset_search_pattern))

            if not code_file_paths:
                print(f"No Code Problem/Solution Parquet files found in {config.DATASET_DIR}.")
            else:
                print(f"Streaming text from up to {num_code_files_to_process} Code Problem/Solution Parquet files...")
                files_processed_code = 0
                for filepath in code_file_paths:
                    if files_processed_code >= num_code_files_to_process:
                        break
                    try:
                        df = pd.read_parquet(filepath) # Loads one file into memory
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
                        df = None # Explicitly free DataFrame memory
                        files_processed_code += 1
                        print(f"  Streamed Code Dataset: {os.path.basename(filepath)}")
                    except Exception as e:
                        print(f"Error processing Code Dataset file {filepath}: {e}")

            # --- 2. Stream from FineWeb Text Dataset ---
            fineweb_search_pattern = os.path.join(config.FINEWEB_DATA_DIR, "*.parquet")
            fineweb_file_paths = sorted(glob.glob(fineweb_search_pattern))

            if not fineweb_file_paths:
                print(f"No FineWeb Parquet files found in {config.FINEWEB_DATA_DIR}.")
            else:
                print(f"Streaming text from up to {num_fineweb_files_to_process} FineWeb Parquet files (max {max_rows_per_fineweb_file} rows per file)...")
                files_processed_fineweb = 0
                for filepath in fineweb_file_paths:
                    if files_processed_fineweb >= num_fineweb_files_to_process:
                        break
                    try:
                        df = pd.read_parquet(filepath) # Loads one file into memory
                        if 'text' in df.columns:
                            rows_processed_in_file = 0
                            for text_doc in df['text'].dropna(): # Iterate directly to avoid list conversion
                                if rows_processed_in_file >= max_rows_per_fineweb_file:
                                    print(f"    Reached max row limit ({max_rows_per_fineweb_file}) for {os.path.basename(filepath)}")
                                    break
                                if text_doc and isinstance(text_doc, str):
                                    outfile.write(text_doc.strip().replace('\n', ' ') + '\n')
                                    total_segments_written += 1
                                    rows_processed_in_file += 1
                        else:
                            print(f"  Warning: Column 'text' not found in FineWeb file {os.path.basename(filepath)}. Skipping.")
                        df = None # Explicitly free DataFrame memory
                        files_processed_fineweb += 1
                        print(f"  Streamed FineWeb Dataset: {os.path.basename(filepath)}")
                    except Exception as e:
                        print(f"Error processing FineWeb file {filepath}: {e}")

    except Exception as e_outer:
        print(f"Error opening or writing to output file {output_train_file_path}: {e_outer}")
        return False, 0 # Indicate failure

    if total_segments_written == 0:
        print("No text data streamed from any source. Tokenizer training cannot proceed.")
        return False, 0

    print(f"\nTotal text segments streamed to {output_train_file_path}: {total_segments_written}")
    return True, total_segments_written


def train_sentencepiece_tokenizer(vocab_size=TARGET_VOCAB_SIZE_TOKENIZER):
    # Stream data to the training file
    success, num_segments = stream_text_for_tokenizer(TRAIN_TEXT_FILE)
    if not success or num_segments == 0:
        print("Failed to prepare training data for tokenizer.")
        return

    model_prefix_path = os.path.join(OUTPUT_TOKENIZER_DIR, TOKENIZER_MODEL_PREFIX)

    print(f"\nStarting SentencePiece tokenizer training...")
    print(f"  Input text file: {TRAIN_TEXT_FILE} ({num_segments} lines)")
    print(f"  Output model prefix: {model_prefix_path}")
    print(f"  Target vocabulary size: {vocab_size}")

    try:
        spm.SentencePieceTrainer.train(
            input=TRAIN_TEXT_FILE,
            model_prefix=model_prefix_path,
            vocab_size=vocab_size,
            model_type='bpe',
            character_coverage=0.9995,
            input_sentence_size=num_segments, # Max sentences to use from input (0 uses all)
                                             # Use a large number if you want to limit but still use most
                                             # If 0, it reads until EOF.
                                             # Let's process all written lines.
            shuffle_input_sentence=True,     # Shuffle input sentences before training (good for large files)
            unk_id=config.UNK_TOKEN_ID,
            bos_id=config.BOS_TOKEN_ID,
            eos_id=config.EOS_TOKEN_ID,
            pad_id=-1,
            # Consider train_extremely_large_corpus=True if TRAIN_TEXT_FILE becomes huge (multi-GB)
            # This enables memory-efficient processing within SentencePiece itself.
            # For now, False is likely fine with the streaming write.
            train_extremely_large_corpus= (True if os.path.getsize(TRAIN_TEXT_FILE) > (2 * 1024**3) else False) # e.g. > 2GB
        )
        print(f"SentencePiece tokenizer training complete.")
        print(f"Model saved with prefix: {model_prefix_path} (e.g., {model_prefix_path}.model)")
    except Exception as e:
        print(f"Error during SentencePiece training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Optionally, delete the large temporary training text file
        if os.path.exists(TRAIN_TEXT_FILE):
            try:
                # os.remove(TRAIN_TEXT_FILE)
                # print(f"Temporary training data file {TRAIN_TEXT_FILE} removed.")
                print(f"Temporary training data file {TRAIN_TEXT_FILE} was NOT removed (for inspection).")
            except OSError as e:
                print(f"Error removing temporary file {TRAIN_TEXT_FILE}: {e}")


if __name__ == '__main__':
    train_sentencepiece_tokenizer(vocab_size=8000)