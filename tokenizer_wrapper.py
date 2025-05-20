# --- START OF FILE tokenizer_wrapper.py ---

# tokenizer_wrapper.py
import sentencepiece as spm
import config
import os
import logging

logger = logging.getLogger(__name__)

class TokenizerWrapper:
    def __init__(self, model_path=config.TOKENIZER_MODEL_PATH):
        self.model_path = model_path
        if not os.path.exists(self.model_path):
            abs_model_path = os.path.abspath(self.model_path)
            if not os.path.exists(abs_model_path):
                raise FileNotFoundError(f"SentencePiece model not found at {self.model_path} or {abs_model_path}. Run tokenizer_training.py.")
            self.model_path = abs_model_path # Use absolute path if found

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(self.model_path)

        # Use SentencePiece's special token IDs if they are defined (not -1)
        # Otherwise, fall back to the hardcoded IDs from config.py
        # This makes the tokenizer more self-contained regarding BOS/EOS/UNK if the model was trained with them.
        self.bos_id = self.sp.bos_id() if self.sp.bos_id() != -1 else config.BOS_TOKEN_ID
        self.eos_id = self.sp.eos_id() if self.sp.eos_id() != -1 else config.EOS_TOKEN_ID
        self.unk_id = self.sp.unk_id() if self.sp.unk_id() != -1 else config.UNK_TOKEN_ID
        
        # PAD_ID is not typically part of SentencePiece model training, so always use from config
        self.pad_id = config.PAD_TOKEN_ID

        # Determine vocab_size: it's the SentencePiece vocab size,
        # but if PAD_ID is outside this range, the effective vocab for nn.Embedding needs to be larger.
        self.vocab_size = self.sp.vocab_size()
        if self.pad_id >= self.vocab_size:
            logger.warning(f"PAD_TOKEN_ID ({self.pad_id}) is >= SentencePiece vocab_size ({self.vocab_size}). "
                           f"Effective vocab_size for nn.Embedding will be {self.pad_id + 1}.")
            self.vocab_size = self.pad_id + 1
        
        logger.info(f"TokenizerWrapper initialized from {self.model_path}.")
        logger.info(f"  SentencePiece internal vocab size: {self.sp.vocab_size()}.")
        logger.info(f"  Effective vocab_size for model (incl. PAD_ID {self.pad_id}): {self.vocab_size}")
        logger.info(f"  Using Token IDs -> BOS: {self.bos_id}, EOS: {self.eos_id}, UNK: {self.unk_id}, PAD: {self.pad_id}")


    def encode(self, s: str, add_bos=False, add_eos=False, max_length=None):
        if not isinstance(s, str): s = str(s) # Ensure input is string
        token_ids = self.sp.encode_as_ids(s)
        
        processed_ids = []
        if add_bos: processed_ids.append(self.bos_id)
        processed_ids.extend(token_ids)
        if add_eos: processed_ids.append(self.eos_id)

        if max_length is not None:
            if len(processed_ids) < max_length:
                # Pad
                processed_ids.extend([self.pad_id] * (max_length - len(processed_ids)))
            elif len(processed_ids) > max_length:
                # Truncate
                # Smart truncation: if EOS was added and is now part of overflow, try to keep it.
                if add_eos and processed_ids[-1] == self.eos_id and max_length > 0:
                    # If BOS also added and space allows, keep both
                    if add_bos and processed_ids[0] == self.bos_id and max_length > 1:
                        # Truncate middle content: BOS + content_truncated + EOS
                        processed_ids = [self.bos_id] + processed_ids[1:max_length-1] + [self.eos_id]
                    else:
                        # Truncate content: content_truncated + EOS
                        processed_ids = processed_ids[:max_length-1] + [self.eos_id]
                else: # Simple truncation from the end
                    processed_ids = processed_ids[:max_length]
        return processed_ids

    def decode(self, ids, skip_special_tokens=True):
        # Convert to list if it's a tensor
        if hasattr(ids, 'tolist'): 
            ids_to_decode = ids.tolist()
        else:
            ids_to_decode = list(ids) # Ensure it's a list of integers

        if skip_special_tokens:
            # Filter out PAD, BOS, EOS, UNK based on *this instance's* special token IDs
            # SentencePiece's decode_ids doesn't inherently know about a separate PAD_ID if it wasn't part of its model.
            # Also filter -1 which can sometimes appear from bad data or padding.
            special_ids_to_skip = {self.pad_id, self.bos_id, self.eos_id, self.unk_id, -1}
            ids_to_decode = [idx for idx in ids_to_decode if idx not in special_ids_to_skip]
        
        return self.sp.decode_ids(ids_to_decode)

try:
    global_tokenizer = TokenizerWrapper()
    TOKENIZER_VOCAB_SIZE = global_tokenizer.vocab_size # This is the effective vocab size for the model
except FileNotFoundError as e:
    logger.critical(f"CRITICAL ERROR initializing TokenizerWrapper: {e}")
    global_tokenizer = None
    TOKENIZER_VOCAB_SIZE = config.FALLBACK_VOCAB_SIZE # Fallback if tokenizer model not found
    logger.critical(f"CRITICAL WARNING: Tokenizer failed to load. Using fallback vocab size: {TOKENIZER_VOCAB_SIZE}.")
except Exception as e_gen: # Catch any other exception during init
    logger.critical(f"CRITICAL An unexpected error occurred initializing TokenizerWrapper: {e_gen}")
    import traceback; traceback.print_exc()
    global_tokenizer = None
    TOKENIZER_VOCAB_SIZE = config.FALLBACK_VOCAB_SIZE
    logger.critical(f"CRITICAL WARNING: Tokenizer failed unexpectedly. Using fallback vocab size: {TOKENIZER_VOCAB_SIZE}.")