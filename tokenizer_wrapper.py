# tokenizer_wrapper.py
import sentencepiece as spm
import config
import os

class TokenizerWrapper:
    def __init__(self, model_path=config.TOKENIZER_MODEL_PATH):
        self.model_path = model_path
        if not os.path.exists(self.model_path):
            abs_model_path = os.path.abspath(self.model_path)
            if not os.path.exists(abs_model_path):
                raise FileNotFoundError(f"SentencePiece model not found at {self.model_path} or {abs_model_path}. Run tokenizer_training.py.")
            self.model_path = abs_model_path

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(self.model_path)

        self.bos_id = self.sp.bos_id() if self.sp.bos_id() != -1 else config.BOS_TOKEN_ID
        self.eos_id = self.sp.eos_id() if self.sp.eos_id() != -1 else config.EOS_TOKEN_ID
        self.unk_id = self.sp.unk_id() if self.sp.unk_id() != -1 else config.UNK_TOKEN_ID
        self.pad_id = config.PAD_TOKEN_ID

        self.vocab_size = self.sp.vocab_size()
        if self.pad_id >= self.vocab_size:
            self.vocab_size = self.pad_id + 1
        
        print(f"TokenizerWrapper initialized from {self.model_path}.")
        print(f"  SentencePiece internal vocab size: {self.sp.vocab_size()}.")
        print(f"  Effective vocab_size for model (incl. PAD_ID {self.pad_id}): {self.vocab_size}")
        print(f"  BOS ID: {self.bos_id}, EOS ID: {self.eos_id}, UNK ID: {self.unk_id}, Model PAD ID: {self.pad_id}")

    def encode(self, s: str, add_bos=False, add_eos=False, max_length=None):
        if not isinstance(s, str): s = str(s)
        token_ids = self.sp.encode_as_ids(s)
        
        processed_ids = []
        if add_bos: processed_ids.append(self.bos_id)
        processed_ids.extend(token_ids)
        if add_eos: processed_ids.append(self.eos_id)

        if max_length is not None:
            if len(processed_ids) < max_length:
                processed_ids.extend([self.pad_id] * (max_length - len(processed_ids)))
            elif len(processed_ids) > max_length:
                if add_eos and processed_ids[-1] == self.eos_id and max_length > 0:
                    if add_bos and processed_ids[0] == self.bos_id and max_length > 1:
                        processed_ids = [self.bos_id] + processed_ids[1:max_length-1] + [self.eos_id]
                    else:
                        processed_ids = processed_ids[:max_length-1] + [self.eos_id]
                else:
                    processed_ids = processed_ids[:max_length]
        return processed_ids

    def decode(self, ids, skip_special_tokens=True):
        ids_to_decode = ids
        if skip_special_tokens:
            ids_to_decode = [idx for idx in ids if idx != self.pad_id and idx != -1]
        return self.sp.decode_ids(ids_to_decode)

try:
    global_tokenizer = TokenizerWrapper()
    TOKENIZER_VOCAB_SIZE = global_tokenizer.vocab_size
except FileNotFoundError as e:
    print(f"CRITICAL ERROR initializing TokenizerWrapper: {e}")
    global_tokenizer = None
    TOKENIZER_VOCAB_SIZE = config.FALLBACK_VOCAB_SIZE
    print(f"CRITICAL WARNING: Tokenizer failed to load. Using fallback vocab size: {TOKENIZER_VOCAB_SIZE}.")
except Exception as e_gen:
    print(f"CRITICAL An unexpected error occurred initializing TokenizerWrapper: {e_gen}")
    import traceback; traceback.print_exc()
    global_tokenizer = None
    TOKENIZER_VOCAB_SIZE = config.FALLBACK_VOCAB_SIZE
    print(f"CRITICAL WARNING: Tokenizer failed unexpectedly. Using fallback vocab size: {TOKENIZER_VOCAB_SIZE}.")