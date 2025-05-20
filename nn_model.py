# --- START OF FILE nn_model.py ---

# nn_model.py
import traceback
import torch
import torch.nn as nn
import torch.optim as optim
import os
import math
import json
import logging
from typing import Dict, Any, Optional, List

import config # Main project config
from tokenizer_wrapper import global_tokenizer # Access to global tokenizer

logger = logging.getLogger(__name__)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = config.POSITIONAL_ENCODING_MAX_LEN):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        if max_len <= 0:
            logger.warning(f"PositionalEncoding max_len was {max_len}, defaulting to 1. This is likely an error in config.")
            max_len = 1 # Prevent errors with torch.arange if max_len is 0 or negative

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        logger.info(f"PositionalEncoding initialized with d_model={d_model}, max_len={max_len}, dropout={dropout}. PE shape: {self.pe.shape}")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, d_model]
        # self.pe shape: [1, max_len, d_model]
        if x.size(1) > self.pe.size(1):
            logger.error(f"Input sequence length ({x.size(1)}) exceeds PositionalEncoding max_len ({self.pe.size(1)}). Truncating or re-initialize PE.")
            # This case should ideally be prevented by input tokenization max_length settings.
            # Or, if dynamic PE resizing is needed, it would be more complex.
            # For now, we rely on the PE being large enough.
            x = x + self.pe[:, :self.pe.size(1)] # Use full PE buffer if input is too long (incorrect but avoids crash)
        else:
            x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class SimpleTransformerSeq2Seq(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 d_model: int = config.D_MODEL,
                 n_heads: int = config.N_HEADS,
                 num_encoder_layers: int = config.NUM_ENCODER_LAYERS,
                 num_decoder_layers: int = config.NUM_DECODER_LAYERS,
                 d_ff: int = config.D_FF,
                 dropout: float = config.TRANSFORMER_DROPOUT,
                 model_run_dir: Optional[str] = None, 
                 learning_rate: float = config.NN_LEARNING_RATE,
                 weight_decay: float = config.OPTIMIZER_WEIGHT_DECAY,
                 pad_token_id: int = config.PAD_TOKEN_ID,
                 bos_token_id: int = config.BOS_TOKEN_ID,
                 eos_token_id: int = config.EOS_TOKEN_ID,
                 positional_encoding_max_len: int = config.POSITIONAL_ENCODING_MAX_LEN
                 ):
        super(SimpleTransformerSeq2Seq, self).__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.d_ff = d_ff
        self.dropout_rate = dropout
        self.model_run_dir = model_run_dir
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        # Store pe_max_len for reference, actual PE module handles its own max_len
        self.positional_encoding_max_len_config = positional_encoding_max_len 
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model, padding_idx=self.pad_token_id)
        self.positional_encoding = PositionalEncoding(self.d_model, self.dropout_rate, max_len=positional_encoding_max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=self.n_heads, dim_feedforward=self.d_ff,
            dropout=self.dropout_rate, batch_first=True, activation='gelu', norm_first=True 
        )
        encoder_norm = nn.LayerNorm(self.d_model) 
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_encoder_layers, norm=encoder_norm)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model, nhead=self.n_heads, dim_feedforward=self.d_ff,
            dropout=self.dropout_rate, batch_first=True, activation='gelu', norm_first=True 
        )
        decoder_norm = nn.LayerNorm(self.d_model) 
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.num_decoder_layers, norm=decoder_norm)
        
        self.fc_out = nn.Linear(self.d_model, self.vocab_size)

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)
        self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        for group in self.optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])

        self.scheduler = None 

        self.current_phase_step_or_epoch: int = 0 # 0-indexed, stores *last completed* step/epoch
        self.history: Dict[str, List] = {
            'prefixlm_batch_loss': [],
            'supervised_train_epoch_loss': [],
            'supervised_validation_loss': [],
        }
        
        self._init_weights() 
        self.to(self.device)
        logger.info(f"SimpleTransformerSeq2Seq initialized on {self.device} with {self.count_parameters():,} parameters.")
        logger.info(f"  Architecture: D_model={d_model}, Heads={n_heads}, EncL={num_encoder_layers}, DecL={num_decoder_layers}, D_ff={d_ff}, PE_max_len={self.positional_encoding.pe.size(1)}")
        if self.model_run_dir:
            logger.info(f"  Model run directory: {self.model_run_dir}")

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        if hasattr(self, 'token_embedding'): # Should always exist
             nn.init.normal_(self.token_embedding.weight, mean=0.0, std=self.d_model**-0.5)
             if self.token_embedding.padding_idx is not None:
                 with torch.no_grad():
                     self.token_embedding.weight[self.token_embedding.padding_idx].fill_(0)


    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _generate_square_subsequent_mask(self, size: int) -> torch.Tensor:
        return torch.triu(torch.full((size, size), float('-inf'), device=self.device), diagonal=1)

    def _create_padding_mask(self, sequence: torch.Tensor) -> torch.Tensor:
        return (sequence == self.pad_token_id) 

    def forward(self, src_token_ids: torch.Tensor, tgt_token_ids_for_training: torch.Tensor) -> torch.Tensor:
        src_padding_mask = self._create_padding_mask(src_token_ids)
        tgt_padding_mask = self._create_padding_mask(tgt_token_ids_for_training)
        tgt_causal_mask = self._generate_square_subsequent_mask(tgt_token_ids_for_training.size(1))

        src_emb = self.positional_encoding(self.token_embedding(src_token_ids) * math.sqrt(self.d_model))
        tgt_emb = self.positional_encoding(self.token_embedding(tgt_token_ids_for_training) * math.sqrt(self.d_model))

        memory = self.transformer_encoder(src_emb, src_key_padding_mask=src_padding_mask)
        
        decoder_output = self.transformer_decoder(
            tgt_emb, memory,
            tgt_mask=tgt_causal_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask
        )
        return self.fc_out(decoder_output)

    @torch.no_grad()
    def generate_solution(self,
                          src_token_ids_batch: torch.Tensor,
                          max_len: int = config.MAX_GOLD_SOLUTION_TOKENS,
                          temperature: float = 0.7,
                          repetition_penalty: float = 1.0,
                          top_p: float = 0.9,
                          top_k: int = 0
                          ) -> List[str]:
        self.eval()
        batch_size = src_token_ids_batch.size(0)

        src_padding_mask = self._create_padding_mask(src_token_ids_batch)
        src_emb = self.positional_encoding(self.token_embedding(src_token_ids_batch) * math.sqrt(self.d_model))
        memory = self.transformer_encoder(src_emb, src_key_padding_mask=src_padding_mask)

        generated_sequences = torch.full((batch_size, 1), self.bos_token_id, dtype=torch.long, device=self.device)
        active_sequences = torch.ones(batch_size, dtype=torch.bool, device=self.device)

        for _ in range(max_len -1): 
            if not active_sequences.any(): break

            current_tgt_tokens = generated_sequences[active_sequences]
            # Ensure current_tgt_tokens does not exceed PE capacity; should be handled by max_len check too
            if current_tgt_tokens.size(1) >= self.positional_encoding.pe.size(1):
                logger.warning(f"Generation exceeded PE max_len ({self.positional_encoding.pe.size(1)}). Stopping early for affected sequences.")
                # This logic might need refinement if we want to handle this more gracefully for some active sequences
                # For now, it implies that if any sequence hits this, all active generation might be less reliable.
                # A simpler approach: if generation length for any sequence gets too long, it's capped by outer loop `max_len`.
                break 

            tgt_causal_mask = self._generate_square_subsequent_mask(current_tgt_tokens.size(1))
            tgt_emb = self.positional_encoding(self.token_embedding(current_tgt_tokens) * math.sqrt(self.d_model))
            
            active_memory = memory[active_sequences]
            active_src_padding_mask = src_padding_mask[active_sequences] if src_padding_mask is not None else None

            decoder_output = self.transformer_decoder(
                tgt_emb, active_memory,
                tgt_mask=tgt_causal_mask,
                memory_key_padding_mask=active_src_padding_mask
            )
            
            current_step_logits = self.fc_out(decoder_output[:, -1, :]) # Logits for the last token

            if repetition_penalty != 1.0 and current_tgt_tokens.size(1) > 0:
                for i in range(current_step_logits.size(0)): # Iterate over batch items
                    penalized_tokens = current_tgt_tokens[i]
                    for token_id_penalize in penalized_tokens:
                        # Avoid penalizing special tokens, could be configured
                        if token_id_penalize.item() in [self.pad_token_id, self.unk_id, self.eos_token_id, self.bos_token_id]: 
                            continue
                        current_step_logits[i, token_id_penalize] /= repetition_penalty 

            next_tokens_for_active = self._sample_next_token(current_step_logits, temperature, top_k, top_p)
            
            full_batch_next_tokens = torch.full((batch_size, 1), self.pad_token_id, dtype=torch.long, device=self.device)
            full_batch_next_tokens[active_sequences] = next_tokens_for_active.unsqueeze(-1)
            
            generated_sequences = torch.cat([generated_sequences, full_batch_next_tokens], dim=1)
            # Update active_sequences: still active if not EOS and not padded (padded implies it was inactive before)
            active_sequences = active_sequences & (full_batch_next_tokens.squeeze(1) != self.eos_token_id)

        output_strings = []
        for i in range(batch_size):
            seq_tokens = generated_sequences[i, :].tolist()
            try: 
                eos_idx = seq_tokens.index(self.eos_token_id)
            except ValueError: 
                eos_idx = len(seq_tokens) # EOS not found, take whole sequence
            
            # Decode from after BOS_ID up to (but not including) EOS_ID
            # Filter PAD_ID before decoding
            clean_tokens = [tok for tok in seq_tokens[1:eos_idx] if tok != self.pad_token_id] 
            output_strings.append(global_tokenizer.decode(clean_tokens, skip_special_tokens=True)) # skip_special_tokens in decode also removes BOS/EOS/UNK
        return output_strings

    def _sample_next_token(self, logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
        if temperature == 0.0: # Greedy decoding
            return torch.argmax(logits, dim=-1)
        
        logits = logits / temperature # Apply temperature

        # Apply Top-K filtering
        if top_k > 0:
            k = min(top_k, logits.size(-1)) # Ensure k is not larger than vocabulary size
            top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)
            # Create a mask for filtering
            mask = torch.full_like(logits, float('-inf'))
            mask.scatter_(-1, top_k_indices, top_k_logits)
            logits = mask
        
        # Apply Top-P (nucleus) filtering
        if top_p > 0.0 and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            
            # Find indices to remove (those with cumulative prob > top_p)
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift one to the right to keep the first token that exceeds top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0 # Never remove the most probable token
            
            # Create a mask for filtering based on original indices
            indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf') # Set logits of tokens to remove to -inf
            
        probs = torch.softmax(logits, dim=-1)

        # Handle potential NaN/Inf in probabilities or all-zero probabilities after filtering
        if torch.isnan(probs).any() or torch.isinf(probs).any() or (torch.sum(probs, dim=-1) == 0).any():
            logger.warning("Invalid probabilities (NaN/Inf/all-zero) for multinomial sampling. Falling back to argmax on filtered logits.")
            # Fallback: if all probs are zero or invalid, pick the highest logit among the filtered ones.
            # If even logits are all -inf, this will pick the first one (index 0).
            return torch.argmax(logits, dim=-1) 
            
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def _get_checkpoint_architecture_config(self) -> Dict[str, Any]:
        """Returns a dictionary of the model's current architecture parameters."""
        return {
            'vocab_size': self.vocab_size, 
            'd_model': self.d_model, 
            'n_heads': self.n_heads,
            'num_encoder_layers': self.num_encoder_layers, 
            'num_decoder_layers': self.num_decoder_layers,
            'd_ff': self.d_ff, 
            'dropout_rate': self.dropout_rate, 
            'pad_token_id': self.pad_token_id,
            'bos_token_id': self.bos_token_id, 
            'eos_token_id': self.eos_token_id,
            'positional_encoding_max_len': self.positional_encoding.pe.size(1) # Actual PE buffer max_len
        }

    def save_checkpoint(self, current_step_or_epoch_completed: int, checkpoint_name: str, is_best: bool = False):
        if not self.model_run_dir: logger.error("Model run directory not set. Cannot save checkpoint."); return
        full_path = os.path.join(self.model_run_dir, checkpoint_name); os.makedirs(self.model_run_dir, exist_ok=True)
        state = {
            'phase_step_or_epoch_completed': current_step_or_epoch_completed, # Store 0-indexed completed step/epoch
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'history': self.history, 
            'tokenizer_vocab_size_at_save': global_tokenizer.vocab_size if global_tokenizer else None,
            'model_architecture_config': self._get_checkpoint_architecture_config() # Save current arch
        }
        torch.save(state, full_path)
        logger.info(f"Checkpoint (Completed Step/Epoch: {current_step_or_epoch_completed}) saved to {full_path}")
        if is_best: # Typically for supervised validation
            if "supervised" in checkpoint_name.lower(): # Be more specific if needed
                 best_path = os.path.join(self.model_run_dir, config.BEST_MODEL_SUPERVISED_CHECKPOINT_FILENAME)
                 torch.save(state, best_path); logger.info(f"Best supervised model checkpoint also saved to {best_path}")

    def load_checkpoint(self, checkpoint_name: str) -> int:
        """
        Loads a checkpoint.
        Returns:
            int: The next step/epoch number to start training from (last_completed + 1).
                 Returns 0 if checkpoint not found or load failed.
        """
        if not self.model_run_dir: 
            logger.warning("Model run directory not set for loading checkpoint. Cannot load.")
            return 0 
            
        load_path = os.path.join(self.model_run_dir, checkpoint_name)
        if not os.path.exists(load_path):
            logger.info(f"No checkpoint found at {load_path}. Phase will start fresh or use other checkpoints if applicable.")
            self.current_phase_step_or_epoch = 0 # Ensure it's 0 if no checkpoint loaded here
            return 0

        try:
            checkpoint = torch.load(load_path, map_location=self.device)
            logger.info(f"Attempting to load checkpoint from {load_path}...")

            # --- Vocab Size Compatibility Check ---
            chkpt_tokenizer_vocab = checkpoint.get('tokenizer_vocab_size_at_save')
            current_tokenizer_vocab = global_tokenizer.vocab_size if global_tokenizer else None
            if chkpt_tokenizer_vocab is not None and current_tokenizer_vocab is not None and \
               chkpt_tokenizer_vocab != current_tokenizer_vocab:
                logger.error(f"VOCAB SIZE MISMATCH! Checkpoint: {chkpt_tokenizer_vocab}, Current Global Tokenizer: {current_tokenizer_vocab}. Checkpoint NOT loaded.")
                return 0

            # --- Architecture Compatibility (if arch config is in checkpoint) ---
            saved_arch_config = checkpoint.get('model_architecture_config')
            current_model_arch_for_comparison = self._get_checkpoint_architecture_config()

            if saved_arch_config:
                logger.info("Checkpoint contains architecture config. Verifying compatibility...")
                critical_arch_keys = ['vocab_size', 'd_model', 'n_heads', 
                                      'num_encoder_layers', 'num_decoder_layers', 'd_ff']
                arch_mismatch = False
                for key in critical_arch_keys:
                    saved_val = saved_arch_config.get(key)
                    current_val = current_model_arch_for_comparison.get(key)
                    if saved_val is not None and current_val is not None and saved_val != current_val:
                        logger.error(f"CRITICAL ARCHITECTURE MISMATCH in '{key}'! Checkpoint: {saved_val}, Current Model: {current_val}.")
                        arch_mismatch = True
                
                if arch_mismatch:
                    logger.error(f"Checkpoint {load_path} NOT loaded due to critical architecture mismatch with current model instance.")
                    return 0
                logger.info("Architecture compatibility verified with checkpoint's saved config.")
            else:
                logger.warning(f"Checkpoint {load_path} lacks 'model_architecture_config'. "
                               "Full architecture compatibility cannot be pre-verified. "
                               "Relying on successful model instantiation with correct parameters from train_orchestrator/chat CLI.")

            # --- Handle Positional Encoding (PE) Tensor Mismatch Directly ---
            model_state_dict = checkpoint['model_state_dict']
            if 'positional_encoding.pe' in model_state_dict:
                chkpt_pe_shape = model_state_dict['positional_encoding.pe'].shape
                curr_pe_shape = self.positional_encoding.pe.shape # PE of the current model instance
                if chkpt_pe_shape != curr_pe_shape:
                    logger.warning(f"Positional encoding 'pe' tensor SIZE MISMATCH. "
                                   f"Checkpoint PE shape: {chkpt_pe_shape}, Current model PE shape: {curr_pe_shape}. "
                                   f"Removing 'positional_encoding.pe' from checkpoint's state_dict. "
                                   "Model will use its own initialized PE tensor.")
                    del model_state_dict['positional_encoding.pe']
            
            # --- Load State Dictionary ---
            # strict=False allows loading if current model has new layers not in checkpoint,
            # or if checkpoint has layers not in current model (e.g. loading older into newer).
            # However, if the *core structure* differs (e.g. num_layers used to init model doesn't match ckpt),
            # this will still fail if the current model *expects* keys that are missing.
            # This is why train_orchestrator/chat MUST init the model with the correct arch first.
            incompatible_keys = self.load_state_dict(model_state_dict, strict=False)
            if incompatible_keys.missing_keys:
                logger.warning(f"Missing keys when loading state_dict (these will use initial values or cause errors if expected): {incompatible_keys.missing_keys}")
            if incompatible_keys.unexpected_keys:
                logger.warning(f"Unexpected keys in state_dict (these were ignored): {incompatible_keys.unexpected_keys}")

            # --- Load Optimizer and Scheduler State ---
            if 'optimizer_state_dict' in checkpoint:
                try:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    logger.info("Optimizer state loaded.")
                    for group in self.optimizer.param_groups: # Ensure initial_lr for scheduler
                        group.setdefault('initial_lr', group['lr'])
                except Exception as e_optim:
                    logger.warning(f"Could not load optimizer state: {e_optim}. Optimizer might start fresh.")
            
            if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None and self.scheduler:
                try:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    logger.info("Scheduler state loaded.")
                except Exception as e_sched:
                    logger.warning(f"Could not load scheduler state: {e_sched}. Scheduler might need reinitialization by training logic.")
            elif checkpoint.get('scheduler_state_dict') is None and self.scheduler:
                logger.info("Checkpoint had no scheduler state (or it was None). Scheduler will be initialized by training logic.")


            self.history.update(checkpoint.get('history', {}))
            # Checkpoint stores 'phase_step_or_epoch_completed' (0-indexed)
            last_completed = checkpoint.get('phase_step_or_epoch_completed', 0) 
            self.current_phase_step_or_epoch = last_completed 
            
            logger.info(f"Checkpoint loaded. Last completed phase step/epoch: {last_completed}. Resuming from step/epoch {last_completed + 1}.")
            return last_completed + 1 # Return the *next* step/epoch to run (1-indexed if epochs, or next step index)
            
        except RuntimeError as e_runtime: 
            logger.error(f"RuntimeError loading state_dict from {load_path}: {e_runtime}")
            logger.error("This often indicates a non-recoverable architecture mismatch (e.g., layer dimensions) "
                         "that wasn't handled by PE deletion or if the model wasn't instantiated correctly before loading.")
            self.current_phase_step_or_epoch = 0; return 0
        except Exception as e:
            logger.error(f"Generic error loading checkpoint from {load_path}: {e}\n{traceback.format_exc()}")
            self.current_phase_step_or_epoch = 0; return 0