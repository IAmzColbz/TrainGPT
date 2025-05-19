# nn_model.py
import torch
import torch.nn as nn
import torch.optim as optim
import config
from tokenizer_wrapper import global_tokenizer
import os
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class SimpleTransformerSeq2Seq(nn.Module):
    # ... (__init__, count_parameters, _generate_square_subsequent_mask, _create_padding_mask, forward as before) ...
    def __init__(self, vocab_size, d_model, n_heads, num_encoder_layers, num_decoder_layers, d_ff, dropout, model_run_dir):
        super(SimpleTransformerSeq2Seq, self).__init__()
        self.vocab_size = vocab_size; self.d_model = d_model; self.model_run_dir = model_run_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=config.PAD_TOKEN_ID)
        effective_max_len = max(config.MAX_PROBLEM_STATEMENT_TOKENS, config.MAX_GOLD_SOLUTION_TOKENS, config.MAX_TEXT_PRETRAIN_SEQ_LEN) + 10
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_len=effective_max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout, batch_first=True, activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout, batch_first=True, activation='gelu')
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.optimizer = optim.AdamW(self.parameters(), lr=config.NN_LEARNING_RATE, weight_decay=config.OPTIMIZER_WEIGHT_DECAY)
        self.scheduler = None
        self.criterion = nn.CrossEntropyLoss(ignore_index=config.PAD_TOKEN_ID)
        self.current_phase_epoch = 0; self.current_phase_total_steps = 0
        self.history = {'clm_train_epoch_loss': [], 'clm_validation_loss': [], 'supervised_train_epoch_loss': [], 'supervised_validation_loss': [], 'supervised_validation_bleu': []}
        self.to(self.device)
        print(f"SimpleTransformerSeq2Seq model initialized on {self.device} with {self.count_parameters()} parameters.")
        print(f"  Run Dir: {self.model_run_dir}, PositionalEncoding max_len: {effective_max_len}")

    def count_parameters(self): return sum(p.numel() for p in self.parameters() if p.requires_grad)
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz, device=self.device)) == 1).transpose(0, 1)
        return mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    def _create_padding_mask(self, sequence): return (sequence == config.PAD_TOKEN_ID).to(self.device)
    def forward(self, src_token_ids, tgt_token_ids_for_training):
        src_padding_mask = self._create_padding_mask(src_token_ids); tgt_padding_mask = self._create_padding_mask(tgt_token_ids_for_training)
        tgt_causal_mask = self._generate_square_subsequent_mask(tgt_token_ids_for_training.size(1))
        src_emb = self.positional_encoding(self.token_embedding(src_token_ids) * math.sqrt(self.d_model))
        tgt_emb = self.positional_encoding(self.token_embedding(tgt_token_ids_for_training) * math.sqrt(self.d_model))
        memory = self.transformer_encoder(src_emb, src_key_padding_mask=src_padding_mask)
        decoder_output = self.transformer_decoder(tgt_emb, memory, tgt_mask=tgt_causal_mask, tgt_key_padding_mask=tgt_padding_mask, memory_key_padding_mask=src_padding_mask)
        return self.fc_out(decoder_output)

    def generate_solution(self, src_token_ids_batch, 
                          max_len=config.MAX_GOLD_SOLUTION_TOKENS, 
                          temperature=0.7, 
                          repetition_penalty=1.0, 
                          top_p=0.0, 
                          top_k=0): # Ensure all new params are in the signature
        self.eval()
        batch_size = src_token_ids_batch.size(0)
        src_padding_mask = self._create_padding_mask(src_token_ids_batch)

        src_emb = self.positional_encoding(self.token_embedding(src_token_ids_batch) * math.sqrt(self.d_model))
        memory = self.transformer_encoder(src_emb, src_key_padding_mask=src_padding_mask)

        tgt_tokens_batch = torch.full((batch_size, 1), config.BOS_TOKEN_ID, dtype=torch.long, device=self.device)

        for step in range(max_len - 1):
            tgt_len = tgt_tokens_batch.size(1)
            tgt_causal_mask = self._generate_square_subsequent_mask(tgt_len)
            tgt_emb = self.positional_encoding(self.token_embedding(tgt_tokens_batch) * math.sqrt(self.d_model))
            
            decoder_output = self.transformer_decoder(tgt_emb, memory,
                                                      tgt_mask=tgt_causal_mask,
                                                      memory_key_padding_mask=src_padding_mask)
            
            current_step_logits = self.fc_out(decoder_output[:, -1, :])

            # Apply Repetition Penalty (if penalty is not 1.0 and not the first token)
            if repetition_penalty != 1.0 and step > 0:
                for batch_idx in range(batch_size):
                    generated_tokens_for_this_sample = tgt_tokens_batch[batch_idx, 1:] # Exclude BOS
                    for token_id_generated in generated_tokens_for_this_sample:
                        # Avoid penalizing special utility tokens excessively
                        if token_id_generated.item() in [config.PAD_TOKEN_ID, config.UNK_TOKEN_ID, config.EOS_TOKEN_ID, config.BOS_TOKEN_ID]:
                            continue
                        if current_step_logits[batch_idx, token_id_generated] > 0:
                            current_step_logits[batch_idx, token_id_generated] /= repetition_penalty
                        else:
                            current_step_logits[batch_idx, token_id_generated] *= repetition_penalty
            
            # Apply Temperature (on logits directly)
            if temperature > 0.0 and temperature != 1.0 : # Avoid division by zero or no-op
                current_step_logits = current_step_logits / temperature
            
            # Get probabilities after temperature
            probs = torch.softmax(current_step_logits, dim=-1)

            # Apply Top-k filtering
            if top_k > 0:
                top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
                # Create a mask for all other tokens
                non_top_k_mask = torch.ones_like(probs, dtype=torch.bool).scatter_(-1, top_k_indices, False)
                probs[non_top_k_mask] = 0.0
                if torch.sum(probs, dim=-1, keepdim=True).min() > 0: # Re-normalize if necessary
                    probs = probs / torch.sum(probs, dim=-1, keepdim=True)
                else: # Fallback if all top_k probs became zero (highly unlikely)
                    # This might happen if top_k is too large and many probs are tiny
                    # Let's just proceed, multinomial might handle it or we can add a small epsilon
                    pass


            # Apply Top-p (Nucleus) filtering (can be applied after top-k or independently)
            # If applied after top-k, it's on the already filtered distribution
            if top_p > 0.0 and top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0 
                
                indices_to_remove = torch.zeros_like(probs, dtype=torch.bool).scatter_(
                    dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                )
                probs[indices_to_remove] = 0.0
                
                if torch.sum(probs, dim=-1, keepdim=True).min() == 0:
                     # If all probabilities are zero after top-p (e.g., top_p too small for distribution)
                     # Fallback: use the original logits before top_p to pick greedily or with temp
                     # This situation needs careful handling. For now, let's assume it won't make all probs zero.
                     # Or, we could add a tiny epsilon to all tokens before this step if sum is 0.
                    pass # Let multinomial handle potentially all-zero probs (it might error)
                else:
                    probs = probs / torch.sum(probs, dim=-1, keepdim=True) # Re-normalize

            # Select next token
            # If temperature was 0 initially (greedy intention) but top_k or top_p are used, we still sample from modified probs
            if temperature == 0.0 and not (top_k > 0 or (top_p > 0.0 and top_p < 1.0)): # True greedy only if no other sampling
                _, next_token_batch = torch.max(current_step_logits, dim=1) # Use original logits for pure greedy
                next_token_batch = next_token_batch.unsqueeze(1)
            else: # Sampling (could be from temp-scaled, top-k, or top-p modified distribution)
                # Ensure probs are valid for multinomial
                if torch.isnan(probs).any() or torch.isinf(probs).any() or (torch.sum(probs, dim=-1) == 0).any():
                    print("Warning: Invalid probabilities for multinomial sampling. Defaulting to greedy on raw logits.")
                    _, next_token_batch = torch.max(self.fc_out(decoder_output[:, -1, :]), dim=1) # Fallback
                    next_token_batch = next_token_batch.unsqueeze(1)
                else:
                    next_token_batch = torch.multinomial(probs, num_samples=1)

            tgt_tokens_batch = torch.cat([tgt_tokens_batch, next_token_batch], dim=1)
            if (next_token_batch == config.EOS_TOKEN_ID).all(): break
        
        output_strings = [global_tokenizer.decode(ids.tolist()) for ids in tgt_tokens_batch]
        return output_strings[0] if batch_size == 1 else output_strings

    # ... (save_checkpoint and load_checkpoint methods as before) ...
    def save_checkpoint(self, epoch_num, checkpoint_name, is_best=False):
        if not self.model_run_dir: print("Error: model_run_dir not set."); return
        full_path = os.path.join(self.model_run_dir, checkpoint_name)
        os.makedirs(self.model_run_dir, exist_ok=True)
        state = {'epoch': epoch_num, 'model_state_dict': self.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(),
                 'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None, 'history': self.history,
                 'tokenizer_vocab_size_at_save': global_tokenizer.vocab_size if global_tokenizer else None,
                 'model_config_snapshot': {'D_MODEL': self.d_model, 'N_HEADS': self.transformer_encoder.layers[0].self_attn.num_heads,
                                           'NUM_ENCODER_LAYERS': self.transformer_encoder.num_layers, 'NUM_DECODER_LAYERS': self.transformer_decoder.num_layers,
                                           'D_FF': self.transformer_encoder.layers[0].linear1.out_features, 'TRANSFORMER_DROPOUT': self.transformer_encoder.layers[0].dropout.p,
                                           'TOKENIZER_VOCAB_SIZE': self.vocab_size}}
        torch.save(state, full_path)
        print(f"Model checkpoint (Epoch: {epoch_num}) saved to {full_path}")

    def load_checkpoint(self, checkpoint_name):
        if not self.model_run_dir: return 0
        load_path = os.path.join(self.model_run_dir, checkpoint_name)
        if not os.path.exists(load_path):
            print(f"No checkpoint found at {load_path}. Starting fresh for this phase.")
            if checkpoint_name == config.CLM_PRETRAIN_CHECKPOINT_FILENAME: self.history['clm_train_epoch_loss'], self.history['clm_validation_loss'] = [], []
            elif checkpoint_name in [config.SUPERVISED_CHECKPOINT_FILENAME, config.BEST_MODEL_SUPERVISED_CHECKPOINT_FILENAME]: self.history.update({'supervised_train_epoch_loss': [], 'supervised_validation_loss': [], 'supervised_validation_bleu': []})
            return 0
        try:
            checkpoint = torch.load(load_path, map_location=self.device)
            chkpt_vocab_size, current_tokenizer_vocab_size = checkpoint.get('tokenizer_vocab_size_at_save'), global_tokenizer.vocab_size if global_tokenizer else None
            if chkpt_vocab_size is not None and current_tokenizer_vocab_size is not None and chkpt_vocab_size != current_tokenizer_vocab_size:
                print(f"CRITICAL WARNING: Vocab size mismatch. Chkpt: {chkpt_vocab_size}, Current: {current_tokenizer_vocab_size}. Checkpoint NOT loaded."); return 0
            self.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint: self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint and self.scheduler: self.scheduler.load_state_dict(checkpoint['scheduler_state_dict']) # Only if self.scheduler is already created
            self.history.update(checkpoint.get('history', {}))
            last_completed_epoch = checkpoint.get('epoch', -1); self.current_phase_epoch = last_completed_epoch
            print(f"Model checkpoint loaded from {load_path}. Resuming phase from epoch {last_completed_epoch + 1}.")
            return last_completed_epoch + 1
        except Exception as e: print(f"Error loading checkpoint from {load_path}: {e}. Starting fresh."); return 0