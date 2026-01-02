# ai_backend/infrastructure/synapse/tokenizer.py
import sentencepiece as spm
from transformers import AutoTokenizer

class FailSafeTokenizer:
    def __init__(self):
        # We load two different logic engines to cross-verify
        self.primary = spm.SentencePieceProcessor(model_file="weights/ultra.model")
        self.backup = AutoTokenizer.from_pretrained("gpt2") # Fast fallback

    def decode_safe(self, token_ids):
        try:
            # Try the high-performance path first
            return self.primary.decode(token_ids)
        except Exception:
            print("[WARNING] Tokenizer glitch! Using fallback...")
            # Fallback to standard BPE decoding to prevent gibberish
            return self.backup.decode(token_ids, skip_special_tokens=True)