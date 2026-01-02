# ai_backend/infrastructure/storage/quantizer.py
import numpy as np

class Quantizer:
    def __init__(self, bits=4):
        self.bits = bits
        self.q_min = -(2**(bits-1))    # -8 for 4-bit
        self.q_max = (2**(bits-1)) - 1 # 7 for 4-bit

    def quantize_block(self, tensor):
        """
        Shrinks weights using Scale and Zero-point.
        formula: q = round(f / scale)
        """
        # 1. Calculate Scale (The 'stretch' factor)
        max_val = np.max(np.abs(tensor))
        if max_val == 0:
            return tensor.astype(np.int8), 1.0
            
        scale = max_val / self.q_max
        
        # 2. Quantize and Clip
        q_tensor = np.round(tensor / scale).astype(np.int8)
        q_tensor = np.clip(q_tensor, self.q_min, self.q_max)
        
        return q_tensor, scale

    def pack_4bit(self, q_tensor):
        """
        Packs two 4-bit numbers into one 8-bit byte to save 50% more space.
        """
        # Logic: (Weight_A << 4) | (Weight_B & 0x0F)
        # This effectively halves the file size on your 2TB drive.
        pass