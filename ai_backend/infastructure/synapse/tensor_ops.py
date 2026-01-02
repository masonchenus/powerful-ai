# ai_backend/infrastructure/synapse/tensor_ops.py
import torch

def matmul_quantized(input_tensor, weight_shard, scale):
    """
    Multiplies a 4-bit weight shard by a 16-bit input.
    Uses 'scale' from the quantizer to restore precision.
    """
    # Dequantize on the fly to save VRAM
    w_fp16 = weight_shard.to(torch.float16) * scale
    return torch.matmul(input_tensor, w_fp16)