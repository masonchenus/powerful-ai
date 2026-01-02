# ai_backend/infrastructure/synapse/activations.py
import torch

def swiglu(x):
    """
    The 'Gold Standard' activation for models like Llama 3 and your 248T Ultra.
    Allows for better gradient flow in deep 10,000-dimension systems.
    """
    x, gate = x.chunk(2, dim=-1)
    return x * torch.nn.functional.silu(gate)