# ai_backend/infrastructure/synapse/sampler.py
import torch.nn.functional as F

def sample_next_token(logits, temperature=0.7, top_p=0.9):
    """
    Filters the AI's top guesses to find the most 'intelligent' next word.
    """
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    # Apply Top-P logic to cut off the 'low probability' garbage
    return nucleus_sampling(probs, top_p)