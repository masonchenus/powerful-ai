# ai_backend/infrastructure/math_engine/kernels/vector_dist.py
import numpy as np
from .precision_ops import gpu_compute_engine

class VectorDistributor:
    def __init__(self, chunk_size=1_000_000):
        self.chunk_size = chunk_size

    def distribute_load(self, formula_id, input_vector):
        """
        Splits millions of equations into GPU-sized bites.
        """
        total_elements = len(input_vector)
        results = []

        # Process in chunks to prevent VRAM overflow
        for i in range(0, total_elements, self.chunk_size):
            chunk = input_vector[i : i + self.chunk_size]
            
            # Send to the CUDA kernel (precision_ops.cu)
            processed_chunk = gpu_compute_engine(formula_id, chunk)
            results.append(processed_chunk)

        return np.concatenate(results)