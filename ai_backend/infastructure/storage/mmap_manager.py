# ai_backend/infrastructure/storage/mmap_manager.py

import mmap
import os
import numpy as np

class MMapManager:
    def __init__(self, weight_path):
        self.weight_path = weight_path
        self.mappings = {}  # Stores open memory maps for each shard
        self.file_handles = {}

    def map_shard(self, shard_id):
        """
        Maps a specific binary shard (e.g., shard_0001.bin) into virtual memory.
        This does NOT load the shard into RAM yet.
        """
        if shard_id in self.mappings:
            return self.mappings[shard_id]

        file_path = os.path.join(self.weight_path, f"shard_{shard_id}.bin")
        
        # 1. Open file in binary read mode
        f = open(file_path, "rb")
        self.file_handles[shard_id] = f
        
        # 2. Memory map the entire file
        # access=mmap.ACCESS_READ ensures we don't accidentally overwrite weights
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        self.mappings[shard_id] = mm
        
        print(f"[MMAP] Shard {shard_id} mapped to virtual memory.")
        return mm

    def get_weight_tensor(self, shard_id, offset, shape, dtype=np.float16):
        """
        Retrieves a specific 'slice' of the brain (w or b).
        The OS only pulls these bytes from the SSD at this exact moment.
        """
        mm = self.map_shard(shard_id)
        
        # Calculate how many bytes to read
        num_elements = np.prod(shape)
        byte_size = num_elements * np.dtype(dtype).itemsize
        
        # Create a view of the memory without copying it (Zero-Copy)
        # This is the secret to trillion-parameter efficiency
        tensor_data = np.frombuffer(mm, dtype=dtype, count=num_elements, offset=offset)
        
        return tensor_data.reshape(shape)

    def close_all(self):
        """Clean up handles to prevent memory leaks."""
        for mm in self.mappings.values():
            mm.close()
        for f in self.file_handles.values():
            f.close()
        print("[MMAP] All virtual memory links severed safely.")