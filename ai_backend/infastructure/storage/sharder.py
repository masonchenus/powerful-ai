# ai_backend/infrastructure/storage/sharder.py
import json
import os
import numpy as np

class Sharder:
    def __init__(self, output_dir="weights/binary_shards/", shard_size_mb=500):
        self.output_dir = output_dir
        self.shard_size_bytes = shard_size_mb * 1024 * 1024
        self.manifest = {"model_size": "248T", "shards": {}}
        os.makedirs(output_dir, exist_ok=True)

    def shard_tensor(self, tensor_name, tensor_data):
        """
        Takes a massive matrix and breaks it into binary chunks.
        """
        # Convert to bytes
        raw_data = tensor_data.tobytes()
        total_bytes = len(raw_data)
        
        shard_count = (total_bytes // self.shard_size_bytes) + 1
        
        for i in range(shard_count):
            shard_id = f"{tensor_name}_s{i:04d}"
            start = i * self.shard_size_bytes
            end = min(start + self.shard_size_bytes, total_bytes)
            
            # Save the binary chunk
            shard_path = os.path.join(self.output_dir, f"{shard_id}.bin")
            with open(shard_path, "wb") as f:
                f.write(raw_data[start:end])
            
            # Record in manifest for the MMap Manager
            self.manifest["shards"][shard_id] = {
                "parent_tensor": tensor_name,
                "byte_range": [start, end],
                "file": shard_path
            }

    def save_manifest(self):
        with open("weights/binary_shards/manifest.json", "w") as f:
            json.dump(self.manifest, f, indent=4)