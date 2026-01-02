# ai_backend/infrastructure/synapse/parallelism.py
import torch
import pynvml # pip install nvidia-ml-py

class ParallelismEngine:
    def __init__(self, model_type="nano", parameter_count=248_000_000_000_000):
        self.model_type = model_type
        self.params = parameter_count
        self.gpu_count = torch.cuda.device_count()
        
    def check_environment(self):
        """
        Decides the strategy based on model size and hardware.
        """
        # 1. NANO LOGIC: Don't overcomplicate small things
        if self.model_type == "nano":
            print("[AUTO] Environment: NANO. Parallelism disabled (Efficiency Mode).")
            return "SINGLE_CORE"

        # 2. ULTRA LOGIC: Check for the 1-GPU Danger Zone
        if self.model_type == "ultra" or self.params > 100_000_000_000:
            if self.gpu_count == 1:
                self._warn_user_of_doom()
                return "OFFLOAD_TO_DISK" # Use mmap heavily to prevent OOM
            else:
                print(f"[AUTO] Environment: BIG. Activating Tensor Parallelism across {self.gpu_count} GPUs.")
                return "TENSOR_PARALLEL"

    def _warn_user_of_doom(self):
        """The warning message for high-risk configurations."""
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        vram_gb = info.total / 1024**3

        print("!" * 50)
        print("CRITICAL HARDWARE WARNING")
        print(f"You are attempting to run a {self.params/1e12}T parameter model.")
        print(f"Hardware detected: 1 GPU with {vram_gb:.2f}GB VRAM.")
        print("DANGER: This will likely crash your OS or hang your GPU.")
        print("ADVICE: Use 4-bit quantization and enable 'Aggressive Disk Offloading'.")
        print("!" * 50)

    def split_tensor(self, tensor):
        """Only splits if the environment is 'BIG'."""
        strategy = self.check_environment()
        if strategy == "TENSOR_PARALLEL":
            # Logic to split matrix across multiple GPUs
            return torch.chunk(tensor, self.gpu_count, dim=-1)
        return tensor # Return as-is for Nano or Single-GPU