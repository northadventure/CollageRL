import torch
import nvidia_smi
import numpy as np 

def get_free_device():
    # Auto-GPU assignment: use a GPU device with lowest-memory usage
    nvidia_smi.nvmlInit()
    deviceCount = nvidia_smi.nvmlDeviceGetCount()
    freem = []
    for i in range(deviceCount):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        freem.append(100*info.free/info.total)
    nvidia_smi.nvmlShutdown()
    device_num = np.argmax(freem).item()
    torch.cuda.set_device(device_num)
    device = torch.device(f"cuda:{device_num}")

    return device

device = get_free_device()  # Use auto-GPU assignment
# device = 'cuda:0'  # ...or set manually