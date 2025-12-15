import torch
import itertools

def get_heuristic_config() -> dict:
    # Get CUDA device properties
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    device = torch.cuda.current_device()
    sm_major, sm_minor = torch.cuda.get_device_capability(device)
    sm_version = sm_major * 10 + sm_minor
    if sm_version >= 80 and sm_version < 90:
        return {
            "BLOCK_M": 128,
            "BLOCK_N": 256,
            "NUM_STAGES": 2,
            "NUM_THREADS": 128,
        }
    elif sm_version >= 90 and sm_version < 100:
        return {
            "BLOCK_M": 128,
            "BLOCK_N": 256,
            "NUM_STAGES": 3,
            "NUM_THREADS": 256,
        }
    else:
        return {
            "BLOCK_M": 128,
            "BLOCK_N": 256,
            "NUM_STAGES": 0,
            "NUM_THREADS": 128,
        }

def build_configs():
    BLOCK_M_LIST = [64, 128, 256]
    BLOCK_N_LIST = [64, 128, 256]
    NUM_STAGES_LIST = [0, 1, 2, 3]
    NUM_THREADS_LIST = [128, 256]
    CONFIGS = list(
        itertools.product(
            BLOCK_M_LIST,
            BLOCK_N_LIST,
            NUM_STAGES_LIST,
            NUM_THREADS_LIST,
        )
    )

    return [
        {
            "BLOCK_M": c[0],
            "BLOCK_N": c[1],
            "NUM_STAGES": c[2],
            "NUM_THREADS": c[3],
        } for c in CONFIGS
    ]