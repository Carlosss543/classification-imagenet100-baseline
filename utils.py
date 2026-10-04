import os
import torch
from torch.distributed import init_process_group

def init_distributed_mode(args):
    if "RANK" in os.environ:
        assert torch.cuda.is_available(), "CUDA is not available. Please run on a machine with a compatible NVIDIA GPU."
        init_process_group(backend="nccl")
        args.ddp_rank = int(os.environ["RANK"])
        args.ddp_local_rank = int(os.environ["LOCAL_RANK"])
        args.ddp_world_size = int(os.environ["WORLD_SIZE"])
        args.distributed = True
        args.master_process = (args.ddp_rank == 0)
        args.device = torch.device(f"cuda:{args.ddp_local_rank}")
        torch.cuda.set_device(args.device)
    else:
        args.ddp_rank = 0
        args.ddp_local_rank = 0
        args.ddp_world_size = 1
        args.distributed = False
        args.master_process = True
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print(f"Device: {args.device}:{torch.cuda.current_device()} {torch.cuda.get_device_name(torch.cuda.current_device())}")
