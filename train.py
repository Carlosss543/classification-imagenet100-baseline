import torch
from data import get_data_loaders
from models import vit_t_16, vit_s_16
import os
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
from utils import init_distributed_mode
from torch.distributed import destroy_process_group
import wandb
from tqdm import tqdm
from engine import train_one_epoch, evaluate


def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--num_classes", type=int, default=100)

    parser.add_argument("--resize_size", type=int, default=256)
    parser.add_argument("--crop_size", type=int, default=224)

    parser.add_argument("--num_epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--accumulation_steps", type=int, default=4)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_warmup_epochs", type=int, default=5)
    parser.add_argument("--lr_warmup_decay", type=float, default=0.033)

    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--mixup_alpha", type=float, default=0.2)
    parser.add_argument("--cutmix_alpha", type=float, default=1.0)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)

    parser.add_argument("--model", type=str, default="vit_s_16")

    parser.add_argument("--checkpoints_interval", type=int, default=50)
    parser.add_argument("--folder_number", type=int, default=1)

    parser.add_argument("--resume_from_checkpoint", action="store_true")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--wandb_run_id", type=str, default=None)

    return parser


def main(args):
    # --- distributed training setup ---
    init_distributed_mode(args)


    # --- load data ---
    batch_size_per_gpu = int(args.batch_size / args.ddp_world_size)
    train_loader, val_loader, train_sampler, _ = get_data_loaders(args, batch_size_per_gpu, distributed=args.distributed, rank=args.ddp_rank, world_size=args.ddp_world_size)


    # --- load model ---
    if args.model == "vit_t_16":
        model = vit_t_16(num_classes=args.num_classes, image_size=args.crop_size).to(args.device)
    elif args.model == "vit_s_16":
        model = vit_s_16(num_classes=args.num_classes, image_size=args.crop_size).to(args.device)
    else:
        raise ValueError(f"Unknown model type '{args.model}'")

    start_epoch = 1

    if args.resume_from_checkpoint:
        assert os.path.exists(args.checkpoint_path), f"Checkpoint path '{args.checkpoint_path}' does not exist."
        if args.master_process:
            print(f"Loading checkpoint '{args.checkpoint_path}'")
        checkpoint = torch.load(args.checkpoint_path, map_location=args.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint["epoch"] + 1

    #model = torch.compile(model)

    if args.distributed:
        model = DDP(model, device_ids=[args.ddp_local_rank])

    if args.master_process:
        print(f"Number of model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.4f}M")


    # --- optimizer, criterion, gradient scaler for fp16 ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, fused=True)

    criterion_train = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    criterion_val = torch.nn.CrossEntropyLoss()

    scaler = torch.amp.GradScaler('cuda')


    # --- learning rate scheduler with warmup and cosine decay ---
    warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs)
    main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs - args.lr_warmup_epochs)
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epochs])


    # --- resume optimizer, lr scheduler, and scaler states if resuming from checkpoint ---
    if args.resume_from_checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        del checkpoint # free up memory


    # --- initialize wandb ---
    config = {k: v for k, v in vars(args).items() if not k.startswith("_")}
    wandb.init(
        project="classification-imagenet100-baseline",
        dir="./wandb_logs",
        config=config,
        group=f"{args.model}",
        mode="online" if args.master_process else "disabled",  # online/disabled
        id=args.wandb_run_id if args.resume_from_checkpoint else None,
        resume="must" if args.resume_from_checkpoint else False
    )


    # --- training loop ---
    for epoch in tqdm(range(start_epoch, args.num_epochs + 1), desc="Epochs", disable=not args.master_process):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_one_epoch(args, train_loader, model, criterion_train, optimizer, args.device, scaler, distributed=args.distributed, master_process=args.master_process)
        lr_scheduler.step()
        _ = evaluate(val_loader, model, criterion_val, args.device, distributed=args.distributed, master_process=args.master_process, wandb_log=True)

        if args.master_process and args.checkpoints_interval is not None and epoch % args.checkpoints_interval == 0:
            model_to_save = model.module if hasattr(model, "module") else model # Unwrap the model if it's wrapped by DDP
            model_to_save = model_to_save._orig_mod if hasattr(model_to_save, "_orig_mod") else model_to_save # Unwrap the model if it's wrapped by torch.compile
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model_to_save.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "args": config
            }
            checkpoint_dir = f"./training_checkpoints/checkpoint{args.folder_number}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(checkpoint, f"{checkpoint_dir}/vit_custom_epoch_{epoch}.pth")

    if args.distributed:
        destroy_process_group()


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
