import torch
import torch.distributed as dist
from contextlib import nullcontext
from tqdm import tqdm
import wandb


def train_one_epoch(args, train_loader, model, criterion, optimizer, scaler, epoch, log_interval=10):
    model.train()

    total_loss = torch.tensor(0.0, device=args.device)
    total_samples = torch.tensor(0, device=args.device)

    for i, (imgs, labels) in enumerate(tqdm(train_loader, total=len(train_loader), desc="Training", disable=not args.master_process, leave=False)):
        imgs, labels = imgs.to(args.device), labels.to(args.device)

        is_accumulation_end = (i+1) % args.accumulation_steps == 0 or (i+1) == len(train_loader)
        sync_context = model.no_sync() if args.distributed and not is_accumulation_end else nullcontext()

        with sync_context:
            with torch.amp.autocast('cuda'):
                output = model(imgs)
                loss = criterion(output, labels)
            loss = loss / args.accumulation_steps # normalize loss to account for gradient accumulation
            scaler.scale(loss).backward()

        if is_accumulation_end:
            if args.clip_grad_norm is not None:
                scaler.unscale_(optimizer) # we should unscale the gradients of optimizer's assigned parameters if do gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.detach() * args.accumulation_steps * imgs.size(0)
        total_samples += labels.size(0)

        # send batch loss to W&B occasionally
        if args.master_process and (i+1) % log_interval == 0:
            wandb.log({"batch_train_loss": loss.detach().item() * args.accumulation_steps})

    if args.distributed:
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)

    avg_loss = total_loss.item() / total_samples.item()

    current_lr = optimizer.param_groups[0]['lr']
    if args.master_process:
        wandb.log({"epoch": epoch, "train_loss": avg_loss, "lr": current_lr})


def evaluate(args, val_loader, model, criterion, epoch, wandb_log=True):
    model.eval()

    total_loss = torch.tensor(0.0, device=args.device)
    total_correct = torch.tensor(0, device=args.device)
    total_correct_top5 = torch.tensor(0, device=args.device)
    total_samples = torch.tensor(0, device=args.device)

    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, total=len(val_loader), desc="Validation", disable=not args.master_process, leave=False):
            imgs, labels = imgs.to(args.device), labels.to(args.device)

            with torch.amp.autocast('cuda'):
                output = model(imgs)
                loss = criterion(output, labels)

            total_loss += loss * imgs.size(0)
            total_correct += (torch.argmax(output, dim=1) == labels).sum()
            total_correct_top5 += (torch.topk(output, k=5, dim=1).indices == labels.unsqueeze(1)).any(dim=1).sum()
            total_samples += labels.size(0)

    if args.distributed:
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_correct_top5, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)

    avg_loss = total_loss.item() / total_samples.item()
    avg_acc = total_correct.item() / total_samples.item()
    avg_acc_top5 = total_correct_top5.item() / total_samples.item()

    if args.master_process and wandb_log:
        wandb.log({"epoch": epoch, "val_loss": avg_loss, "val_acc": avg_acc, "val_acc_top5": avg_acc_top5})

    return avg_loss, avg_acc, avg_acc_top5
