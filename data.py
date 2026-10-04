from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data.dataloader import default_collate
from mixup_cutmix import get_mixup_cutmix
import torch
import matplotlib.pyplot as plt


def get_data_loaders(args, batch_size, shuffle_val=False, distributed=False, rank=0, world_size=1):

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # mean and std for ImageNet

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(args.crop_size, interpolation=InterpolationMode.BILINEAR, antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(interpolation=InterpolationMode.BILINEAR, num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    val_transform = transforms.Compose([
        transforms.Resize(args.resize_size, interpolation=InterpolationMode.BILINEAR, antialias=True),
        transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    train_dataset = datasets.ImageFolder(f"{args.data_dir}/train", transform=train_transform)
    val_dataset = datasets.ImageFolder(f"{args.data_dir}/val", transform=val_transform)

    # add cutmix and mixup to the training dataset
    mixup_cutmix = get_mixup_cutmix(mixup_alpha=args.mixup_alpha, cutmix_alpha=args.cutmix_alpha, num_classes=args.num_classes, use_v2=False)
    if mixup_cutmix is not None:
        def collate_fn(batch):
            return mixup_cutmix(*default_collate(batch))
    else:
        collate_fn = default_collate

    train_sampler = None
    val_sampler = None
    if distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=shuffle_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=shuffle_val if val_sampler is None else False,
        sampler=val_sampler,
        num_workers=2,
        pin_memory=True,
        persistent_workers=False,
    )

    # # display some image samples in results/augmented_images.png
    # sample_imgs, _ = next(iter(train_loader))
    # sample_imgs = sample_imgs[:16]  # take first 16 images
    # sample_imgs = sample_imgs.cpu().permute(0, 2, 3, 1) * torch.tensor(std).view(1, 1, 1, 3) + torch.tensor(mean).view(1, 1, 1, 3)
    # sample_imgs = torch.clamp(sample_imgs, 0, 1).numpy()
    # fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    # for i, ax in enumerate(axes.flat):
    #     ax.imshow(sample_imgs[i])
    #     ax.axis('off')
    # plt.savefig("augmented_images.png")
    # plt.close()

    return train_loader, val_loader, train_sampler, val_sampler
