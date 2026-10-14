# Vision Transformer Baseline on ImageNet100

This project provides a Vision Transformer baseline on ImageNet-100 for evaluating and comparing different architectures at a lower cost than training on ImageNet-1K. The code is designed to be relatively easy to read and modify while taking advantage of the hardware acceleration features provided by PyTorch (float16, kernel fusion, DistributedDataParallel).

## Data

The directory passed to `--data_dir` must contain the `train` and `val` directories. Each split must be organized into class subdirectories, as expected by `torchvision.datasets.ImageFolder`:

```text
imagenet100/
|-- train/
|   |-- class_001/
|   |   `-- image.jpg
|   `-- class_002/
|       `-- image.jpg
`-- val/
    |-- class_001/
    |   `-- image.jpg
    `-- class_002/
        `-- image.jpg
```

The class directory names must match between `train` and `val`. With the default configuration, the dataset must contain 100 classes.

## Getting Started

From the repository root, for example, start training on two GPUs with:

```bash
torchrun --nproc_per_node=2 train.py --data_dir /path/to/imagenet100
```

For a single process/GPU:

```bash
python train.py --data_dir /path/to/imagenet100
```

The default `--batch_size` (`256`) is the number of images per forward pass across all GPUs: it is split among the processes (with two GPUs, each therefore processes 128 images). `--accumulation_steps` (default `4`) accumulates gradients across multiple forward passes before an optimizer update. The effective batch size, i.e. the number of images processed per update, is `batch_size × accumulation_steps`, or 256 × 4 = 1024 by default, regardless of the number of GPUs.

Experiments are logged to the W&B project `classification-imagenet100-baseline`, and local W&B files are stored in `./wandb_logs`.

## Main Options

| Option | Default | Description |
| --- | --- | --- |
| `--model` | `vit_s_16` | Architecture: `vit_s_16` or `vit_t_16`. |
| `--num_classes` | `100` | Number of classes in the dataset. |
| `--num_epochs` | `300` | Total number of epochs. |
| `--batch_size` | `256` | Number of images per forward pass across all GPUs. |
| `--accumulation_steps` | `4` | Number of forward passes accumulated per update. |
| `--lr` | `0.001` | Learning rate. |
| `--weight_decay` | `0.05` | AdamW weight decay. |
| `--crop_size` | `224` | Square input image size for the model. |
| `--resize_size` | `256` | Resize dimension before the validation center crop. |
| `--mixup_alpha` | `0.2` | MixUp alpha parameter; set to zero to disable MixUp. |
| `--cutmix_alpha` | `1.0` | CutMix alpha parameter; set to zero to disable CutMix. |
| `--checkpoints_interval` | `50` | Save a checkpoint every N epochs. |
| `--folder_number` | `1` | Checkpoint subdirectory number. |

The script applies a learning-rate warmup followed by cosine decay. Training images are augmented randomly; validation uses resizing and a center crop.

## Results

Training with the default parameters produces the following accuracies.

| Model      | Top-1 Accuracy | Top-5 Accuracy |
| ---------- | -------------- | -------------- |
| `vit_s_16` | 79.8%          | 92.5%          |
| `vit_t_16` | 78.8%          | 92.7%          |

## Checkpoints and Resuming

Checkpoints are saved every `--checkpoints_interval` epochs in `training_checkpoints/checkpoint<folder_number>/`. To resume training, provide the checkpoint path, enable `--resume_from_checkpoint`, and specify the ID of the W&B run to resume:

```bash
torchrun --nproc_per_node=2 train.py \
	--data_dir /path/to/imagenet100 \
	--resume_from_checkpoint \
	--checkpoint_path training_checkpoints/checkpoint1/vit_custom_epoch_50.pth \
	--wandb_run_id RUN_ID
```

Resuming restores the model weights and the states of the optimizer, scheduler, and scaler, then continues from the epoch after the checkpoint. W&B must be able to resume the corresponding existing run.
