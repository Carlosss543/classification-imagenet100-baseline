# Baseline Vision Transformer sur ImageNet100

Ce projet fournit une baseline Vision Transformer sur ImageNet-100 pour évaluer et comparer différentes architectures à moindre coût par rapport à un entraînement sur ImageNet-1K. Le code a été conçu pour être relativement simple et facile à modifier tout en utilisant les diverses accélérations matérielles offertes par PyTorch (float16, fusion de kernels, DistributedDataParallel).

## Données

Le dossier fourni par `--data_dir` doit contenir les répertoires `train` et `val`. Chaque split doit être organisé en sous-dossiers par classe, conformément à `torchvision.datasets.ImageFolder` :

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

Les noms des dossiers de classes doivent correspondre entre `train` et `val`. Avec la configuration par défaut, le jeu de données doit contenir 100 classes.

## Lancement

Depuis la racine du dépôt, lancez par exemple l'entraînement sur deux GPU :

```bash
torchrun --nproc_per_node=2 train.py --data_dir /chemin/vers/imagenet100
```

Pour un seul processus/GPU :

```bash
python train.py --data_dir /chemin/vers/imagenet100
```

Le `--batch_size` par défaut (`256`) est le nombre d'images par forward pass, tous GPU confondus : il est réparti entre les processus (avec deux GPU, chacun traite donc 128 images). `--accumulation_steps` (par défaut `4`) accumule les gradients de plusieurs forward passes avant une mise à jour de l'optimiseur. Le batch effectif, c'est-à-dire le nombre d'images vues par mise à jour, vaut `batch_size × accumulation_steps`, soit 256 × 4 = 1024 par défaut, quel que soit le nombre de GPU.

Les expériences sont envoyées au projet W&B `classification-imagenet100-baseline` et les fichiers locaux de W&B sont enregistrés dans `./wandb_logs`.

## Options principales

| Option | Défaut | Description |
| --- | --- | --- |
| `--model` | `vit_s_16` | Architecture : `vit_s_16` ou `vit_t_16`. |
| `--num_classes` | `100` | Nombre de classes dans le dataset. |
| `--num_epochs` | `300` | Nombre total d'epochs. |
| `--batch_size` | `256` | Nombre d'images par forward pass, tous GPU confondus. |
| `--accumulation_steps` | `4` | Nombre de forward passes accumulés par mise à jour. |
| `--lr` | `0.001` | Learning rate. |
| `--weight_decay` | `0.05` | Weight decay d'AdamW. |
| `--crop_size` | `224` | Taille carrée des images en entrée du modèle. |
| `--resize_size` | `256` | Taille de redimensionnement avant le recadrage de validation. |
| `--mixup_alpha` | `0.2` | Paramètre alpha de MixUp; une valeur nulle désactive MixUp. |
| `--cutmix_alpha` | `1.0` | Paramètre alpha de CutMix; une valeur nulle désactive CutMix. |
| `--checkpoints_interval` | `50` | Sauvegarde un checkpoint toutes les N epochs. |
| `--folder_number` | `1` | Numéro du sous-dossier de checkpoints. |

Le script applique un warmup du learning rate, puis une décroissance cosinus. Les images d'entraînement sont augmentées aléatoirement; la validation utilise redimensionnement et recadrage central.

## Résultats

En entraînant avec les paramètres par défaut, on obtient les accuracies suivantes.

| Modèle     | Accuracy Top-1 | Accuracy Top-5 |
| ---------- | -------------- | -------------- |
| `vit_s_16` | 79.8%          | 92.5%          |
| `vit_t_16` | 78.8%          | 92.7%          |

## Checkpoints et reprise

Les checkpoints sont écrits toutes les `--checkpoints_interval` epochs dans `training_checkpoints/checkpoint<folder_number>/`. Pour reprendre, fournissez le chemin du checkpoint, activez `--resume_from_checkpoint` et donnez l'identifiant de la run W&B à reprendre :

```bash
torchrun --nproc_per_node=2 train.py \
	--data_dir /chemin/vers/imagenet100 \
	--resume_from_checkpoint \
	--checkpoint_path training_checkpoints/checkpoint1/vit_custom_epoch_50.pth \
	--wandb_run_id ID_DE_LA_RUN
```

La reprise restaure les poids du modèle, l'état de l'optimizer, du scheduler et du scaler, puis continue à l'epoch suivant celle du checkpoint. W&B doit pouvoir reprendre la run existante correspondante.
