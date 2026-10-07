# Baseline Vision Transformer sur ImageNet100

Ce projet entraîne un Vision Transformer sur ImageNet100 afin de fournir une baseline solide pour tester des architectures sans devoir utiliser l'ensemble d'ImageNet-1K.

## Données

Le dossier donné avec `--data_dir` doit contenir les répertoires `train` et `val`. Chaque split doit être organisé en sous-dossiers par classe, conformément à `torchvision.datasets.ImageFolder` :

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

Le `--batch_size` par défaut (`256`) est la taille globale du lot et est réparti entre les processus. Avec deux GPU, chaque processus traite donc 128 images par lot. `--accumulation_steps` (par défaut `4`) accumule plusieurs lots avant une mise à jour de l'optimiseur. Par exemple, pour un batch_size de 256 et 4 accumulation steps le batch effectif est de 1024.

Les expériences sont envoyées au projet W&B `classification-imagenet100-baseline` et les fichiers locaux de W&B sont enregistrés dans `./wandb_logs`.

## Options principales

| Option | Défaut | Description |
| --- | --- | --- |
| `--model` | `vit_s_16` | Architecture : `vit_s_16` ou `vit_t_16`. |
| `--num_classes` | `100` | Nombre de classes dans le dataset. |
| `--num_epochs` | `300` | Nombre total d'epochs. |
| `--batch_size` | `256` | Taille globale du lot, avant accumulation. |
| `--accumulation_steps` | `4` | Nombre de lots accumulés par mise à jour. |
| `--lr` | `0.001` | Taux d'apprentissage. |
| `--weight_decay` | `0.05` | Weight decay d'AdamW. |
| `--crop_size` | `224` | Taille carrée des images en entrée du modèle. |
| `--resize_size` | `256` | Taille de redimensionnement avant le recadrage de validation. |
| `--mixup_alpha` | `0.2` | Paramètre alpha de MixUp; une valeur nulle désactive MixUp. |
| `--cutmix_alpha` | `1.0` | Paramètre alpha de CutMix; une valeur nulle désactive CutMix. |
| `--checkpoints_interval` | `50` | Sauvegarde un checkpoint toutes les N époques. |
| `--folder_number` | `1` | Numéro du sous-dossier de checkpoints. |

Le script applique un warmup du taux d'apprentissage, puis une décroissance cosinus. Les images d'entraînement sont augmentées aléatoirement; la validation utilise redimensionnement et recadrage central.

## Checkpoints et reprise

Les checkpoints sont écrits toutes les `--checkpoints_interval` epochs dans `training_checkpoints/checkpoint<folder_number>/`. Pour reprendre, fournissez le chemin du checkpoint, activez `--resume_from_checkpoint` et donnez l'identifiant de la run W&B à reprendre :

```bash
torchrun --nproc_per_node=2 train.py \
	--data_dir /chemin/vers/imagenet100 \
	--resume_from_checkpoint \
	--checkpoint_path training_checkpoints/checkpoint1/vit_custom_epoch_50.pth \
	--wandb_run_id ID_DE_LA_RUN
```

La reprise restaure les poids du modèle, l'état de l'optimiseur, du scheduler et du scaler, puis continue à l'époque suivant celle du checkpoint. W&B doit pouvoir reprendre la run existante correspondante.
