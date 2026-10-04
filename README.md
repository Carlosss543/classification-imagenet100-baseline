Code pour l'entrainement d'un vision transformer sur le jeu de données imagenet100

sert à avoir une baseline pour tester des architectures sans devoir utiliser tout imagenet1k

Commande pour lancer le code de classification 
torchrun --nproc_per_node=2 train.py --data_dir path_to_dataset
