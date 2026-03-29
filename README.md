# Projet Calcul parallèle : le jeu de la vie

## Introduction

Le but de ce projet a été de paralléliser le jeu de la vie sur une grille torique en utilisant
une décomposition de domaine. Plusieurs types de découpages ont été explorés et des comparaisons
de performances ont été réalisées.

L'ensemble des fichiers ont été codés en `Python` avec la librairie de calcul parallèle `mpi4py`.

## Présentation des fichiers

La parallélisation repose sur le principe suivant :
- séparation des processus en un **processus d'affichage** (rank global 0) et des **processus de calcul** (les autres) ;
- chaque processus de calcul possède une sous-grille locale avec des **cellules fantômes** (ghost cells) sur ses bords, mises à jour par échange avec les voisins à chaque itération ;
- les résultats sont envoyés au processus 0 via `gather` pour mise à jour et affichage.

Le projet est scindé en 4 fichiers :

- `game_of_life_2process.py` : version de référence à exactement 2 processus. Un processus calcule l'intégralité de la grille, l'autre l'affiche. Pas de découpage de domaine.
- `game_of_life_nprocess_column.py` : parallélisation 1D par **colonnes**. La grille est découpée en bandes verticales, chaque processus de calcul gère une bande. Les échanges de ghost cells se font uniquement entre voisins gauche/droite.
- `game_of_life_nprocess_line.py` : parallélisation 1D par **lignes**. La grille est découpée en bandes horizontales. Les échanges de ghost cells se font uniquement entre voisins haut/bas.
- `game_of_life_nprocess_box.py` : parallélisation **2D par boîtes**. La grille est découpée en sous-grilles rectangulaires. Chaque processus échange des ghost cells avec ses 8 voisins (haut, bas, gauche, droite et les 4 diagonales).

## Lancement
```bash
mpiexec -n 2 python game_of_life_2process.py [pattern]          # exactement N=2
mpiexec -n N python game_of_life_nprocess_column.py [pattern]   # N >= 2
mpiexec -n N python game_of_life_nprocess_line.py [pattern]      # N >= 2
mpiexec -n N python game_of_life_nprocess_box.py [pattern]      # N >= 2
```

Patterns disponibles : `glider`, `glider_gun`, `acorn`, `pulsar`, `floraison`, `blinker`, `toad`, `beacon`, `boat`, `space_ship`, `die_hard`, `block_switch_engine`, `u`, `flat`.

Exemple :
```bash
mpiexec -n 5 python game_of_life_nprocess_box.py glider_gun
```

## Benchmarks et courbes de performances

Deux fichiers permettent de mesurer et visualiser les performances :
- `bench.py` : script de benchmark sans affichage pygame, à lancer avec `mpiexec`
- `run_bench.sh` : script bash qui automatise tous les runs et génère les graphes

### Sur Linux / Mac
```bash
bash run_bench.sh glider 200
```
Cela lance automatiquement les 9 runs (3 décompositions × 3 nombres de processus) et génère `performances_comparaison.png`.

### Sur Windows
Le script bash ne fonctionne pas nativement. Deux options :

**Option 1 — Git Bash ou WSL :**
```bash
bash run_bench.sh glider 200
```

**Option 2 — Lancer les benchmarks à la main un par un :**
```bash
mpiexec -n 2 python bench.py --decomp column --pattern glider --iters 200
mpiexec -n 3 python bench.py --decomp column --pattern glider --iters 200
mpiexec -n 5 python bench.py --decomp column --pattern glider --iters 200

mpiexec -n 2 python bench.py --decomp row --pattern glider --iters 200
mpiexec -n 3 python bench.py --decomp row --pattern glider --iters 200
mpiexec -n 5 python bench.py --decomp row --pattern glider --iters 200

mpiexec -n 2 python bench.py --decomp 2d --pattern glider --iters 200
mpiexec -n 3 python bench.py --decomp 2d --pattern glider --iters 200
mpiexec -n 5 python bench.py --decomp 2d --pattern glider --iters 200
```
Puis noter les temps affichés et lancer :
```bash
python plot_performances.py
```
Le script vous demandera d'entrer les valeurs à la main et générera `performances_comparaison.png`.

## Performances

Les performances ont été mesurées en termes de **temps moyen par itération** (ms) en fonction du nombre de processus de calcul, pour chaque décomposition, sur le pattern `glider` (grille 100×90), sur Windows avec Microsoft MPI.

| Processus de calcul | 1D Colonnes (ms) | 1D Lignes (ms) | 2D Boîtes (ms) |
|:-------------------:|:----------------:|:--------------:|:--------------:|
| 1                   | 0.349            | 0.375          | 0.501          |
| 2                   | 1.020            | 0.936          | 1.084          |
| 4                   | 15.756           | 13.949         | 13.045         |

### Analyse

Les résultats montrent que le temps de calcul **augmente** avec le nombre de processus, ce qui est contre-intuitif mais s'explique par plusieurs facteurs :

**Domination de la latence de communication.** Sur une machine Windows locale, chaque opération `Send/Recv` a une latence fixe élevée (de l'ordre de quelques ms). Avec une grille 100×90 découpée en 4 sous-grilles, chaque sous-grille ne contient qu'environ 25×22 cellules — le calcul prend moins de 0.5 ms — mais les échanges de ghost cells imposent un overhead bien supérieur. On observe que `comm_ms` représente plus de 95% du temps total à 4 processus de calcul.

**Avantage relatif de la décomposition 2D.** Malgré cet effet, la décomposition 2D reste la plus rapide à 4 processus (13.0 ms contre 15.8 ms pour les colonnes). En effet, les sous-grilles 2D sont plus carrées, ce qui minimise le périmètre total des ghost cells échangées. Cet avantage est prédit par la complexité théorique : le volume de communication croît en $O(\sqrt{n})$ pour la décomposition 2D, contre $O(n)$ pour les décompositions 1D.

