# Projet Calcul parallèle : le jeu de la vie

## Introduction 

Le but de ce projet a été de paralléliser le jeu de la vie sur une grille torique en utilisant 
une décomposition de domaine. Plusieurs types de découpages ont été explorés et des comparaisons
de performances ont été réalisées.

L'ensemble des fichiers ont été codés à l'aide d'un langage `Python` et de la librairie de calcul parallèle `mpi4py`.

## Présentation des fichiers

La parallélisation a été réalisée dans l'esprit suivant :
- séparation des processus en un groupe d'affichage (processus de rank global 0) et un groupe de calcul (les autres processeurs);
- 

Le projet est scindé en 4 fichiers :
- `game_of_life_2process` : 
- `game_of_life_nprocess_line` : 
- `game_of_life_nprocess_column` : 
- `game_of_life_nprocess_box` : 


## Performances 
