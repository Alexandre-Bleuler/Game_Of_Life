"""
Script de comparaison des performances des 3 décompositions du Jeu de la Vie.

Pour chaque décomposition et chaque nombre de processus, tu entres à la main
le temps moyen par itération affiché dans le terminal.

Lancer avec : python plot_performances.py
"""

import matplotlib.pyplot as plt
import numpy as np

# Création des arrays de données 

nb_process=np.array([2,3,4,5,6,7])
row_time=np.array([0.0678,0.0352,0.0250,0.0211,0.0186,0.0198])
column_time=np.array([0.0680,0.0349,0.0247,0.0196,0.0179,0.0209])
box_time=np.array([0.0672,0.0392,0.0299,0.0196,0.0189,0.0189])

# Affichage 

plt.plot(nb_process, row_time, color="r")
plt.plot(nb_process, column_time, color="g")
plt.plot(nb_process, box_time, color="b")

plt.legend(["row", "column","box"])
plt.xlabel("Nombre total de processus")
plt.ylabel("Temps d'itération moyen sur 500 itérations")
plt.show()




