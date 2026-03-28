"""
Script de comparaison des performances des 3 décompositions du Jeu de la Vie.

Pour chaque décomposition et chaque nombre de processus, tu entres à la main
le temps moyen par itération affiché dans le terminal.

Lancer avec : python plot_performances.py
"""

import matplotlib.pyplot as plt
import numpy as np

# ---- Nombre de processus de CALCUL testés (sans compter rank 0) ----
n_calc = [1, 2, 4]   # correspond à -n 2, -n 3, -n 5
n_proc = [2, 3, 5]   # nombre total de processus (pour les labels)

# ---- Noms des 3 décompositions ----
decompositions = {
    "1D Colonnes  (game_of_life_nprocess_column.py)": {},
    "1D Lignes    (game_of_life_nprocess_row.py)"   : {},
    "2D Boîtes    (game_of_life_nprocess_box.py)"          : {},
}

print("=" * 60)
print("SAISIE DES TEMPS MOYENS PAR ITÉRATION")
print("=" * 60)
print("Unité attendue : millisecondes (ms)")
print("(La valeur est affichée dans le terminal à chaque run)")
print()

results = {}

for nom in decompositions:
    print(f"--- {nom} ---")
    temps = []
    for nc, np_ in zip(n_calc, n_proc):
        while True:
            try:
                val = float(input(f"  mpiexec -n {np_} ... ({nc} processus de calcul) → temps moy (ms) : "))
                temps.append(val)
                break
            except ValueError:
                print("  Entrée invalide, entre un nombre.")
    results[nom] = temps
    print()

# ---- Tracé ----
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors  = ["royalblue", "tomato", "seagreen"]
markers = ["o", "s", "^"]

# Graphe 1 : temps moyen par itération
ax1 = axes[0]
for (nom, temps), color, marker in zip(results.items(), colors, markers):
    label = nom.split("(")[0].strip()
    ax1.plot(n_calc, temps, marker=marker, color=color, linewidth=2,
             markersize=8, label=label)
ax1.set_xlabel("Nombre de processus de calcul")
ax1.set_ylabel("Temps moyen par itération (ms)")
ax1.set_title("Temps de calcul")
ax1.set_xticks(n_calc)
ax1.set_xticklabels([f"{nc}\n(-n {np_})" for nc, np_ in zip(n_calc, n_proc)])
ax1.legend()
ax1.grid(True, alpha=0.3)

# Graphe 2 : speedup relatif au 1 processus de calcul
ax2 = axes[1]
for (nom, temps), color, marker in zip(results.items(), colors, markers):
    label = nom.split("(")[0].strip()
    speedup = [temps[0] / t for t in temps]
    ax2.plot(n_calc, speedup, marker=marker, color=color, linewidth=2,
             markersize=8, label=label)

# Courbe speedup idéal
ax2.plot(n_calc, n_calc, "k--", linewidth=1, label="Speedup idéal")
ax2.set_xlabel("Nombre de processus de calcul")
ax2.set_ylabel("Speedup (t1 / tn)")
ax2.set_title("Speedup")
ax2.set_xticks(n_calc)
ax2.set_xticklabels([f"{nc}\n(-n {np_})" for nc, np_ in zip(n_calc, n_proc)])
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.suptitle("Comparaison des décompositions - Jeu de la Vie MPI", fontsize=13)
plt.tight_layout()
plt.savefig("performances_comparaison.png", dpi=150)
print("✓ Graphe sauvegardé : performances_comparaison.png")
plt.show()
