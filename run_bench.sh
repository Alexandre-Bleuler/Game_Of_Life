#!/bin/bash
# run_bench.sh - Lance tous les benchmarks et génère la courbe de performances
#
# Usage : bash run_bench.sh [pattern] [iters]
# Exemple : bash run_bench.sh glider 200

PATTERN=${1:-glider}
ITERS=${2:-200}
RESULT_FILE="bench_results.txt"
DECOMPS=("column" "row" "2d")
N_PROCS=(2 3 5)   # -n 2 = 1 calcul, -n 3 = 2 calcul, -n 5 = 4 calcul

echo "=============================================="
echo " BENCHMARK JEU DE LA VIE MPI"
echo " Pattern : $PATTERN | Itérations : $ITERS"
echo "=============================================="

# Nettoyer les anciens résultats
> "$RESULT_FILE"

for decomp in "${DECOMPS[@]}"; do
    echo ""
    echo "--- Décomposition : $decomp ---"
    for n in "${N_PROCS[@]}"; do
        echo -n "  mpiexec -n $n ... "
        # Lance le benchmark et capture uniquement la ligne BENCH_RESULT
        result=$(mpiexec -n "$n" python bench.py \
                    --decomp "$decomp" \
                    --pattern "$PATTERN" \
                    --iters "$ITERS" 2>/dev/null \
                 | grep "^BENCH_RESULT")
        if [ -z "$result" ]; then
            echo "ERREUR (aucun résultat reçu)"
        else
            echo "$result" | tee -a "$RESULT_FILE"
        fi
    done
done

echo ""
echo "=============================================="
echo " Résultats sauvegardés dans : $RESULT_FILE"
echo " Génération du graphe..."
echo "=============================================="

python - << 'PYEOF'
import matplotlib.pyplot as plt
import numpy as np
import re

results = {}   # {decomp: {nproc: {calc, comm, total}}}

with open("bench_results.txt") as f:
    for line in f:
        line = line.strip()
        if not line.startswith("BENCH_RESULT"):
            continue
        d      = re.search(r"decomp=(\w+)",     line).group(1)
        n      = int(re.search(r"nproc=(\d+)",  line).group(1))
        calc   = float(re.search(r"calc_ms=([\d.]+)",  line).group(1))
        comm   = float(re.search(r"comm_ms=([\d.]+)",  line).group(1))
        total  = float(re.search(r"total_ms=([\d.]+)", line).group(1))
        results.setdefault(d, {})[n] = {"calc": calc, "comm": comm, "total": total}

if not results:
    print("Aucun résultat à tracer.")
    exit(1)

labels = {"column": "1D Colonnes", "row": "1D Lignes", "2d": "2D Boîtes"}
colors  = {"column": "royalblue",  "row": "tomato",    "2d": "seagreen"}
markers = {"column": "o",          "row": "s",          "2d": "^"}

n_calc_labels = {2: "1", 3: "2", 5: "4"}  # nproc → nb processus de calcul

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for decomp, data in results.items():
    nprocs = sorted(data.keys())
    n_calcs = [int(n_calc_labels[n]) for n in nprocs]
    totals  = [data[n]["total"] for n in nprocs]
    calcs   = [data[n]["calc"]  for n in nprocs]
    comms   = [data[n]["comm"]  for n in nprocs]
    speedup = [totals[0] / t for t in totals]

    kw = dict(color=colors[decomp], marker=markers[decomp], linewidth=2, markersize=8, label=labels[decomp])

    axes[0].plot(n_calcs, totals,  **kw)
    axes[1].plot(n_calcs, speedup, **kw)

    # Graphe calcul vs comm en barres groupées
    x = np.arange(len(n_calcs))

# Graphe 1 : temps total
axes[0].set_title("Temps moyen par itération")
axes[0].set_xlabel("Processus de calcul")
axes[0].set_ylabel("Temps (ms)")
axes[0].set_xticks(list(map(int, n_calc_labels.values())))
axes[0].legend(); axes[0].grid(True, alpha=0.3)

# Graphe 2 : speedup
all_nc = sorted(set(int(v) for v in n_calc_labels.values()))
axes[1].plot(all_nc, all_nc, "k--", linewidth=1, label="Idéal")
axes[1].set_title("Speedup")
axes[1].set_xlabel("Processus de calcul")
axes[1].set_ylabel("Speedup (t1/tn)")
axes[1].set_xticks(all_nc)
axes[1].legend(); axes[1].grid(True, alpha=0.3)

# Graphe 3 : calcul vs comm pour n=5 (le plus intéressant)
ax3 = axes[2]
decomp_list = [d for d in ["column","row","2d"] if d in results and 5 in results[d]]
x = np.arange(len(decomp_list))
width = 0.35
calcs5 = [results[d][5]["calc"]  for d in decomp_list]
comms5 = [results[d][5]["comm"]  for d in decomp_list]
ax3.bar(x - width/2, calcs5, width, label="Calcul",  color="steelblue")
ax3.bar(x + width/2, comms5, width, label="Comm",    color="coral")
ax3.set_xticks(x)
ax3.set_xticklabels([labels[d] for d in decomp_list])
ax3.set_title("Calcul vs Communication (n=5)")
ax3.set_ylabel("Temps (ms)")
ax3.legend(); ax3.grid(True, alpha=0.3, axis="y")

plt.suptitle(f"Comparaison décompositions MPI - pattern={open('bench_results.txt').readline()}", fontsize=11)
plt.tight_layout()
plt.savefig("performances_comparaison.png", dpi=150)
print("✓ Graphe sauvegardé : performances_comparaison.png")
plt.show()
PYEOF
