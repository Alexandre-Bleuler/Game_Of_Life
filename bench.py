"""
bench.py - Script de benchmark sans affichage pygame.
Lance avec :
    mpiexec -n N python bench.py --decomp [column|row|2d] --pattern glider --iters 200
Affiche en fin de run :
    BENCH_RESULT decomp=column nproc=3 calc_ms=11.23 comm_ms=2.45 total_ms=13.68
"""

import numpy as np
import sys
import time
import argparse
from mpi4py import MPI


# ─────────────────────────────────────────────
# Parse args (tous les processus les lisent)
# ─────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--decomp",  choices=["column","row","2d"], required=True)
parser.add_argument("--pattern", default="glider")
parser.add_argument("--iters",   type=int, default=200)
args = parser.parse_args()

globCom = MPI.COMM_WORLD.Dup()
nbp  = globCom.size
rank = globCom.rank

if nbp < 2:
    if rank == 0: print("Il faut au moins 2 processus.")
    sys.exit(1)

n_calc    = nbp - 1
calc_rank = rank - 1

# ─────────────────────────────────────────────
# Patterns
# ─────────────────────────────────────────────
dico_patterns = {
    'blinker' : ((5,5),   [(2,1),(2,2),(2,3)]),
    'toad'    : ((6,6),   [(2,2),(2,3),(2,4),(3,3),(3,4),(3,5)]),
    'glider'  : ((100,90),[(1,1),(2,2),(2,3),(3,1),(3,2)]),
    'acorn'   : ((100,100),[(51,52),(52,54),(53,51),(53,52),(53,55),(53,56),(53,57)]),
    'glider_gun': ((400,400),[(51,76),(52,74),(52,76),(53,64),(53,65),(53,72),(53,73),(53,86),(53,87),(54,63),(54,67),(54,72),(54,73),(54,86),(54,87),(55,52),(55,53),(55,62),(55,68),(55,72),(55,73),(56,52),(56,53),(56,62),(56,66),(56,68),(56,69),(56,74),(56,76),(57,62),(57,68),(57,76),(58,63),(58,67),(59,64),(59,65)]),
}

if args.pattern not in dico_patterns:
    if rank == 0: print(f"Pattern inconnu: {args.pattern}")
    sys.exit(1)

dim, pattern = dico_patterns[args.pattern]

# ─────────────────────────────────────────────
# Broadcast pattern
# ─────────────────────────────────────────────
if rank == 0:
    flat = np.array([v for pt in pattern for v in pt], dtype=np.int32)
else:
    flat = None
dim  = globCom.bcast(dim,  root=0)
flat = globCom.bcast(flat, root=0)
pattern = [(flat[k], flat[k+1]) for k in range(0, len(flat), 2)]

# ─────────────────────────────────────────────
# Grilles locales selon la décomposition
# ─────────────────────────────────────────────
if rank != 0:
    com_calc  = globCom.Split(color=1, key=calc_rank)
    nbp_calc  = com_calc.size
    rank_calc = com_calc.rank

    # ── COLUMN ──────────────────────────────
    if args.decomp == "column":
        col_dim = dim[1]
        reste   = col_dim % nbp_calc
        nx_loc  = col_dim // nbp_calc + (1 if rank_calc < reste else 0)
        x_loc   = sum(col_dim//nbp_calc + (1 if c < reste else 0) for c in range(rank_calc))
        cells   = np.zeros((dim[0], nx_loc+2), dtype=np.uint8)
        for ig, jg in pattern:
            jl = jg - x_loc + 1
            if 1 <= jl <= nx_loc:
                cells[ig, jl] = 1
        before = (rank_calc - 1) % nbp_calc
        after  = (rank_calc + 1) % nbp_calc

    # ── ROW ──────────────────────────────────
    elif args.decomp == "row":
        row_dim = dim[0]
        reste   = row_dim % nbp_calc
        ny_loc  = row_dim // nbp_calc + (1 if rank_calc < reste else 0)
        y_loc   = sum(row_dim//nbp_calc + (1 if r < reste else 0) for r in range(rank_calc))
        cells   = np.zeros((ny_loc+2, dim[1]), dtype=np.uint8)
        for ig, jg in pattern:
            il = ig - y_loc + 1
            if 1 <= il <= ny_loc:
                cells[il, jg] = 1
        before = (rank_calc - 1) % nbp_calc
        after  = (rank_calc + 1) % nbp_calc

    # ── 2D ───────────────────────────────────
    elif args.decomp == "2d":
        nbp_row = int(np.sqrt(nbp_calc))
        while nbp_calc % nbp_row != 0:
            nbp_row -= 1
        nbp_col  = nbp_calc // nbp_row
        rank_row = rank_calc // nbp_col
        rank_col = rank_calc  % nbp_col

        row_dim = dim[0]; reste_r = row_dim % nbp_row
        ny_loc  = row_dim // nbp_row + (1 if rank_row < reste_r else 0)
        y_loc   = sum(row_dim//nbp_row + (1 if r < reste_r else 0) for r in range(rank_row))

        col_dim = dim[1]; reste_c = col_dim % nbp_col
        nx_loc  = col_dim // nbp_col + (1 if rank_col < reste_c else 0)
        x_loc   = sum(col_dim//nbp_col + (1 if c < reste_c else 0) for c in range(rank_col))

        cells = np.zeros((ny_loc+2, nx_loc+2), dtype=np.uint8)
        for ig, jg in pattern:
            if y_loc <= ig < y_loc+ny_loc and x_loc <= jg < x_loc+nx_loc:
                cells[ig-y_loc+1, jg-x_loc+1] = 1

        def grank(rr, rc):
            return (rr % nbp_row)*nbp_col + (rc % nbp_col) + 1
        up_g  = grank(rank_row-1, rank_col);  dn_g  = grank(rank_row+1, rank_col)
        lf_g  = grank(rank_row,   rank_col-1); rt_g  = grank(rank_row,   rank_col+1)
        ul_g  = grank(rank_row-1, rank_col-1); ur_g  = grank(rank_row-1, rank_col+1)
        dl_g  = grank(rank_row+1, rank_col-1); dr_g  = grank(rank_row+1, rank_col+1)
else:
    com_screen = globCom.Split(color=0, key=0)


# ─────────────────────────────────────────────
# Fonction de calcul commune
# ─────────────────────────────────────────────
def compute(cells, decomp):
    ny, nx = cells.shape

    if decomp == "column":
        r0, r1, c0, c1 = 0, ny, 1, nx-1
    elif decomp == "row":
        r0, r1, c0, c1 = 1, ny-1, 0, nx
    else:  # 2d
        r0, r1, c0, c1 = 1, ny-1, 1, nx-1

    c = cells[r0:r1, c0:c1]

    voisins = (
        np.roll(cells, -1, axis=0)[r0:r1, c0:c1] +
        np.roll(cells,  1, axis=0)[r0:r1, c0:c1] +
        np.roll(cells, -1, axis=1)[r0:r1, c0:c1] +
        np.roll(cells,  1, axis=1)[r0:r1, c0:c1] +
        np.roll(np.roll(cells, -1, axis=0), -1, axis=1)[r0:r1, c0:c1] +
        np.roll(np.roll(cells, -1, axis=0),  1, axis=1)[r0:r1, c0:c1] +
        np.roll(np.roll(cells,  1, axis=0), -1, axis=1)[r0:r1, c0:c1] +
        np.roll(np.roll(cells,  1, axis=0),  1, axis=1)[r0:r1, c0:c1]
    )

    next_cells = np.zeros_like(cells)
    next_cells[r0:r1, c0:c1] = (
        ((c == 1) & ((voisins == 2) | (voisins == 3))) |
        ((c == 0) & (voisins == 3))
    ).astype(np.uint8)

    return next_cells


# ─────────────────────────────────────────────
# Boucle principale
# ─────────────────────────────────────────────
temps_calc = 0.0
temps_comm = 0.0
N = args.iters

for it in range(N):

    if rank == 0:
        # Rank 0 ne calcule pas, il fait juste avancer le gather
        globCom.gather([], root=0)
        continue

    # ── échange ghost cells ──────────────────
    t0 = time.time()

    # CORRECTION : on n'échange que s'il y a plus d'un processus de calcul
    if args.decomp == "column" and nbp_calc > 1:
        buf_b = np.empty(cells.shape[0], dtype=np.uint8)
        buf_a = np.empty(cells.shape[0], dtype=np.uint8)
        sc = np.ascontiguousarray(cells[:,1])
        ec = np.ascontiguousarray(cells[:,-2])
        if rank_calc % 2 == 0:
            com_calc.Send(sc, dest=before); com_calc.Recv(buf_a, source=after)
            com_calc.Send(ec, dest=after);  com_calc.Recv(buf_b, source=before)
        else:
            com_calc.Recv(buf_a, source=after);  com_calc.Send(sc, dest=before)
            com_calc.Recv(buf_b, source=before); com_calc.Send(ec, dest=after)
        cells[:,0] = buf_b; cells[:,-1] = buf_a

    elif args.decomp == "row" and nbp_calc > 1:
        sr = np.ascontiguousarray(cells[1,:])
        er = np.ascontiguousarray(cells[-2,:])
        if rank_calc % 2 == 0:
            com_calc.Send(sr, dest=before); com_calc.Recv(cells[-1,:], source=after)
            com_calc.Send(er, dest=after);  com_calc.Recv(cells[0,:],  source=before)
        else:
            com_calc.Recv(cells[-1,:], source=after);  com_calc.Send(sr, dest=before)
            com_calc.Recv(cells[0,:],  source=before); com_calc.Send(er, dest=after)

    elif args.decomp == "2d" and nbp_calc > 1:
        b_up=np.empty(nx_loc,dtype=np.uint8); b_dn=np.empty(nx_loc,dtype=np.uint8)
        b_lf=np.empty(ny_loc,dtype=np.uint8); b_rt=np.empty(ny_loc,dtype=np.uint8)
        b_ul=np.empty(1,dtype=np.uint8); b_ur=np.empty(1,dtype=np.uint8)
        b_dl=np.empty(1,dtype=np.uint8); b_dr=np.empty(1,dtype=np.uint8)
        reqs = [
            globCom.Irecv(b_up,source=up_g,tag=1), globCom.Irecv(b_dn,source=dn_g,tag=2),
            globCom.Irecv(b_lf,source=lf_g,tag=3), globCom.Irecv(b_rt,source=rt_g,tag=4),
            globCom.Irecv(b_ul,source=ul_g,tag=5), globCom.Irecv(b_ur,source=ur_g,tag=6),
            globCom.Irecv(b_dl,source=dl_g,tag=7), globCom.Irecv(b_dr,source=dr_g,tag=8),
            globCom.Isend(np.ascontiguousarray(cells[1,1:-1]),  dest=up_g,tag=2),
            globCom.Isend(np.ascontiguousarray(cells[-2,1:-1]), dest=dn_g,tag=1),
            globCom.Isend(np.ascontiguousarray(cells[1:-1,1]),  dest=lf_g,tag=4),
            globCom.Isend(np.ascontiguousarray(cells[1:-1,-2]), dest=rt_g,tag=3),
            globCom.Isend(np.array([cells[1,1]],  dtype=np.uint8), dest=ul_g,tag=8),
            globCom.Isend(np.array([cells[1,-2]], dtype=np.uint8), dest=ur_g,tag=7),
            globCom.Isend(np.array([cells[-2,1]], dtype=np.uint8), dest=dl_g,tag=6),
            globCom.Isend(np.array([cells[-2,-2]],dtype=np.uint8), dest=dr_g,tag=5),
        ]
        MPI.Request.Waitall(reqs)
        cells[0,1:-1]=b_up; cells[-1,1:-1]=b_dn
        cells[1:-1,0]=b_lf; cells[1:-1,-1]=b_rt
        cells[0,0]=b_ul[0]; cells[0,-1]=b_ur[0]
        cells[-1,0]=b_dl[0]; cells[-1,-1]=b_dr[0]

    temps_comm += time.time() - t0

    # ── calcul ──────────────────────────────
    t0 = time.time()
    cells = compute(cells, args.decomp)
    temps_calc += time.time() - t0

    # ── envoi résultats à rank 0 ─────────────
    globCom.gather([], root=0)


# ─────────────────────────────────────────────
# Résumé (rank 1 uniquement pour éviter doublons)
# ─────────────────────────────────────────────
if rank == 1:
    mc = temps_calc / N * 1000
    mm = temps_comm / N * 1000
    mt = mc + mm
    # Ligne parseable par le bash
    print(f"BENCH_RESULT decomp={args.decomp} nproc={nbp} calc_ms={mc:.3f} comm_ms={mm:.3f} total_ms={mt:.3f}")
    sys.stdout.flush()