"""
Le jeu de la vie
################
Le jeu de la vie est un automate cellulaire inventé par Conway se basant normalement sur une grille infinie
de cellules en deux dimensions. Ces cellules peuvent prendre deux états :
    - un état vivant
    - un état mort
A l'initialisation, certaines cellules sont vivantes, d'autres mortes.
Le principe du jeu est alors d'itérer de telle sorte qu'à chaque itération, une cellule va devoir interagir avec
les huit cellules voisines (gauche, droite, bas, haut et les quatre en diagonales.) L'interaction se fait selon les
règles suivantes pour calculer l'itération suivante :
    - Une cellule vivante avec moins de deux cellules voisines vivantes meurt ( sous-population )
    - Une cellule vivante avec deux ou trois cellules voisines vivantes reste vivante
    - Une cellule vivante avec plus de trois cellules voisines vivantes meurt ( sur-population )
    - Une cellule morte avec exactement trois cellules voisines vivantes devient vivante ( reproduction )

Pour ce projet, on change légèrement les règles en transformant la grille infinie en un tore contenant un
nombre fini de cellules. Les cellules les plus à gauche ont pour voisines les cellules les plus à droite
et inversement, et de même les cellules les plus en haut ont pour voisines les cellules les plus en bas
et inversement.

On itère ensuite pour étudier la façon dont évolue la population des cellules sur la grille.
"""


import pygame as pg
import numpy as np
import time
import sys
from mpi4py import MPI
import game_of_life_2process as gr

class Grille_box:
    """
    Grille torique décrivant l'automate cellulaire avec découpage 2D.
    
    Chaque processus possède une sous-grille avec des cellules fantômes (ghost cells)
    pour communiquer avec ses voisins.
    
    Les indices sont organisés ainsi:
        - cells[0, :] et cells[-1, :] : lignes fantômes (haut et bas)
        - cells[:, 0] et cells[:, -1] : colonnes fantômes (gauche et droite)
        - cells[1:-1, 1:-1] : cellules internes (celles qu'on calcule vraiment)
    """
    def __init__(self, rank_row, nbp_row, rank_col, nbp_col, dim, init_pattern=None, 
                 color_life=pg.Color("black"), color_dead=pg.Color("white")):
        

        row_dim = dim[0]
        reste_row = row_dim % nbp_row
        # Nombre de lignes locales pour ce processus
        ny_loc = row_dim // nbp_row + (1 if rank_row < reste_row else 0)
        
        # Calcul du décala pour savoir où on se trouve dans la grille globale
        # C'est important pour reconstruire l'image plus tard
        y_loc = sum(row_dim // nbp_row + (1 if r < reste_row else 0) for r in range(rank_row))
        
        #  Découpage selon les colonnes 
        col_dim = dim[1]
        reste_col = col_dim % nbp_col
        nx_loc = col_dim // nbp_col + (1 if rank_col < reste_col else 0)
        
        x_loc = sum(col_dim // nbp_col + (1 if c < reste_col else 0) for c in range(rank_col))
        
        # On garde ces infos pour les communications avec le processus d'affichage
        self.ny_loc = ny_loc
        self.y_loc = y_loc
        self.nx_loc = nx_loc
        self.x_loc = x_loc
        
        # Dimensions avec cellules fantômes +2 pour les ghost cells
        # Comme expliqué dans le Cours 4 page 13
        self.dimensions = (ny_loc + 2, nx_loc + 2)

        # Initialisation de la grille locale
        self.cells = np.zeros(self.dimensions, dtype=np.uint8)
        
        if init_pattern is not None:
            # On ne garde que les cellules du pattern qui sont dans notre domaine
            # C'est le principe de la décomposition de domaine
            for v in init_pattern:
                i_global, j_global = v
                if (y_loc <= i_global < y_loc + ny_loc) and (x_loc <= j_global < x_loc + nx_loc):
                    i_loc = i_global - y_loc + 1  # +1 pour sauter la ligne fantôme
                    j_loc = j_global - x_loc + 1  # +1 pour sauter la colonne fantôme
                    self.cells[i_loc, j_loc] = 1
        else:
            # Pattern aléatoire pour tester
            self.cells[1:-1, 1:-1] = np.random.randint(2, size=(ny_loc, nx_loc), dtype=np.uint8)
        
        self.col_life = color_life
        self.col_dead = color_dead

    def compute_next_iteration(self):

        ny, nx = self.dimensions
        next_cells = np.zeros(self.dimensions, dtype=np.uint8)
        diff_cells = []

        # On ne calcule que sur les cellules internes (pas sur les ghost cells)
        for i in range(1, ny-1): 
            i_above = (i+ny-1)%ny # +ny pour le cas particulier i==0
            i_below = (i+ny+1)%ny
            for j in range(1,nx-1): # not updating ghost columns, it will be done by another process
                j_left = (j-1+nx)%nx
                j_right= (j+1)%nx
                voisins_i = [i_above,i_above,i_above, i     , i      , i_below, i_below, i_below]
                voisins_j = [j_left ,j      ,j_right, j_left, j_right, j_left , j      , j_right]
                voisines = np.array(self.cells[voisins_i,voisins_j])
                nb_voisines_vivantes = np.sum(voisines)
                if self.cells[i,j] == 1: # Si la cellule est vivante
                    if (nb_voisines_vivantes < 2) or (nb_voisines_vivantes > 3):
                        next_cells[i,j] = 0 # Cas de sous ou sur population, la cellule meurt
                        diff_cells.append((i-1)*(nx-2)+j-1)
                    else:
                        next_cells[i,j] = 1 # Sinon elle reste vivante
                elif nb_voisines_vivantes == 3: # Cas où cellule morte mais entourée exactement de trois vivantes
                    next_cells[i,j] = 1         # Naissance de la cellule
                    diff_cells.append((i-1)*(nx-2)+j-1)
                else:
                    next_cells[i,j] = 0         # Morte, elle reste morte.
        self.cells = next_cells
        return np.ascontiguousarray(diff_cells)




def update_grid_2d(grid, y_loc, x_loc, diff, ny_loc, nx_loc):
    for number in diff: 
        i=number//nx_loc
        j=number%nx_loc
        grid.cells[y_loc+i,x_loc+j]=1-grid.cells[y_loc+i,x_loc+j]


if __name__ == '__main__':
    # Initialisation MPI
    globCom = MPI.COMM_WORLD.Dup()
    nbp = globCom.size
    rank = globCom.rank

    if nbp < 2:
        if rank == 0:
            print("Il faut au moins 2 processus.")
        sys.exit(1)

    #  Organisation des processus 
    #  on sépare explicitement rank 0 (affichage) des ranks 1..N-1 (calcul).
    # Dans le code original, rank 0 était inclus dans la grille de calcul ce qui
    # causait un deadlock parce-qu'il  calculait jamais mais était compté dans nbp_row/nbp_col.
    n_calc = nbp - 1          # nombre de processus de calcul
    calc_rank = rank - 1      # rang dans la grille de calcul (invalide pour rank 0)

    # On essaie d'avoir une grille la plus carrée possible
    nbp_row = int(np.sqrt(n_calc))
    while n_calc % nbp_row != 0:
        nbp_row -= 1
    nbp_col = n_calc // nbp_row
    
    if rank == 0:
        print("=" * 50)
        print("JEU DE LA VIE - VERSION PARALLELE MPI")
        print("=" * 50)
        print(f"Grille de processus: {nbp_row} lignes x {nbp_col} colonnes")
        print(f"Nombre total de processus: {nbp} (1 pour affichage, {n_calc} pour calcul)")
        print("=" * 50)

    # on calcule rank_row/rank_col uniquement pour les processus de calcul
    if rank != 0:
        rank_row = calc_rank // nbp_col
        rank_col = calc_rank % nbp_col

        # quand la grille n'était pas carrée (ex: 1x3, col_comm n'avait qu'1 seul processus).
        # On utilise uniquement globCom avec les rangs globaux des 8 voisins.
        # Le rang global d'un processus de calcul = son calc_rank + 1 (car rank 0 = affichage).
        def grank(rr, rc):
            """Rang global du processus de calcul à la position (rr, rc) dans la grille."""
            return (rr % nbp_row) * nbp_col + (rc % nbp_col) + 1

        up_g    = grank(rank_row - 1, rank_col)      # voisin haut
        down_g  = grank(rank_row + 1, rank_col)      # voisin bas
        left_g  = grank(rank_row,     rank_col - 1)  # voisin gauche
        right_g = grank(rank_row,     rank_col + 1)  # voisin droite
        ul_g    = grank(rank_row - 1, rank_col - 1)  # coin haut-gauche
        ur_g    = grank(rank_row - 1, rank_col + 1)  # coin haut-droit
        dl_g    = grank(rank_row + 1, rank_col - 1)  # coin bas-gauche
        dr_g    = grank(rank_row + 1, rank_col + 1)  # coin bas-droit

    if rank == 0:
        pg.init()
    
    dico_patterns = { # Dimension et pattern dans un tuple
        'blinker' : ((5,5),[(2,1),(2,2),(2,3)]),
        'toad'    : ((6,6),[(2,2),(2,3),(2,4),(3,3),(3,4),(3,5)]),
        "acorn"   : ((100,100), [(51,52),(52,54),(53,51),(53,52),(53,55),(53,56),(53,57)]),
        "beacon"  : ((6,6), [(1,3),(1,4),(2,3),(2,4),(3,1),(3,2),(4,1),(4,2)]),
        "boat" : ((5,5),[(1,1),(1,2),(2,1),(2,3),(3,2)]),
        "glider": ((100,90),[(1,1),(2,2),(2,3),(3,1),(3,2)]),
        "glider_gun": ((400,400),[(51,76),(52,74),(52,76),(53,64),(53,65),(53,72),(53,73),(53,86),(53,87),(54,63),(54,67),(54,72),(54,73),(54,86),(54,87),(55,52),(55,53),(55,62),(55,68),(55,72),(55,73),(56,52),(56,53),(56,62),(56,66),(56,68),(56,69),(56,74),(56,76),(57,62),(57,68),(57,76),(58,63),(58,67),(59,64),(59,65)]),
        "space_ship": ((25,25),[(11,13),(11,14),(12,11),(12,12),(12,14),(12,15),(13,11),(13,12),(13,13),(13,14),(14,12),(14,13)]),
        "die_hard" : ((100,100), [(51,57),(52,51),(52,52),(53,52),(53,56),(53,57),(53,58)]),
        "pulsar": ((17,17),[(2,4),(2,5),(2,6),(7,4),(7,5),(7,6),(9,4),(9,5),(9,6),(14,4),(14,5),(14,6),(2,10),(2,11),(2,12),(7,10),(7,11),(7,12),(9,10),(9,11),(9,12),(14,10),(14,11),(14,12),(4,2),(5,2),(6,2),(4,7),(5,7),(6,7),(4,9),(5,9),(6,9),(4,14),(5,14),(6,14),(10,2),(11,2),(12,2),(10,7),(11,7),(12,7),(10,9),(11,9),(12,9),(10,14),(11,14),(12,14)]),
        "floraison" : ((40,40), [(19,18),(19,19),(19,20),(20,17),(20,19),(20,21),(21,18),(21,19),(21,20)]),
        "block_switch_engine" : ((400,400), [(201,202),(201,203),(202,202),(202,203),(211,203),(212,204),(212,202),(214,204),(214,201),(215,201),(215,202),(216,201)]),
        "u" : ((200,200), [(101,101),(102,102),(103,102),(103,101),(104,103),(105,103),(105,102),(105,101),(105,105),(103,105),(102,105),(101,105),(101,104)]),
        "flat" : ((200,400), [(80,200),(81,200),(82,200),(83,200),(84,200),(85,200),(86,200),(87,200), (89,200),(90,200),(91,200),(92,200),(93,200),(97,200),(98,200),(99,200),(106,200),(107,200),(108,200),(109,200),(110,200),(111,200),(112,200),(114,200),(115,200),(116,200),(117,200),(118,200)])
    }
    
    # Récupération du choix depuis la ligne de commande
    
    choice = 'glider'
    if len(sys.argv) > 1 :
        choice = sys.argv[1]
    resx = 800
    resy = 800
    if len(sys.argv) > 3 :
        resx = int(sys.argv[2])
        resy = int(sys.argv[3])
    print(f"Pattern initial choisi : {choice}")
    print(f"resolution ecran : {resx,resy}")
    try:
        init_pattern = dico_patterns[choice]
    except KeyError:
        print("No such pattern. Available ones are:", dico_patterns.keys())
        exit(1)
    

    if rank == 0:
        grid_global = gr.Grille(*init_pattern)
        appli = gr.App((resx, resy), grid_global)
    else:
        grid_loc = Grille_box(rank_row, nbp_row, rank_col, nbp_col,*init_pattern)


    mustContinue = True
    iter_count = 0
    
    # mesures de performance
    temps_calcul = 0.0
    temps_comm = 0.0
    debut_global = time.time()
    
    while mustContinue:
        if rank == 0:
            

            # on affiched'abord l'état courant (permet de voir l'état initial
            # dès le démarrage), puis on attend les mises à jour des workers
            appli.draw()
            time.sleep(0.05)

            for event in pg.event.get():
                if event.type == pg.QUIT:
                    mustContinue = False
                    #signal d'arrêt envoyé avec tag=99 dédié pour ne pas
                    # interférer avec les messages de données (tag=50)
                    for i in range(1, nbp):
                        globCom.send(False, dest=i, tag=99)

            if not mustContinue:
                break

            # Réception des sous-grilles de tous les processus de calcul
            for n in range(n_calc):
                data = globCom.recv(source=MPI.ANY_SOURCE, tag=50)
                n, y_loc, x_loc, diff, ny_loc_r, nx_loc_r = data
                update_grid_2d(grid_global, y_loc, x_loc, diff, ny_loc_r, nx_loc_r)

        else:

            debut_comm = time.time()

            ny_loc = grid_loc.ny_loc
            nx_loc = grid_loc.nx_loc

            # échange des ghost cells via globCom uniquement (plus de row_comm/col_comm).
            # La règle des tags est symétrique : si je reçois "du haut" avec tag=1,
            # mon voisin du haut m'envoie avec tag=1 parce que pour lui je suis "son bas".
            # Ainsi chaque paire est cohérente des deux côtés.

            # Buffers de réception pour les ghost cells
            buf_up    = np.empty(nx_loc, dtype=np.uint8)
            buf_down  = np.empty(nx_loc, dtype=np.uint8)
            buf_left  = np.empty(ny_loc, dtype=np.uint8)
            buf_right = np.empty(ny_loc, dtype=np.uint8)
            buf_ul    = np.empty(1, dtype=np.uint8)
            buf_ur    = np.empty(1, dtype=np.uint8)
            buf_dl    = np.empty(1, dtype=np.uint8)
            buf_dr    = np.empty(1, dtype=np.uint8)

            # Données à envoyer : bords intérieurs de la sous-grille 
            send_top    = np.ascontiguousarray(grid_loc.cells[1,  1:-1])   # ma 1ère ligne réelle
            send_bottom = np.ascontiguousarray(grid_loc.cells[-2, 1:-1])  # ma dernière ligne réelle
            send_left   = np.ascontiguousarray(grid_loc.cells[1:-1, 1])   # ma 1ère colonne réelle
            send_right  = np.ascontiguousarray(grid_loc.cells[1:-1, -2])  # ma dernière colonne réelle
            send_ul     = np.array([grid_loc.cells[1,   1]],  dtype=np.uint8)
            send_ur     = np.array([grid_loc.cells[1,  -2]],  dtype=np.uint8)
            send_dl     = np.array([grid_loc.cells[-2,  1]],  dtype=np.uint8)
            send_dr     = np.array([grid_loc.cells[-2, -2]],  dtype=np.uint8)

            # Lancer toutes les réceptions et envois non-bloquants en une seule liste
            # tag=1 : "ligne venant du haut"   tag=2 : "ligne venant du bas"
            # tag=3 : "col venant de gauche"   tag=4 : "col venant de droite"
            # tag=5..8 : coins (ul, ur, dl, dr)
            reqs = [
                globCom.Irecv(buf_up,    source=up_g,    tag=1),
                globCom.Irecv(buf_down,  source=down_g,  tag=2),
                globCom.Irecv(buf_left,  source=left_g,  tag=3),
                globCom.Irecv(buf_right, source=right_g, tag=4),
                globCom.Irecv(buf_ul,    source=ul_g,    tag=5),
                globCom.Irecv(buf_ur,    source=ur_g,    tag=6),
                globCom.Irecv(buf_dl,    source=dl_g,    tag=7),
                globCom.Irecv(buf_dr,    source=dr_g,    tag=8),
                # Mon voisin du haut reçoit ma ligne du haut avec tag=2 (je suis son bas)
                globCom.Isend(send_top,    dest=up_g,    tag=2),
                globCom.Isend(send_bottom, dest=down_g,  tag=1),
                globCom.Isend(send_left,   dest=left_g,  tag=4),
                globCom.Isend(send_right,  dest=right_g, tag=3),
                globCom.Isend(send_ul,     dest=ul_g,    tag=8),
                globCom.Isend(send_ur,     dest=ur_g,    tag=7),
                globCom.Isend(send_dl,     dest=dl_g,    tag=6),
                globCom.Isend(send_dr,     dest=dr_g,    tag=5),
            ]
            MPI.Request.Waitall(reqs)

            # Copier les données reçues dans les ghost cells
            grid_loc.cells[0,  1:-1] = buf_up      # ghost haut
            grid_loc.cells[-1, 1:-1] = buf_down    # ghost bas
            grid_loc.cells[1:-1, 0]  = buf_left    # ghost gauche
            grid_loc.cells[1:-1, -1] = buf_right   # ghost droite
            grid_loc.cells[0,  0]    = buf_ul[0]   # coin haut-gauche
            grid_loc.cells[0,  -1]   = buf_ur[0]   # coin haut-droit
            grid_loc.cells[-1, 0]    = buf_dl[0]   # coin bas-gauche
            grid_loc.cells[-1, -1]   = buf_dr[0]   # coin bas-droit

            temps_comm += time.time() - debut_comm

            # Calcul de la prochaine génération 
            debut_calcul = time.time()
            diff = grid_loc.compute_next_iteration()
            temps_calcul += time.time() - debut_calcul

            # Envoi de la sous-grille complète au processus 0
            # on envoie aussi ny_loc pour que update_grid_2d puisse
            # reconstruire correctement la zone dans la grille globale
            globCom.send([rank, grid_loc.y_loc, grid_loc.x_loc, diff,
                          grid_loc.ny_loc, grid_loc.nx_loc], dest=0, tag=50)

            iter_count += 1
            
            # Affichage des perfs toutes les 100 itérations
            if iter_count % 100 == 0 and rank == 1:
                print(f"\n[Perf] Itération {iter_count}:")
                print(f"       Temps calcul: {temps_calcul/iter_count*1000:.2f} ms/it")
                print(f"       Temps comm: {temps_comm/iter_count*1000:.2f} ms/it")
                print(f"       Ratio comm/calcul: {temps_comm/temps_calcul:.2f}")

            # MODIF: tag=99 dédié pour le signal d'arrêt afin de ne pas
            # confondre avec les messages de données 
            if globCom.Iprobe(source=0, tag=99):
                mustContinue = globCom.recv(source=0, tag=99)

    if rank == 0:
        pg.quit()
        print("\n" + "=" * 50)
        print("RÉSUMÉ DES PERFORMANCES")
        print("=" * 50)
        print(f"Nombre de processus de calcul: {n_calc}")
        print(f"Grille de processus: {nbp_row}x{nbp_col}")
        print(f"Dimensions globales: {init_pattern[0][0]}x{init_pattern[0][1]}")
        print("=" * 50)
    
    elif rank == 1:
        temps_total = time.time() - debut_global
        print(f"\n[Rank {rank}] Temps total: {temps_total:.2f} s")
        print(f"[Rank {rank}] Itérations: {iter_count}")
        if iter_count > 0:
            print(f"[Rank {rank}] Temps moyen par itération: {temps_total/iter_count*1000:.2f} ms")
            print(f"[Rank {rank}] Dont calcul: {temps_calcul/iter_count*1000:.2f} ms ({temps_calcul/temps_total*100:.1f}%)")
            print(f"[Rank {rank}] Dont comm: {temps_comm/iter_count*1000:.2f} ms ({temps_comm/temps_total*100:.1f}%)")