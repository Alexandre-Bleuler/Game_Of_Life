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
règles suivantes pour calculer l'irération suivante :
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

class Grille:
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
        
        # ----- Découpage selon les lignes (load balancing) -----
        # Chaque processus reçoit un nombre de lignes qui peut varier d'une unité
        # pour équilibrer quand la division n'est pas parfaite
        row_dim = dim[0]
        reste_row = row_dim % nbp_row
        # Nombre de lignes locales pour ce processus
        ny_loc = row_dim // nbp_row + (1 if rank_row < reste_row else 0)
        
        # Calcul du décalage (offset) pour savoir où on se trouve dans la grille globale
        # C'est important pour reconstruire l'image plus tard
        y_loc = 0
        for r in range(rank_row):
            y_loc += row_dim // nbp_row + (1 if r < reste_row else 0)
        
        # ----- Découpage selon les colonnes -----
        col_dim = dim[1]
        reste_col = col_dim % nbp_col
        nx_loc = col_dim // nbp_col + (1 if rank_col < reste_col else 0)
        
        x_loc = 0
        for c in range(rank_col):
            x_loc += col_dim // nbp_col + (1 if c < reste_col else 0)
        
        # On garde ces infos pour les communications avec le processus d'affichage
        self.ny_loc = ny_loc
        self.y_loc = y_loc
        self.nx_loc = nx_loc
        self.x_loc = x_loc
        
        # Dimensions avec cellules fantômes (+2 pour les ghost cells)
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
        """
        Calcule la prochaine génération.
        IMPORTANT: Les ghost cells doivent être à jour avant d'appeler cette fonction!
        """
        ny, nx = self.dimensions
        next_cells = np.zeros(self.dimensions, dtype=np.uint8)
        diff_cells = []
        
        # On ne calcule que sur les cellules internes (pas sur les ghost cells)
        for i in range(1, ny-1):
            for j in range(1, nx-1):
                # Compter les voisins - on utilise TOUTES les cellules (internes + ghost)
                # grace au modulo, ça gère automatiquement les bords toriques
                voisins = 0
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        # Le modulo sur ny/nx permet d'accéder aux ghost cells
                        ni = (i + di) % ny
                        nj = (j + dj) % nx
                        voisins += self.cells[ni, nj]
                
                # Application des règles du jeu de la vie
                if self.cells[i, j] == 1:
                    if voisins in [2, 3]:
                        next_cells[i, j] = 1
                    else:
                        # La cellule meurt - on note sa position locale
                        diff_cells.append((i-1) * self.nx_loc + (j-1))
                else:
                    if voisins == 3:
                        next_cells[i, j] = 1
                        diff_cells.append((i-1) * self.nx_loc + (j-1))
        
        self.cells = next_cells
        return diff_cells


class App:
    """
    Fenêtre d'affichage - Ne tourne que sur le processus 0
    """
    def __init__(self, geometry, grid):
        self.grid = grid
        # Calcul de la taille d'une cellule en pixels
        self.size_x = geometry[1] // grid.dimensions[1]
        self.size_y = geometry[0] // grid.dimensions[0]
        
        # On affiche les lignes de la grille seulement si les cellules sont assez grandes
        if self.size_x > 4 and self.size_y > 4:
            self.draw_color = pg.Color('lightgrey')
        else:
            self.draw_color = None
            
        # Ajustement de la taille de la fenêtre
        self.width = grid.dimensions[1] * self.size_x
        self.height = grid.dimensions[0] * self.size_y
        # Création de la fenêtre Pygame
        self.screen = pg.display.set_mode((self.width, self.height))
        pg.display.set_caption(f"Jeu de la Vie - MPI {MPI.COMM_WORLD.size} processus")

    def compute_rectangle(self, i, j):
        """Calcul le rectangle correspondant à la cellule (i,j) pour l'affichage"""
        return (self.size_x * j, self.height - self.size_y * (i + 1), self.size_x, self.size_y)

    def draw(self):
        """Dessine toute la grille"""
        # Dessiner chaque cellule
        for i in range(self.grid.dimensions[0]):
            for j in range(self.grid.dimensions[1]):
                if self.grid.cells[i, j] == 0:
                    color = self.grid.col_dead
                else:
                    color = self.grid.col_life
                self.screen.fill(color, self.compute_rectangle(i, j))
        
        # Dessiner les lignes de la grille si demandé
        if self.draw_color is not None:
            for i in range(self.grid.dimensions[0]):
                pg.draw.line(self.screen, self.draw_color, 
                           (0, i * self.size_y), (self.width, i * self.size_y))
            for j in range(self.grid.dimensions[1]):
                pg.draw.line(self.screen, self.draw_color,
                           (j * self.size_x, 0), (j * self.size_x, self.height))
        
        pg.display.update()


def update_grid_2d(grid, y_loc, x_loc, diff, nx_loc):
    """
    Met à jour la grille globale à partir des différences locales
    Fonction utilitaire pour le processus d'affichage
    """
    for idx in diff:
        i = idx // nx_loc
        j = idx % nx_loc
        grid.cells[y_loc + i, x_loc + j] = 1 - grid.cells[y_loc + i, x_loc + j]


if __name__ == '__main__':
    # ----- Initialisation MPI -----
    globCom = MPI.COMM_WORLD.Dup()
    nbp = globCom.size
    rank = globCom.rank

    # ----- Organisation des processus en grille 2D -----
    # On essaie d'avoir une grille la plus carrée possible
    nbp_row = int(np.sqrt(nbp))
    while nbp % nbp_row != 0:
        nbp_row -= 1
    nbp_col = nbp // nbp_row
    
    rank_row = rank // nbp_col
    rank_col = rank % nbp_col

    if rank == 0:
        print("=" * 50)
        print("JEU DE LA VIE - VERSION PARALLELE MPI")
        print("=" * 50)
        print(f"Grille de processus: {nbp_row} lignes x {nbp_col} colonnes")
        print(f"Nombre total de processus: {nbp} (1 pour affichage, {nbp-1} pour calcul)")
        print("=" * 50)

    # Création des communicateurs pour les lignes et colonnes
    # Ça facilite les communications entre voisins
    row_comm = globCom.Split(color=rank_row, key=rank_col)
    col_comm = globCom.Split(color=rank_col, key=rank_row)

    # Identification des voisins (gestion du tore)
    up = (rank_row - 1) % nbp_row
    down = (rank_row + 1) % nbp_row
    left = (rank_col - 1) % nbp_col
    right = (rank_col + 1) % nbp_col

    # Rangs globaux des voisins (pour les communications diagonales)
    up_rank = up * nbp_col + rank_col
    down_rank = down * nbp_col + rank_col
    left_rank = rank_row * nbp_col + left
    right_rank = rank_row * nbp_col + right
    up_left_rank = up * nbp_col + left
    up_right_rank = up * nbp_col + right
    down_left_rank = down * nbp_col + left
    down_right_rank = down * nbp_col + right

    # ----- Initialisation Pygame (seulement sur le processus 0) -----
    if rank == 0:
        pg.init()
    
    # ----- Paramètres du problème -----
    # Dictionnaire des patterns pré-définis
    dico_patterns = {
        'blinker': ((5,5), [(2,1), (2,2), (2,3)]),
        'toad': ((6,6), [(2,2), (2,3), (2,4), (3,3), (3,4), (3,5)]),
        'glider': ((100,90), [(1,1), (2,2), (2,3), (3,1), (3,2)]),
        'acorn': ((100,100), [(51,52), (52,54), (53,51), (53,52), (53,55), (53,56), (53,57)]),
        'beacon': ((6,6), [(1,3), (1,4), (2,3), (2,4), (3,1), (3,2), (4,1), (4,2)]),
        'pulsar': ((17,17), [(2,4), (2,5), (2,6), (7,4), (7,5), (7,6), (9,4), (9,5), (9,6), 
                            (14,4), (14,5), (14,6), (2,10), (2,11), (2,12), (7,10), (7,11), 
                            (7,12), (9,10), (9,11), (9,12), (14,10), (14,11), (14,12)]),
    }
    
    # Récupération du choix depuis la ligne de commande
    choice = 'glider'
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    
    resx, resy = 800, 800
    if len(sys.argv) > 3:
        resx, resy = int(sys.argv[2]), int(sys.argv[3])
    
    if rank == 0:
        print(f"Pattern initial : {choice}")
        print(f"Résolution écran : {resx, resy}")
    
    # ----- Distribution du pattern à tous les processus -----
    # Le processus 0 lit le pattern et le broadcast à tout le monde
    if rank == 0:
        try:
            dim_globale, pattern_global = dico_patterns[choice]
        except KeyError:
            print(f"Erreur: pattern '{choice}' inconnu")
            print("Patterns disponibles:", list(dico_patterns.keys()))
            globCom.Abort(1)
    else:
        dim_globale = None
        pattern_global = None
    
    # Broadcast des dimensions globales
    dim_globale = globCom.bcast(dim_globale, root=0)
    
    # Broadcast du pattern (plus compliqué car c'est une liste de tuples)
    if rank == 0:
        pattern_size = len(pattern_global)
    else:
        pattern_size = None
    
    pattern_size = globCom.bcast(pattern_size, root=0)
    
    if rank == 0:
        # Aplatir la liste pour l'envoi MPI
        pattern_flat = []
        for (i, j) in pattern_global:
            pattern_flat.extend([i, j])
        pattern_array = np.array(pattern_flat, dtype=np.int32)
    else:
        pattern_array = np.empty(pattern_size * 2, dtype=np.int32)
    
    # Broadcast du tableau aplati
    globCom.Bcast(pattern_array, root=0)
    
    # Reconstruction du pattern pour tous les processus
    pattern_local = []
    for k in range(0, len(pattern_array), 2):
        pattern_local.append((pattern_array[k], pattern_array[k+1]))
    
    # ----- Création des grilles locales -----
    # Tous les processus créent leur grille locale
    grid_loc = Grille(rank_row, nbp_row, rank_col, nbp_col, dim_globale, 
                      init_pattern=pattern_local)
    
    # Le processus 0 crée aussi une grille globale pour l'affichage
    if rank == 0:
        grid_global = Grille(0, 1, 0, 1, dim_globale, init_pattern=pattern_local)
        appli = App((resx, resy), grid_global)
    else:
        appli = None

    # ----- Boucle principale -----
    first_iter = True
    mustContinue = True
    iter_count = 0
    
    # Pour les mesures de performance (Cours 2)
    temps_calcul = 0.0
    temps_comm = 0.0
    debut_global = time.time()
    
    while mustContinue:
        if rank == 0:
            # ----- PROCESSUS 0 : AFFICHAGE -----
            
            if not first_iter:
                # Réception des mises à jour de tous les processus de calcul
                recu = 0
                while recu < nbp - 1:
                    # On reçoit de n'importe quelle source
                    data = globCom.recv(source=MPI.ANY_SOURCE)
                    proc_rank, y_loc, x_loc, diff, nx_loc = data
                    update_grid_2d(grid_global, y_loc, x_loc, diff, nx_loc)
                    recu += 1

            # Affichage de la grille
            appli.draw()
            
            # Gestion des événements (fermeture de la fenêtre)
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    mustContinue = False
                    # On prévient tout le monde qu'il faut s'arrêter
                    for i in range(1, nbp):
                        globCom.send(False, dest=i)
            
            if first_iter:
                first_iter = False
                
            # Petite pause pour avoir une animation visible
            time.sleep(0.05)

        else:
            # ----- PROCESSUS DE CALCUL (rank != 0) -----
            
            if not first_iter:
                debut_comm = time.time()
                
                # === ÉTAPE 1: Communications non-bloquantes ===
                # Comme expliqué dans le Cours 1, on utilise Irecv/Isend
                # pour pouvoir chevaucher calcul et communication
                req_list = []
                
                # Échange des lignes fantômes (vertical)
                # Réception de la ligne du haut (dans ghost cell du bas)
                req = row_comm.Irecv(grid_loc.cells[-1, 1:-1], source=down, tag=11)
                req_list.append(req)
                # Réception de la ligne du bas (dans ghost cell du haut)
                req = row_comm.Irecv(grid_loc.cells[0, 1:-1], source=up, tag=12)
                req_list.append(req)
                
                # Envoi de ma ligne du haut vers le haut
                req = row_comm.Isend(grid_loc.cells[1, 1:-1], dest=up, tag=12)
                req_list.append(req)
                # Envoi de ma ligne du bas vers le bas
                req = row_comm.Isend(grid_loc.cells[-2, 1:-1], dest=down, tag=11)
                req_list.append(req)
                
                # Attendre la fin des communications verticales
                MPI.Request.Waitall(req_list)
                req_list = []
                
                # Échange des colonnes fantômes (horizontal)
                # Réception de la colonne de gauche (dans ghost cell de droite)
                req = col_comm.Irecv(grid_loc.cells[1:-1, -1], source=right, tag=21)
                req_list.append(req)
                # Réception de la colonne de droite (dans ghost cell de gauche)
                req = col_comm.Irecv(grid_loc.cells[1:-1, 0], source=left, tag=22)
                req_list.append(req)
                
                # Envoi de ma colonne de gauche vers la gauche
                req = col_comm.Isend(grid_loc.cells[1:-1, 1], dest=left, tag=22)
                req_list.append(req)
                # Envoi de ma colonne de droite vers la droite
                req = col_comm.Isend(grid_loc.cells[1:-1, -2], dest=right, tag=21)
                req_list.append(req)
                
                MPI.Request.Waitall(req_list)
                req_list = []
                
                # Échange des coins (diagonales) - plus simple en bloquant
                # Envoyer mes 4 coins internes
                globCom.send(grid_loc.cells[1, 1], dest=up_left_rank, tag=31)
                globCom.send(grid_loc.cells[1, -2], dest=up_right_rank, tag=32)
                globCom.send(grid_loc.cells[-2, 1], dest=down_left_rank, tag=33)
                globCom.send(grid_loc.cells[-2, -2], dest=down_right_rank, tag=34)
                
                # Recevoir les 4 coins fantômes
                grid_loc.cells[0, 0] = globCom.recv(source=down_right_rank, tag=31)
                grid_loc.cells[0, -1] = globCom.recv(source=down_left_rank, tag=32)
                grid_loc.cells[-1, 0] = globCom.recv(source=up_right_rank, tag=33)
                grid_loc.cells[-1, -1] = globCom.recv(source=up_left_rank, tag=34)
                
                temps_comm += time.time() - debut_comm
                
                # === ÉTAPE 2: Calcul de la prochaine génération ===
                debut_calcul = time.time()
                diff = grid_loc.compute_next_iteration()
                temps_calcul += time.time() - debut_calcul
                
                # === ÉTAPE 3: Envoi des résultats au processus 0 ===
                globCom.send([rank, grid_loc.y_loc, grid_loc.x_loc, diff, grid_loc.nx_loc], 
                           dest=0)

                iter_count += 1
                
                # Affichage des perfs toutes les 100 itérations
                if iter_count % 100 == 0 and rank == 1:  # Seulement le processus 1 pour éviter la pollution
                    print(f"\n[Perf] Itération {iter_count}:")
                    print(f"       Temps calcul: {temps_calcul/iter_count*1000:.2f} ms/it")
                    print(f"       Temps comm: {temps_comm/iter_count*1000:.2f} ms/it")
                    print(f"       Ratio comm/calcul: {temps_comm/temps_calcul:.2f}")

            if first_iter:
                first_iter = False
            
            # Vérifier si le processus 0 demande l'arrêt
            if globCom.Iprobe(source=0):
                mustContinue = globCom.recv(source=0)

    # ----- Fin du programme -----
    if rank == 0:
        pg.quit()
        print("\n" + "=" * 50)
        print("RÉSUMÉ DES PERFORMANCES")
        print("=" * 50)
        print(f"Nombre de processus de calcul: {nbp-1}")
        print(f"Grille de processus: {nbp_row}x{nbp_col}")
        print(f"Dimensions globales: {dim_globale[0]}x{dim_globale[1]}")
        print("=" * 50)
    
    elif rank == 1:
        temps_total = time.time() - debut_global
        print(f"\n[Rank {rank}] Temps total: {temps_total:.2f} s")
        print(f"[Rank {rank}] Itérations: {iter_count}")
        print(f"[Rank {rank}] Temps moyen par itération: {temps_total/iter_count*1000:.2f} ms")
        print(f"[Rank {rank}] Dont calcul: {temps_calcul/iter_count*1000:.2f} ms ({temps_calcul/temps_total*100:.1f}%)")
        print(f"[Rank {rank}] Dont comm: {temps_comm/iter_count*1000:.2f} ms ({temps_comm/temps_total*100:.1f}%)")