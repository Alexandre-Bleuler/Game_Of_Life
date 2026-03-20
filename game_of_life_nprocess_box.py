"""
Le jeu de la vie parallélisé selon des boîtes rectangulaires de tailles similaires.
"""

import pygame as pg
import numpy as np
import time
import sys
from mpi4py import MPI
import game_of_life_2process as gr
import game_of_life_nprocess_column as grc
import game_of_life_nprocess_row as grr


ITER_PRINT=500 # Printing performances every ITER_PRINT iterations

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
        
        if nbp<2:
            raise ValueError("Can't use a Grid_Box object with nbp_row<2 or nbp_col<2!")
        
        # Defining the local dimensions of the grid and saving the corresponding
        # begining column and row in the global grid.

        row_dim = dim[0]
        col_dim = dim[1]
        reste_row = row_dim % nbp_row
        reste_col = col_dim % nbp_col

        ny_loc = row_dim // nbp_row + (1 if rank_row < reste_row else 0)
        y_loc = ny_loc*rank_row + (reste_row if rank>=reste_row else 0)

        nx_loc = col_dim // nbp_col + (1 if rank_col < reste_col else 0)
        x_loc = nx_loc*rank_col + (reste_col if rank>=reste_col else 0)

        self.ny_loc = ny_loc
        self.y_loc = y_loc
        self.nx_loc = nx_loc
        self.x_loc = x_loc
        self.nx_global=dim[1]
        self.dimensions = (ny_loc + 2, nx_loc + 2)

        # Initializing the grid

        if init_pattern is not None:
            self.cells = np.zeros(self.dimensions, dtype=np.uint8)
            indices_i = np.array([(v[0]-y_loc+1)%dim[0] for v in init_pattern])
            indices_j = np.array([(v[1]-x_loc+1)%dim[1] for v in init_pattern])
            mask = (indices_i< self.dimensions[0])*(indices_j<self.dimensions[1])
            self.cells[indices_i[mask],indices_j[mask]] = 1 
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
                        diff_cells.append((self.y_loc+i-1)*self.nx_global+self.x_loc+j-1)
                    else:
                        next_cells[i,j] = 1 # Sinon elle reste vivante
                elif nb_voisines_vivantes == 3: # Cas où cellule morte mais entourée exactement de trois vivantes
                    next_cells[i,j] = 1         # Naissance de la cellule
                    diff_cells.append((self.y_loc+i-1)*self.nx_global+self.x_loc+j-1)
                else:
                    next_cells[i,j] = 0         # Morte, elle reste morte.
        self.cells = next_cells
        return np.ascontiguousarray(diff_cells)


if __name__ == '__main__':
    import time
    import sys

    from mpi4py import MPI

    globCom = MPI.COMM_WORLD.Dup()
    nbp     = globCom.size
    rank    = globCom.rank
    name    = MPI.Get_processor_name()

    if nbp < 2: 
        raise ValueError("Need at least 2 processes to parallelize the Game of Life!")

    if rank==0:
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
    choice = 'glider'
    if len(sys.argv) > 1 :
        choice = sys.argv[1]
    resx = 800
    resy = 800
    if len(sys.argv) > 3 :
        resx = int(sys.argv[2])
        resy = int(sys.argv[3])
    if rank==0:
        print(f"Pattern initial choisi : {choice}")
        print(f"resolution ecran : {resx,resy}")
    try:
        init_pattern = dico_patterns[choice]
    except KeyError:
        print("No such pattern. Available ones are:", dico_patterns.keys())
        exit(1)
    
    if rank==0:

        # Spliting globCom

        com_screen= globCom.Split(color=(rank!=0), key=rank)
        nbp_screen=com_screen.size
        rank_screen=com_screen.rank

        # Creating the global grid

        grid = gr.Grille(*init_pattern)
        appli = gr.App((resx, resy), grid)
        first_iter=True
        diff=[]
        

    else: 

        appli = None

        # Spliting globCom

        com_calc = globCom.Split(color=(rank!=0), key=rank)
        nbp_calc=com_calc.size
        rank_calc=com_calc.rank

        # Writing nbp_calc as m*n with m>=1 and m<=n
        # while trying to have m and n the closest possible. 

        m=int(np.sqrt(nbp_calc))
        while nbp_calc % m != 0:
            m -= 1
        n = nbp_calc // m
    
        # Putting more processes on the biggest global grid's axis

        if init_pattern[0][0]>=init_pattern[0][1]:
            np_on_rows=n
            np_on_cols=m
        else:
            np_on_rows=m
            np_on_cols=n

        print(f"Grille de processus: {np_on_rows}  x {np_on_cols} ")

        # Spliting computing processes along lines of the grid

        com_row=com_calc.Split(color=rank_calc//np_on_cols, key=rank)
        nbp_row=com_row.size
        rank_row=com_row.rank

        # Spliting computing processes along columns of the grid

        com_col=com_calc.Split(color=rank_calc//np_on_rows, key=rank)
        nbp_col=com_col.size
        rank_col=com_col.rank
        
        # Creating the local grid

        if nbp_row==1 and nbp_col==1:
            grid_loc=gr.Grille(*init_pattern)
        elif nbp_row==1:
            grid_loc=grc.Grille_Column(rank_col, nbp_col, *init_pattern)
        elif nbp_col==1:
            grid_loc=grr.Grille_Row(rank_row, nbp_row, *init_pattern)
        else:
            grid_loc=Grille_box(rank_row, nbp_row, rank_col, nbp_col, *init_pattern)
        # Getting row neighbours in com_row
        
        if nbp_row>1:
            next_row_process=(rank_row+1)%nbp_row
            before_row_process=(rank_row-1)%nbp_row

        # Getting column neighbours in com_row

        if nbp_col>1:
            next_col_process=(rank_col+1)%nbp_col
            before_col_process=(rank_col-1)%nbp_col

        # Getting diagonal neighbours in com_calc

        if nbp_row>1 and nbp_col>1:
            low_left_process= before_row_process*nbp_col + before_col_process
            low_right_process= before_row_process*nbp_col + next_col_process
            up_left_process= next_row_process*nbp_col + before_col_process
            up_right_process= next_row_process*nbp_col + next_col_process

        # Making time measurements

        total_compute_time=0
        number_iter=0
        
    mustContinue = True
    
    while mustContinue:

        if rank == 0:
           
        #time.sleep(0.5) # A régler ou commenter pour vitesse maxi

            if first_iter:
                first_iter=False

            else:
                diff_list=globCom.gather(diff, root=0)
                for sub_diff in diff_list:
                    gr.update_grid(grid, sub_diff)

            t2 = time.time()
            appli.draw()
            t3 = time.time()
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    mustContinue = False

        else:

            t1 = time.time()

            # Getting value for ghost lines 
            # WARNING: the order of the Send and Recv is crucial in order to avoid blocking the program.

            # General case

            if nbp_row>1 and nbp_col>1:

                # Making copies in order to avoid "BufferError: dlpack: buffer is not contiguous"
                first_col_no_phantom = np.copy(grid_loc.cells[1:-1,1])
                last_col_no_phantom = np.copy(grid_loc.cells[1:-1,-2])
                phantom_col_before = np.empty(grid_loc.dimensions[0]-2, dtype=np.uint8)
                phantom_col_next = np.empty(grid_loc.dimensions[0]-2, dtype=np.uint8)

                if rank_row%2==0:
                    # Ghost rows
                    com_row.Send(grid_loc.cells[1,1:-1], dest=before_row_process)
                    com_row.Recv(grid_loc.cells[-1,1:-1], source=next_row_process)
                    com_row.Send(grid_loc.cells[-2,1:-1], dest=next_row_process)
                    com_row.Recv(grid_loc.cells[0,1:-1], source=before_row_process)

                    # Ghost rows
                    com_col.Send(first_col_no_phantom, dest=before_col_process)
                    com_col.Recv(phantom_col_next, source=next_col_process)
                    com_col.Send(last_col_no_phantom, dest=next_col_process)
                    com_col.Recv(phantom_col_before, source=before_col_process)

                    # Ghost vertexes
                    com_calc.Send(grid_loc.cells[0,0], dest=low_left_process)
                    com_calc.Recv(grid_loc.cells[-1,-1], source=up_right_process)
                    com_calc.Send(grid_loc.cells[0,-1], dest=low_right_process)
                    com_calc.Recv(grid_loc.cells[-1,0], source=up_left_process)
                    com_calc.Send(grid_loc.cells[-1,0], dest=up_left_process)
                    com_calc.Recv(grid_loc.cells[0,-1], source=low_right_process)
                    com_calc.Send(grid_loc.cells[-1,-1], dest=up_right_process)
                    com_calc.Recv(grid_loc.cells[0,0], source=low_left_process) 

                else:
                    # Ghost rows
                    com_row.Recv(grid_loc.cells[-1,1:-1], source=next_row_process)
                    com_row.Send(grid_loc.cells[1,1:-1], dest=before_row_process)
                    com_row.Recv(grid_loc.cells[0,1:-1], source=before_row_process)
                    com_row.Send(grid_loc.cells[-2,1:-1], dest=next_row_process)
            
                    # Ghost columns
                    com_col.Recv(phantom_col_next, source=next_col_process)
                    com_col.Send(first_col_no_phantom, dest=before_col_process)
                    com_col.Recv(phantom_col_before, source=before_col_process)
                    com_col.Send(last_col_no_phantom, dest=next_col_process)

                    # Ghost vertexes
                    com_calc.Recv(grid_loc.cells[-1,-1], source=up_right_process)
                    com_calc.Send(grid_loc.cells[0,0], dest=low_left_process)
                    com_calc.Recv(grid_loc.cells[-1,0], source=up_left_process)
                    com_calc.Send(grid_loc.cells[0,-1], dest=low_right_process)
                    com_calc.Recv(grid_loc.cells[0,-1], source=low_right_process)
                    com_calc.Send(grid_loc.cells[-1,0], dest=up_left_process)
                    com_calc.Recv(grid_loc.cells[0,0], source=low_left_process)
                    com_calc.Send(grid_loc.cells[-1,-1], dest=up_right_process)

                grid_loc.cells[1:-1,0] = phantom_col_before
                grid_loc.cells[1:-1,-1] = phantom_col_next    

            # Case that decays to parallelizing only thrhough rows

            elif nbp_col==1:
                if nbp_row%2==0:
                    com_row.Send(grid_loc.cells[1,:], dest=before_row_process)
                    com_row.Recv(grid_loc.cells[-1,:], source=next_row_process)
                    com_row.Send(grid_loc.cells[-2,:], dest=next_row_process)
                    com_row.Recv(grid_loc.cells[0,:], source=before_row_process)
                else:
                    com_row.Recv(grid_loc.cells[-1,:], source=next_row_process)
                    com_row.Send(grid_loc.cells[1,:], dest=before_row_process)
                    com_row.Recv(grid_loc.cells[0,:], source=before_row_process)
                    com_row.Send(grid_loc.cells[-2,:], dest=next_row_process)

            # Case that decays to parallelizing only thrhough columns

            else:
                first_no_phantom = np.copy(grid_loc.cells[:,1])
                last_no_phantom = np.copy(grid_loc.cells[:,-2])
                phantom_before = np.empty(grid_loc.dimensions[0], dtype=np.uint8)
                phantom_next = np.empty(grid_loc.dimensions[0], dtype=np.uint8)

                if rank_calc%2==0:
                    com_col.Send(first_no_phantom, dest=before_col_process)
                    com_col.Recv(phantom_next, source=next_col_process)
                    com_col.Send(last_no_phantom, dest=next_col_process)
                    com_col.Recv(phantom_before, source=before_col_process)
                else:
                    com_col.Recv(phantom_next, source=next_col_process)
                    com_col.Send(first_no_phantom, dest=before_col_process)
                    com_col.Recv(phantom_before, source=before_col_process)
                    com_col.Send(last_no_phantom, dest=next_col_process)

                grid_loc.cells[:,0] = phantom_before
                grid_loc.cells[:,-1] = phantom_next
          

            # Computing the local grid next iteration 
        
            diff = grid_loc.compute_next_iteration() 
    
            t2=time.time()

           # Sending the results 

            globCom.gather(diff, root=0)   

            # Perfoamance measurements 

            total_compute_time+=t2-t1
            number_iter+=1
            if(number_iter%500==0):
                mean_iter_time=total_compute_time/number_iter
                print(f"""Computing process of rank {rank_calc} computed {number_iter} iterations,""",
                    f"""for an average computing time of {mean_iter_time} seconds.""", sep=" ")
                sys.stdout.flush()
      
    if rank==0:
        pg.quit()
        globCom.Abort()