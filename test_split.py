from mpi4py import MPI

globCom = MPI.COMM_WORLD.Dup()
nbp     = globCom.size
rank    = globCom.rank
name    = MPI.Get_processor_name()

if rank != 0:
    com_calculs = globCom.Split(color=(rank!=0), key=rank)
else:
    com_affichage = globCom.Split(color=(rank!=0), key=rank)
if rank != 0:
    com.Set_name("Groupe de calcul")
else:
    com.Set_name("Groupe d'affichage")
print(com.size)
print(f"communicator {com.Get_name()} processus {rank} : {com.rank}")