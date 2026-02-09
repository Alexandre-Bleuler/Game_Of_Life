from mpi4py import MPI

globCom = MPI.COMM_WORLD.Dup()
nbp     = globCom.size
rank    = globCom.rank
name    = MPI.Get_processor_name()

com = globCom.Split(color=(rank!=0), key=rank)
print(com.size)
print(f"processus {rank} : {com.rank}")

