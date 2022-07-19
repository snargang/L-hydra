import sys, time, pdb, os, argparse

from lm_emb_hydra_functions import *
from numpy import *
from scipy.optimize import minimize, fmin_l_bfgs_b
from mpi4py import MPI # must use mpirun-openmpi-clang38

''' 
Parallel landmark embedding
'''

# MPI init
comm = MPI.COMM_WORLD
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()
name = MPI.Get_processor_name()

# Define static variables with argparse
parser = argparse.ArgumentParser()
parser.add_argument("-n", type=int, help="number of nodes in graph (int)")
parser.add_argument("-L", type=int, help="number of landmark nodes in graph (int)")
parser.add_argument("-d", type=int, help="dimension of embedding (int)")
parser.add_argument("-s", type=str, help="source location of datafiles (str)")

# set args to variables
args = parser.parse_args()
nodes = args.n
landmarks = args.L
dim = args.d
dataloc = args.s

if rank == 0:
	print("Loading data...")
	D_land, land_ids, nonland_ids = load_data_4land(nodes, landmarks, dataloc)
	x_start_tmp = load(os.path.join(dataloc, 'xstart.npy'))
	x_start = x_start_tmp[land_ids,]
else:
	D_land = None
	land_ids = None
	nonland_ids = None
	x_start = None
	
D_land = comm.bcast(D_land,root=0)
land_ids = comm.bcast(land_ids,root=0)
nonland_ids = comm.bcast(nonland_ids,root=0)
x_start = comm.bcast(x_start,root=0)
param_list = {'dim': dim, 'D_landmark' : D_land, 'landmarks0' : landmarks, 'x_start0' : x_start}

def embed_landmarks(nodes, landmarks, dim, dataloc, seed):

	if rank == 0:
		print("\n", "*"*60)
		print("Embedding landmarks for n = %i and L = %i in Dimension %i" %(nodes, landmarks, dim))
		print("*"*60)

	# START LANDMARK EMBEDDING
	if rank == 0: print("\nStarting landmark embedding...")

	hypy_land_time, landmark_rel_error, land_coord, curv = run_single_landmark_embedding(param_list, nodes, land_ids, dataloc, seed)

	print("landmark error for rank %i: %f" %(rank,landmark_rel_error))

	# save all times, errors, embedding coordinates, and curvatures
	C = zeros((nodes, dim))
	C[land_ids,:] = land_coord
	savetxt(os.path.join(dataloc, 'curv_rank' + str(rank) + '.txt'),[curv])
	savetxt(os.path.join(dataloc, 'fval_rank' + str(rank) + '.txt'),[landmark_rel_error]) 
	savetxt(os.path.join(dataloc, 'land_time_rank' + str(rank) + '.txt'),[hypy_land_time])
	save(os.path.join(dataloc, 'land_coord_rank' + str(rank)),C[land_ids,:])

	return hypy_land_time, landmark_rel_error, land_coord, curv

# generate seed from rank
rn = random.RandomState(rank + 13)
seed = rn.randint(1000)

# run landmark embedding
hypy_land_time, landmark_rel_error, land_coord, curv = embed_landmarks(nodes, landmarks, dim, dataloc, seed)




