import multiprocessing as mp
import sys, time, pdb, os, argparse

from lm_emb_hydra_functions import *
from numpy import *
from scipy.optimize import minimize, fmin_l_bfgs_b
from mpi4py import MPI # must use mpirun-openmpi-clang38

'''
Parallel nonlandmark embedding
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
parser.add_argument("-sp", type=int, help="choose nonlandmark split (int)")
parser.add_argument("-s", type=str, help="source location of datafiles (str)")

# set args to variables
args = parser.parse_args()
nodes = args.n
landmarks = args.L
dim = args.d
nn = args.sp
dataloc = args.s
n_split = loadtxt(os.path.join(dataloc, 'nonland_nsplit.txt')) # total number of 
nprocs2 = size

if rank == 0:
	print("\n", "*"*60)
	print("Embedding for n = %i and L = %i in Dimension %i" %(nodes, landmarks, dim))
	print("*"*60)

land_ids, nonland_ids = load_data_4nonland(nodes, landmarks, dataloc)	

if rank == 0: print("\nStarting nonlandmark embedding...")

if rank == 0:
	print("Loading data on rank = 0...")
	D0 = load(os.path.join(dataloc, 'D_nonland_split_' + str(nn).zfill(3) + '.npy')).astype('int32')
	Xstart0 = load(os.path.join(dataloc, 'xstart_split_' + str(nn).zfill(3) + '.npy')).astype('int32')
	L_points = load(os.path.join(dataloc, 'land_coord_opt.npy'))
	land_err_temp_hypy = loadtxt(os.path.join(dataloc, 'fopt.txt'))
	csoln = loadtxt(os.path.join(dataloc, 'copt.txt'))
	curv = float(csoln)
else:
	D0 = None
	Xstart0 = None
	L_points = None
	land_err_temp_hypy = None
	csoln = None
	curv = None

D0 = comm.bcast(D0,root=0)
Xstart0 = comm.bcast(Xstart0,root=0)
L_points = comm.bcast(L_points,root=0)
land_err_temp_hypy = comm.bcast(land_err_temp_hypy,root=0)
csoln = comm.bcast(csoln,root=0)
curv = comm.bcast(curv,root=0)

def embed_nonlandmark0(xstart,dim,L,k,rL_idx,d0,factr=1e12):
	'''
	embed all nonlandmarks relative to landmarks using random start
	ignore rL_idx
	'''
	def g(x):
		return pre_nonland_obj_percol(x,dim,L,k,rL_idx,d0)
	bnds = [ (None, None) for i in range(len(xstart))]
	x,f,d = fmin_l_bfgs_b(g,xstart,\
							bounds=bnds, \
							factr=factr)
	return f,x,d
	
def embed_nonlandmark_recursive(xstart,dim,L,k,rL_idxs,d0,factr=1e12):
	x = xstart.copy()
	factrs = [1e12 for r in range(len(rL_idxs))]
	factrs[-1] = 1e13
	for rr in range(len(rL_idxs)):
		f,x,d = embed_nonlandmark0(x,dim,L,k,rL_idxs[rr],d0,factr=factrs[rr])
	return f,x,d

def F2(args):
	''' 
	Main nonlandmark embedding 
	'''
	seed = args[0]
	dim = args[1]
	L = args[2]
	k = args[3]
	rL = args[4]
	idx = args[5]
	D0 = args[6]
	node_ids = args[7] # list of nodes
	X0 = args[8]
	n_nodes = len(node_ids)
	X = []
	Fval = []
	rn2 = random.RandomState(seed)
	disp_every = .25
	niter = []
	funcalls = []
	rLs = [100*2**i for i in range(int(log2(landmarks/100))+1)]
	rL_idx = [sort(rn2.permutation(land_ids.copy())[:rL]) for rL in rLs]
	rL_in_land_idx = [in1d(land_ids,rL) for rL in rL_idx]
	for i in range(n_nodes):
		if idx == 0:
			if i % int(n_nodes*disp_every) == 0: print(" %3i percent done" %(around(100*i/float(n_nodes))))
		x0 = X0[i,:]
		ftemp, x, d = embed_nonlandmark_recursive(x0,dim,L,k,rL_in_land_idx,D0[:,node_ids[i]])
		X.append(x)
		Fval.append(ftemp)
		niter.append(d['nit'])
		funcalls.append(d['funcalls'])
	return Fval, array(X)

rn = random.RandomState(131)

start = time.time()
if rank == 0: print("starting split %i out of %i" %(nn+1,n_split))
# load split nonland matrix
_,n_nonland_temp = D0.shape

node_ids = array_split(range(n_nonland_temp),nprocs2)

# ready inputs for split
seeds = [rn.randint(1e3) for jj in range(nprocs2)]
nonland_rL = 100 # default
inputs2 = [[seeds[i],dim,L_points,curv,nonland_rL,i,D0,node_ids[i],Xstart0] for i in range(len(node_ids))]
l = [len(n) for n in node_ids]
if rank == 0: print("Each proc gets %i nodes..." %(int(mean(l))))

start0 = time.time()
solns2 = F2(inputs2[rank])
end0 = time.time()

f2vals = solns2[0]
x2solns = solns2[1]
save(os.path.join(dataloc, 'temp', 'x2s_split' + str(nn) + '_rank' + str(rank)),x2solns)
save(os.path.join(dataloc, 'temp', 'f2s_split' + str(nn) + '_rank' + str(rank)),f2vals)

end = time.time()
hypy_nonland_time = end - start
if rank == 0: print("**Parallel split took %.2f seconds" %(hypy_nonland_time))
savetxt(os.path.join(dataloc, 'temp', 'nonland_time_split' + str(nn) + '_rank' + str(rank) + '.txt'),[hypy_nonland_time])

