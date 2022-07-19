import multiprocessing as mp
import sys, time, pdb, os, argparse

from lm_emb_hydra_functions import *
from numpy import *
from scipy.optimize import minimize, fmin_l_bfgs_b

'''
Validate embedding results
'''

# Define static variables with argparse
parser = argparse.ArgumentParser()
parser.add_argument("-n", type=int, help="number of nodes in graph (int)")
parser.add_argument("-L", type=int, help="number of landmark nodes in graph (int)")
parser.add_argument("-d", type=int, help="dimension of embedding (int)")
parser.add_argument("-p", type=int, help="processors for multiprocessing library (int)")
parser.add_argument("-s1", type=str, help="source location of datafiles (str)")
parser.add_argument("-s2", type=str, help="where to save datafiles (str)")

# set args to variables
args = parser.parse_args()
nodes = args.n
landmarks = args.L
dim = args.d
nprocs = args.p
dataloc = args.s1
savedir = args.s2

print("\n", "*"*60)
print("Embedding for n = %i and L = %i in Dimension %i" %(nodes, landmarks, dim))
print("*"*60)

print("Loading data...")
D_land, land_ids, nonland_ids, D_test_pairs0, D_test_dist0 = load_data_main(nodes, landmarks, dataloc)
param_list = {'dim': dim, 'D_landmark' : D_land}

# Begin Testing
Ctest = load(os.path.join(dataloc, 'hypy_coord.npy'))
land_time = float(loadtxt(os.path.join(dataloc, 'hypy_land_time.txt')))
nonland_time = float(loadtxt(os.path.join(dataloc, 'hypy_nonland_time.txt')))
csoln = float(loadtxt(os.path.join(dataloc, 'hypy_curv.txt')))

nonland_avg_rel_error = float(loadtxt(os.path.join(dataloc, 'nonland_avg_rel_error.txt')))
landmark_rel_error = float(loadtxt(os.path.join(dataloc, 'landmark_rel_error.txt')))

land_soln = Ctest[land_ids]
nonland_soln = Ctest[nonland_ids]
curv_soln = csoln
xsoln_land = append(land_soln.flatten(),curv_soln)

def test_points(ids):
	E = []
	Ernd = []
	for ii in ids:
		p1_id, p2_id = D_test_pairs0[ii]
		d_true = D_test_dist0[ii]
		curv = csoln
		p1 = Ctest[p1_id]
		p2 = Ctest[p2_id]
		d_est = hype_dist(p1,p2,curv)
		err = abs(d_est - d_true)
		err2 = abs(around(d_est) - d_true)
		E.append(err)
		Ernd.append(err2)
	return array(E), array(Ernd)

# test pairs
print("\nValidating...")
start_val = time.time()
test_split2 = array_split(range(len(D_test_dist0)),nprocs)
pool2 = mp.Pool(processes=nprocs) # must start new pool to 
test_vals2_0 = pool2.map(test_points,test_split2)
test_vals2 = [array(soln[0]).reshape(len(soln[0]),1) for soln in test_vals2_0]
test_vals2rnd = [array(soln[1]).reshape(len(soln[1]),1) for soln in test_vals2_0]
test_err = vstack(test_vals2)
test_rel_err = linalg.norm(test_err)/linalg.norm(D_test_dist0)
test_err_rnd = vstack(test_vals2rnd)
test_rel_err_rnd = linalg.norm(test_err_rnd)/linalg.norm(D_test_dist0)
print("  Validation error:           ", test_rel_err)
print("  Validation error w/ round.: ", test_rel_err_rnd)
print("  Total time for validtion is ", time.time() - start_val)
print("-" * 40)
pool2.close()

# save metadat
info = {}
algo = 'hypy'
info['algorithm'] = algo
info['nodes'] = nodes
info['landmarks'] = landmarks
info['nonlandmarks'] = nodes - landmarks
info['dim'] = dim
info['curvature'] = curv_soln
info['embedding coordinates'] = Ctest
info['landmark error'] = landmark_rel_error
info['nonlandmark error'] = nonland_avg_rel_error
info['test error'] = test_rel_err
info['test error with rounding'] = test_rel_err_rnd
info['total time for landmarks'] = land_time
info['total time for nonlandmarks'] = nonland_time
info['total time'] = land_time + nonland_time
# save info
file_prefix = "n" + str(nodes) + "_L" + str(landmarks) + '_dim' + str(dim).zfill(2)
file_name = algo + '_' + file_prefix
save_obj(info,file_name,savedir)
clean(dataloc)