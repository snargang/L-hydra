import multiprocessing as mp
import sys, time, pdb, os, pickle

from numpy import *
from scipy.optimize import minimize, fmin_l_bfgs_b

'''
Functions used during embedding processes

First section: 
	functions used during pre and post processing
Second section: 
	objective functions for stress minimization
Third section: functions for landmark embedding
''' 

# **********************************
# Pre and post processing functions
# **********************************

def preprocdata(nodes, landmarks, nodes_per_run, dataloc, maxprocs4seed=1000, cleanup=False):
	print("\n", "*"*60)
	print("Pre-processing: splitting D mtx for n = %i and L = %i" %(nodes, landmarks))
	print("*"*60)

	if cleanup: clean(dataloc)
	
	# calculate the number of partitions for the D matrix
	n_split = int(ceil((nodes-landmarks)/float(nodes_per_run)))
	savetxt(dataloc + '/nonland_nsplit.txt', [n_split],fmt='%i')

	print("Splitting nonlandmark matrix")
	preproc(nodes, landmarks, dataloc, n_split, maxprocs4seed)

def landmark_preproc(nodes, landmarks, dim, dataloc):
	input_var = [" -n "," -L "," -d "," -s "]
	input_val = [nodes, landmarks, dim, dataloc]
	input_tot = ""
	for i,var in enumerate(input_var):
		input_tot += var + str(input_val[i])
	return input_tot

def nonlandmark_preproc(nodes, landmarks, dim, split, dataloc):
	input_var = [" -n "," -L "," -d "," -sp ", " -s "]
	input_val = [nodes, landmarks, dim, split, dataloc]
	input_tot = ""
	os.system('mkdir ' + os.path.join(dataloc, '/temp'))
	for i,var in enumerate(input_var):
		input_tot += var + str(input_val[i])
	return input_tot

def validation_preproc(nodes, landmarks, dim, nproc, dataloc, savedir):
	input_var = [" -n "," -L "," -d "," -p ", " -s1 ", " -s2 "]
	input_val = [nodes, landmarks, dim, nproc, dataloc, savedir]
	input_tot = ""
	for i,var in enumerate(input_var):
		input_tot += var + str(input_val[i])
	return input_tot

def save_obj(obj, name, savedir=''):
    with open(savedir + '/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def load_obj(name, savedir='' ):
    with open(savedir + '/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def clean(dataloc):
	print("Cleaning up data folder...")
	os.system('rm ' + os.path.join(dataloc, '*_split_*'))
	os.system('rm ' + os.path.join(dataloc, 'D_land.npy'))
	os.system('rm ' + os.path.join(dataloc, 'D_nonland.npy'))
	os.system('rm ' + os.path.join(dataloc, 'curv.txt'))
	os.system('rm ' + os.path.join(dataloc, 'd_nonland_true.txt'))
	os.system('rm ' + os.path.join(dataloc, '*rank*'))

def load_data_preproc(nodes, landmarks, dataloc):
	# Load full distance matrix for all nodes
	print("loading testing data...")
	D_test_points = loadtxt(os.path.join(dataloc, 'test_point_distances.txt'), dtype='int')
	D_test_pairs = loadtxt(os.path.join(dataloc, 'test_points.txt'), dtype='int')

	# define landmark ids and get distance matrix from only landmarks to nodes
	print("loading land_ids...")
	land_ids = loadtxt(os.path.join(dataloc, 'landmark_ids.txt'), dtype='int32')
	D_land2nodes = load(os.path.join(dataloc, 'dist_mtx_landmarks2nodes.npy'))

	print("finding nonland_ids...")
	ind = ~in1d(range(nodes),land_ids)
	print("length land_ids..." + str(len(land_ids)))
	print("length ind..." + str(len(ind)))
	print("length D_land2nodes..." + str(len(D_land2nodes)))
	D_nonland = D_land2nodes[:,ind]
	nonland_ids = arange(nodes)[ind]

	return D_land2nodes, D_land2nodes[:,land_ids], land_ids, D_nonland, nonland_ids, D_test_pairs, D_test_points

def load_data_main(nodes, landmarks, dataloc=''):
	# Load full distance matrix for all nodes
	D_test_points = loadtxt(os.path.join(dataloc, 'test_point_distances.txt'), dtype='int32')
	D_test_pairs = loadtxt(os.path.join(dataloc, 'test_points.txt'), dtype='int32')

	# define landmark ids and get distance matrix from only landmarks to nodes
	land_ids = loadtxt(os.path.join(dataloc, 'landmark_ids.txt'), dtype='int32')
	D_land = load(os.path.join(dataloc, 'D_land.npy')).astype('int')

	ind = ~in1d(range(nodes),land_ids)
	nonland_ids = arange(nodes)[ind]

	return D_land, land_ids, nonland_ids, D_test_pairs, D_test_points
 
def load_data_4land(nodes, landmarks, dataloc=''):
	# define landmark ids and get distance matrix from only landmarks to nodes
	land_ids = loadtxt(os.path.join(dataloc, 'landmark_ids.txt'), dtype='int32')
	D_land = load(os.path.join(dataloc, 'D_land.npy')).astype('int64')
	ind = ~in1d(range(nodes),land_ids)
	nonland_ids = arange(nodes)[ind]

	return D_land, land_ids, nonland_ids

def load_data_4nonland(nodes, landmarks, dataloc=''):
	# define landmark ids and get distance matrix from only landmarks to nodes
	land_ids = loadtxt(os.path.join(dataloc, 'landmark_ids.txt'), dtype='int32')
	ind = ~in1d(range(nodes),land_ids)
	nonland_ids = arange(nodes)[ind]

	return land_ids, nonland_ids

def split_nonland_mtx(nodes, landmarks, dataloc, n_split):
	node_ids = array_split(arange(nodes - landmarks),n_split)
	# load data and do pre-processing
	D_land2nodes, D_land, land_ids, D_nonland, nonland_ids, D_test_pairs0, D_test_dist0 = load_data_preproc(nodes, landmarks, dataloc)
	save(os.path.join(dataloc, 'D_nonland'), D_nonland)
	save(os.path.join(dataloc, 'D_land'), D_land)
	for ii,nn in enumerate(node_ids):
		print("Splitting %i out of %i" %(ii,len(node_ids)))
		D_nonland_temp = D_nonland[:,nn]
		save(os.path.join(dataloc, 'D_nonland_split_', str(ii).zfill(3)), D_nonland_temp)

	stemp = sum(D_nonland**2,0)
	print(stemp[:10])
	d_nonland_true = sqrt(abs(stemp)) # norm along rows, 
	savetxt(os.path.join(dataloc, 'd_nonland_true.txt'), d_nonland_true)

	return 0.0

def preproc(nodes, landmarks, dataloc, nprocs, maxprocs4seeds):
	node_ids = array_split(arange(nodes - landmarks),nprocs)
	# load data and do pre-processing
	print("Loading data...")
	D_land2nodes, D_land, land_ids, D_nonland, nonland_ids, D_test_pairs0, D_test_dist0 = load_data_preproc(nodes, landmarks, dataloc)
	print("Saving D_land and D_nonland...")
	print("Data types for nonlandmark and landmark matrices: ", D_nonland.dtype, D_land.dtype)
	save(os.path.join(dataloc, 'D_nonland'),D_nonland)
	save(os.path.join(dataloc, 'D_land'), D_land)
	for ii,nn in enumerate(node_ids):
		print("Splitting %i out of %i" %(ii+1,len(node_ids)))
		D_nonland_temp = D_nonland[:,nn]
		save(os.path.join(dataloc, 'D_nonland_split_' + str(ii).zfill(3)), D_nonland_temp)
	ns = 20
	ls = array_split(range(nodes-landmarks),ns)
	# square and sum chunks to avoid dtype errors (< 0 for int8)
	S = []
	for ii,lsi in enumerate(ls):
		print("squaring nonlandmark matrix %i out of %i" %(ii,ns))
		Dtemp = D_nonland[:,lsi].astype('int')
		S.append(sum(Dtemp**2,0))
	stemp = hstack(S)
	print(stemp[stemp < 0])
	d_nonland_true = sqrt(stemp) # norm along rows, 
	savetxt(os.path.join(dataloc, 'd_nonland_true.txt'), d_nonland_true)

	# generate random seed list
	print("generating random seed list...")
	seed_list = random.randint(1,max(maxprocs4seeds,1e4),maxprocs4seeds)
	savetxt(os.path.join(dataloc, 'random_seed_list.txt'), seed_list, fmt='%i')

	return 0.0

# **********************************
# Objective function definitions
# **********************************

def f(X,Y):
	return .5*linalg.norm(X-Y, ord='fro')**2

def df(X,Y):
	return X-Y

def hype_dist(p1,p2,curv):
	k = curv
	c = 1./sqrt(abs(k))
	x = dot(p1,p2)
	usq1 = dot(p1,p1) + 1.0
	usq2 = dot(p2,p2) + 1.0
	udotu = x - sqrt(usq1)*sqrt(usq2)
	acoshudotu = real(arccosh(-udotu + 0j))
	d = acoshudotu * c
	return d

def land_obj(x,**kwargs):
	dim = kwargs['dim']
	D0 = kwargs['D_landmark']
	grad = zeros(len(x))
	points = len(x[:-1])//dim
	k = x[-1]
	c = 1./sqrt(abs(k))
	P = x[:-1].reshape((points,dim))

	X = dot(P,P.T)
	Usq = diag(X) + 1.0
	U = sqrt(Usq)
	UdotU = X - outer(U,U)
	acoshUdotU = real(arccosh(-UdotU + 0j))
	fill_diagonal(acoshUdotU,0.0)

	D = acoshUdotU*c;
	D[isnan(D)] = 0
	dE = df(D,D0)
	err = f(D,D0)/2.0 # since matrix is symmetric

	# compute gradient w.r.t. k
	gradk = - dE * acoshUdotU * .5 * k * abs(k)**-2.5
	grad[-1] = sum(sum(gradk))/2.0 # divide by two since matrix is symmetric

	# compute gradient
	tol = 1e-12
	Atemp = UdotU**2 - 1.0
	Atemp[Atemp<=0] = tol
	A = - 1./sqrt(Atemp) * c
	B = outer(1./U,U)
	C = dE * A
	H1 = C * B
	hsum = sum(H1,1)
	L = len(D0)
	Ci = zeros((len(C),len(C)))
	for i in range(L):
		ui = P[i,:]
		hi = hsum[i]
		fill_diagonal(Ci,C[i])
		g = sum(dot(P.T,Ci),1) - hi*ui
		grad[dim*i:(i+1)*dim] = g

	return err, grad

def nonland_obj(x, node, **kwargs):
	'''
	computes nonlandmark error for an entire nonlandmark
	matrix D0
	'''
	# node is index of D_nonland not D_land2nodes
	dim = kwargs['dim']
	D0 = kwargs['D_nonland']
	L = kwargs['Land_points']
	n_land, dim = L.shape
	nl_id = node
	k = kwargs['k']
	c = 1./sqrt(abs(k))

	X = dot(L,x)
	u = sqrt(dot(x,x) + 1.0)
	V = sqrt(sum(L.T**2,0) + 1.0)
	udotV = outer(u,V) - X
	d = real(arccosh(udotV + 0j)) * c
	d0 = D0[:,nl_id]
	err = f(d,d0)
	derr = df(d,d0)

	# compute gradient
	tol = 1e-12
	Atemp = udotV**2 - 1.0
	Atemp[Atemp<=0] = tol
	A = - 1./sqrt(Atemp) * c
	B = outer(1./u,V)
	C = derr * A
	H1 = C * B
	H1.shape = (n_land,1)
	C.shape = (n_land,1)

	ui = x
	Ui = tile(ui,(n_land,1)).T
	Hi = diag(H1[:,0])
	Ci = diag(C[:,0])
	G = dot(L.T,Ci) - dot(Ui,Hi)
	grad = sum(G,1)

	return err, grad

def nonland_obj_percol(x, dim, L, k, d0):
	'''
	computes nonlandmark error for fixed column d0 of the 
	nonlandmark matrix
	'''
	land_idx = arange(len(L))
	Lnew = L[land_idx,:]
	n_land, dim = Lnew.shape
	c = 1./sqrt(abs(k))

	X = dot(Lnew,x)
	u = sqrt(dot(x,x) + 1.0)
	V = sqrt(sum(Lnew.T**2,0) + 1.0)
	udotV = outer(u,V) - X
	d = real(arccosh(udotV + 0j)) * c
	err = f(d,d0)
	derr = df(d,d0)

	# compute gradient
	tol = 1e-12
	Atemp = udotV**2 - 1.0
	Atemp[Atemp<=0] = tol
	A = - 1./sqrt(Atemp) * c
	B = outer(1./u,V)
	C = derr * A
	H1 = C * B
	H1.shape = (n_land,1)
	C.shape = (n_land,1)

	ui = x
	Ui = tile(ui,(n_land,1)).T
	Hi = diag(H1[:,0])
	Ci = diag(C[:,0])
	G = dot(Lnew.T,Ci) - dot(Ui,Hi)
	grad = sum(G,1)

	return err, grad

def pre_nonland_obj_percol(x, dim, L, k, rL_in_land_idx, d0):
	'''
	computes nonlandmark error for fixed column d0 of the 
	nonlandmark matrix
	'''
	Lnew = L[rL_in_land_idx,:]
	d0 = d0[rL_in_land_idx]
	n_land, dim = Lnew.shape
	c = 1./sqrt(abs(k))

	X = dot(Lnew,x)
	u = sqrt(dot(x,x) + 1.0)
	V = sqrt(sum(Lnew.T**2,0) + 1.0)
	udotV = outer(u,V) - X
	d = real(arccosh(udotV + 0j)) * c
	err = f(d,d0)
	derr = df(d,d0)

	# compute gradient
	tol = 1e-12
	Atemp = udotV**2 - 1.0
	Atemp[Atemp<=0] = tol
	A = - 1./sqrt(Atemp) * c
	B = outer(1./u,V)
	C = derr * A
	H1 = C * B
	H1.shape = (n_land,1)
	C.shape = (n_land,1)

	ui = x
	Ui = tile(ui,(n_land,1)).T
	Hi = diag(H1[:,0])
	Ci = diag(C[:,0])
	G = dot(Lnew.T,Ci) - dot(Ui,Hi)
	grad = sum(G,1)

	return err, grad

# **********************************
# Landmark Embedding functions
# **********************************

def embed_prelandmarks(xstart,factr,param_list):
	def f(x):
		return land_obj(x,**param_list)
	bnds = [ (None, None) for i in range(len(xstart))]
	x,f,d = fmin_l_bfgs_b(f,xstart, bounds=bnds, factr=factr)
	return x,f,d
	
def embed_pre_nonlandmarks(xstart,node,param_list2):
	def g(x):
		return nonland_obj(x,node,**param_list2)
	bnds = [ (None, None) for i in range(len(xstart))]
	x,f,d = fmin_l_bfgs_b(g,xstart, bounds=bnds, factr=1e12)
	return x,f,d

def pre_embedding(dim, D_land, landmarks0, xstart, seed=133):
	D0 = D_land
	x0 = xstart
	nL = len(D0)
	rn = random.RandomState(seed)
	if nL <= 32:
		x = append(rn.rand(1), x0.flatten())
		param_list = {'dim': dim, 'D_landmark' : D_land}
		x1,_,_ = embed_prelandmarks(x,1e10,param_list)
		return x1
	else:
		# find nearest power of two and embed
		exp = floor(log2(nL))
		if abs(exp - log2(nL)) == 0:
			# if nL is a perfect power of 2, then use lowest exp
			rL = int(2**(exp-1))
		else:
			rL = int(2**(exp))
		land_ids_new = array(sort(rn.permutation(nL)[:rL])).astype('int')
		D_land2nodes_new = D0[land_ids_new,:]
		D_land_new = D_land2nodes_new[:,land_ids_new]
		xstart_new = x0[land_ids_new]
		# embed new set of sub-landmarks
		param_list_new = {'dim': dim, 'D_landmark' : D_land_new, 'x_start0' : xstart_new}
		x_land_soln = pre_embedding(dim, D_land_new,landmarks0,xstart_new)
		# solve exactly (comment out for faster runs)
		if rL > 32 and rL <= int(landmarks0/2):
			x1temp,_,_ = embed_prelandmarks(x_land_soln,1e10,param_list_new)
			x_land_soln = x1temp
		L_new = x_land_soln[:-1].reshape((rL,dim))
		curv_new = x_land_soln[-1]
		# embed nonlandmarks
		nonland_ids_new = delete(arange(nL),land_ids_new)
		D_nonland_new = D_land2nodes_new[:,nonland_ids_new]
		nl_ids = range(len(nonland_ids_new))
		nonland_xmin = []
		for nl_id in nl_ids:
			param_list2_new = {'dim': dim, 'Land_points': L_new, 'D_nonland': D_nonland_new, 'k': curv_new}
			xstart2_0 = x0[nonland_ids_new[nl_id]]
			x2,_,_ = embed_pre_nonlandmarks(xstart2_0,nl_id,param_list2_new)
			nonland_xmin.append(x2)
		# fill in embedded points
		C0 = zeros((nL,dim))
		for ii,id in enumerate(land_ids_new):
			C0[id,:] = L_new[ii,:]
		for ii,xmin_nl in enumerate(nonland_xmin):
			C0[nonland_ids_new[ii],:] = xmin_nl
		return append(C0.flatten(),curv_new)

def pre_embed(s, param_list):
	''' 
	Call pre_embedding 
	'''
	dim = param_list['dim']
	D_land = param_list['D_landmark']
	landmarks0 = param_list['landmarks0']
	xstart = param_list['x_start0']
	x = pre_embedding(dim,D_land,landmarks0,xstart,seed=s)
	return land_obj(x,**param_list)[0], x

def embed_landmarks(xstart,param_list,factr=1e10):
	def f(x):
		return land_obj(x,**param_list)
	bnds = [ (None, None) for i in range(len(xstart))]
	x,f,d = fmin_l_bfgs_b(f,xstart,\
							bounds=bnds,\
							factr=factr)
	return f,x

def F(args):
	'''
	Main embedding algorithm (run for multiple seeds)
	'''
	seed = args[0]
	param_list = args[1]
	f0,x0 = pre_embed(seed, param_list)
	return embed_landmarks(x0,param_list)

def run_landmark_embedding(param_list, np, nodes, land_ids, dataloc):
	start = time.time()
	rn_test = random.RandomState(163)
	inputs1 = [[rn_test.randint(1e3),param_list] for i in range(np)]
	pool = mp.Pool(processes=np)
	solns = pool.map(F, inputs1)
	fvals = [output[0] for output in solns]
	min_index = argmin(fvals)
	fval,xsoln = solns[min_index]
	land_error = fval
	dim = param_list['dim']
	D_land = param_list['D_landmark']
	landmarks = len(D_land)
	Xsoln = xsoln[:-1].reshape(landmarks,dim)
	csoln = xsoln[-1]
	landmark_rel_error = sqrt(2*land_error)/linalg.norm(D_land,ord='fro')
	print("Total landmark time is %3.5f seconds" %(time.time()-start))
	C = zeros((nodes, dim))
	C[land_ids,:] = Xsoln
	savetxt(os.path.join(dataloc, 'curv.txt'),[csoln])
	savetxt(os.path.join(dataloc, 'fval.txt'),[fval])
	save(os.path.join(dataloc, 'C'),C)
	save(os.path.join(dataloc, 'L_points'),C[land_ids,:])

	hypy_land_time = time.time() - start
	savetxt(os.path.join(dataloc, 'hypy_land_time.txt'),[hypy_land_time])
	savetxt(os.path.join(dataloc, 'landmark_rel_error.txt'),[landmark_rel_error])

	L_points = C[land_ids]
	curv = csoln
	pool.close()

	return hypy_land_time, landmark_rel_error, L_points, curv, C

def run_single_landmark_embedding(param_list, nodes, land_ids, dataloc, seed):
	start = time.time()

	fval, xsoln = F((seed,param_list))
	land_error = fval
	dim = param_list['dim']
	D_land = param_list['D_landmark']
	landmarks = len(D_land)
	Xsoln = xsoln[:-1].reshape(landmarks,dim)
	csoln = xsoln[-1]
	landmark_rel_error = sqrt(2*land_error)/linalg.norm(D_land,ord='fro')
	hypy_land_time = time.time() - start

	return hypy_land_time, landmark_rel_error, Xsoln, csoln

def gather_landmarks(nodes, landmarks, dim, dataloc, nprocs):
	'''
	Get optimal embedding out of different ranks
	'''
	fvals = zeros(nprocs)
	land_times = zeros(nprocs)
	for r in range(nprocs):
		fvals[r] = loadtxt(os.path.join(dataloc, 'fval_rank' + str(r) + '.txt'))
		land_times[r] = loadtxt(os.path.join(dataloc, 'land_time_rank' + str(r) + '.txt'))
	opt_r = argmin(fvals) # optimal (min) error
	fopt = fvals[opt_r] # relative error (not actual objective fun)
	topt = amax(land_times)
	copt = loadtxt(os.path.join(dataloc, 'curv_rank' + str(opt_r) + '.txt')) # optimal curvature
	land_coord = load(os.path.join(dataloc, 'land_coord_rank' + str(opt_r) + '.npy'))
	savetxt(os.path.join(dataloc, 'landmark_rel_error.txt'), [fopt])

	# save optimal landmark coordinates and curvature
	save(os.path.join(dataloc, 'land_coord_opt'), land_coord)
	savetxt(os.path.join(dataloc, 'copt.txt'), [copt])
	savetxt(os.path.join(dataloc, 'hypy_curv.txt'), [copt])
	savetxt(os.path.join(dataloc, 'fopt.txt'), [fopt])
	savetxt(os.path.join(dataloc, 'hypy_land_time.txt'), [amax(land_times)])
	return fopt, land_coord, copt, topt, opt_r

def gather_nonlandmarks_split(nodes, landmarks, dim, dataloc, split, nprocs):
	F2vals = []
	X2solns = []
	NLtimes = []
	for r in range(nprocs):
		ftemp = load(os.path.join(dataloc, 'temp', 'f2s_split' + str(split) + '_rank' + str(r) + '.npy'))
		xtemp = load(os.path.join(dataloc, 'temp', 'x2s_split' + str(split) + '_rank' + str(r) + '.npy'))
		timetemp = loadtxt(os.path.join(dataloc, 'temp', 'nonland_time_split' + str(split) + '_rank' + str(r) + '.txt'))
		F2vals.append(ftemp)
		X2solns.append(xtemp)
		NLtimes.append(float(timetemp))
	f2s = hstack(F2vals).flatten()
	x2s = vstack(X2solns)
	hypy_nonland_time = amax(NLtimes)

	save(os.path.join(dataloc, 'f2vals_split_' + str(split)), f2s)
	save(os.path.join(dataloc, 'x2solns_split_' + str(split)), x2s)
	save(os.path.join(dataloc, 'nonland_time_split_' + str(split)), hypy_nonland_time)

	return f2s, x2s, hypy_nonland_time

def gather_nonlandmarks(nodes, landmarks, dim, dataloc, n_split):
	F2vals = []
	X2solns = []
	NLtimes = []
	for nn in range(n_split):
		ftemp = load(os.path.join(dataloc, 'f2vals_split_' + str(nn) + '.npy'))
		xtemp = load(os.path.join(dataloc, 'x2solns_split_' + str(nn) + '.npy'))
		timetemp = load(os.path.join(dataloc, 'nonland_time_split_' + str(nn) + '.npy'))
		F2vals.append(ftemp)
		X2solns.append(xtemp)
		NLtimes.append(float(timetemp))

	f2s = hstack(F2vals).flatten()
	x2s = vstack(X2solns)
	hypy_nonland_time = sum(NLtimes)

	d_nonland_true = loadtxt(os.path.join(dataloc, 'd_nonland_true.txt'))
	nonland_avg_rel_error = mean(sqrt(2*f2s).flatten()/d_nonland_true)

	print("Total nonlandmark time is %3.5f seconds" %hypy_nonland_time)
	
	# add nonland coordinates to main embedding
	land_ids = loadtxt(os.path.join(dataloc, 'landmark_ids.txt')).astype('int')
	nonland_ids = loadtxt(os.path.join(dataloc, 'nonlandmark_ids.txt')).astype('int')
	land_coord = load(os.path.join(dataloc, 'land_coord_opt.npy'))
	C = zeros((nodes, dim))
	C[land_ids,:] = land_coord
	C[nonland_ids,:] = x2s
	save(os.path.join(dataloc, 'hypy_coord'), C)
	savetxt(os.path.join(dataloc, 'hypy_nonland_time.txt'), [hypy_nonland_time])
	savetxt(os.path.join(dataloc, 'nonland_avg_rel_error.txt'), [nonland_avg_rel_error])

	print("Non-landmark error: ", nonland_avg_rel_error)
	return C, nonland_avg_rel_error, hypy_nonland_time

def cleanup_at_end(dataloc):
	print("Cleaning up data folder...")
	os.system('rm ' + os.path.join(dataloc, 'temp/*.*'))
	os.system('rm ' + os.path.join(dataloc, '*opt*'))
	os.system('rm ' + os.path.join(dataloc, 'd_nonland_true.txt'))
	os.system('rm ' + os.path.join(dataloc, '*split*'))
	os.system('rm ' + os.path.join(dataloc, '*rank*'))