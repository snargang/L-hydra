import multiprocessing as mp
import pandas as pd
import os, snap, time, pdb, sys, argparse

from numpy import *

def preproc_graph(filename):
	''' 
	Get connected graph 
	'''
	print("Working on %s \n" %filename)
	print("Generating graph from edge list")
	# load edge list into snap
	Graph0 = snap.LoadEdgeList(snap.PUNGraph, filename, 0, 1, '\t')
	# get edges
	V0 = Graph0.GetNodes()
	# delete zero degree nodes
	snap.DelZeroDegNodes(Graph0)
	print("Done generating graph!")

	# get max weakly connected component
	print( "Generating connected graph") 
	Graph = snap.GetMxWcc(Graph0)
	V = Graph.GetNodes()
	E = Graph.GetEdges()
	print("Done generating graph with V = %i, E = %i!, V0 = %i" %(V, E, V0))

	# get nodes included in weakly connected graph (which could be 
	# a proper subset of original set)
	# Find one edge in graph and find all connected nodes
	for EI in Graph.Edges():
		conn_node = EI.GetSrcNId() # start with one edge
		break # only need one edge since connected
	CnCom = snap.TIntV()
	snap.GetNodeWcc(Graph, conn_node, CnCom)
	conn_node_ids = sort(array([node for node in CnCom]))

	return Graph, conn_node_ids, V, E, V0

def weighted_values(values, probabilities, size, scale=1):
	'''
	without replacement
	'''
	size_scale = scale*size
	bins = add.accumulate(probabilities)
	v = values[digitize(rn.random_sample(size_scale), bins)]
	v = unique(v)
	if len(v) < size:
		return weighted_values(values, probabilities,size,2*scale)
		return v
	else:
		return v[:size]

def get_landmarks_ids(Graph,nL,savedir):
	''' 
	Choose landmarks based on weighted distribution 
	
	get node degree for each node
	node ids are not in consecutive order
	'''
	print("Getting Nodal Degrees...")
	OutDegV = snap.TIntPrV()
	snap.GetNodeOutDegV(Graph, OutDegV)
	node_degree = zeros((V,2))
	for i,item in enumerate(OutDegV):
	    node_degree[i,1] = item.GetVal2()
	    node_degree[i,0] = item.GetVal1()
	node_degree = node_degree[node_degree[:,0].argsort()]
	node_degree2 = node_degree[node_degree[:,1].argsort()]
	node_degree = node_degree.astype(int)
	node_degree2 = flipud(node_degree2.astype(int))
	node_list = sort(node_degree[:,0]).astype(int)
	
	# sample from nodal degree
	node_deg = node_degree[:,1].copy()
	node_deg[node_deg <= 2] = 0 # set probab to zero for deg <= n
	probabilities = 1.0*node_deg/sum(node_deg)

	Llist = []
	land_idx = weighted_values(node_degree[:,0],probabilities,1)
	Llist.append(land_idx[0])
	lcount = 1
	check_random_n = min(16,nL)
	while lcount < nL:
		# sample from degree distribution
		curr_land_idx = weighted_values(node_degree[:,0],probabilities,1)[0]
		# check if sample is not in current list
		if curr_land_idx not in Llist:
			# compute distances to current landmarks
			Llist.append(curr_land_idx)
			lcount += 1
			print(lcount, " landmarks so far...")
	land_ids0 = sort(node_degree[Llist][:,0])[:nL]
	print(node_degree[Llist][:,1])

	# create directory
	print("Creating directory for input data...")
	os.system('mkdir ' + savedir)

	return land_ids0, node_list

def save_id_data(node_list,land_ids0, savedir):
	'''
	map land ids for V0 to node lists on V
	'''
	id_in = in1d(node_list, land_ids0)
	land_ids = where(id_in)[0]
	id_notin = ~id_in
	nonland_ids = where(id_notin)[0]
	savetxt(os.path.join(savedir, 'landmark_ids.txt'),land_ids,fmt='%i')
	savetxt(os.path.join(savedir, 'nonlandmark_ids.txt'),nonland_ids,fmt='%i')
	savetxt(os.path.join(savedir, 'part_id0.ord'),nonland_ids,fmt='%i') # for rigel only
	part_id = array([0,len(nonland_ids)]); part_id.shape = (1,2)
	savetxt(os.path.join(savedir, 'part_id.num'),part_id,fmt='%i') # for rigel only

	return land_ids, nonland_ids


def split(seq, procs):
	''' 
	for parallel proccessing of re-labeling
	seq is a list of input parameters
	procs are the number of processors
	'''
	avg = len(seq) / float(procs)
	out = []
	last = 0.0
	while last < len(seq):
		out.append(seq[int(last):int(last + avg)])
		last += avg
	return out

def get_land_D_mtx(land_ids_temp):
	print("Getting landmark -> node distance matrix...")
	L2n = zeros((len(land_ids_temp),V0),dtype='int8')
	temp = zeros(V0,dtype='int8')
	for i,l in enumerate(land_ids_temp):
		text_to_append = "getting sssp for node %i out of %i" %(i+1,len(land_ids_temp))
		print("getting sssp for node %i out of %i" %(i+1,len(land_ids_temp)))
		os.system("echo '" + text_to_append + "\n' >> percent_done_4landmarks.txt")
		sys.stdout.flush()
		NIdToDistH = snap.TIntH()
		shortestPath = snap.GetShortPath(Graph, int(l), NIdToDistH)
		for item in NIdToDistH:
			L2n[i,item-1] = int8(NIdToDistH[item])
			temp[item-1] = NIdToDistH[item]
		save(os.path.join(savedir, 'L2n_i' + str(i).zfill(3)), temp)
	L2n = L2n[:,conn_node_ids-1]
	return L2n

def get_land_D_mtx_np(land_ids0,savedir,np):
	print("Getting Landmark D matrix...")
	os.system('touch percent_done_4landmarks.txt')
	# get landmarks in parallel
	start = time.time()
	land_split = split(land_ids0,np)
	pool = mp.Pool(processes=np)
	L2n_all = pool.map(get_land_D_mtx,land_split)
	end = time.time()
	# concatenate all L2n's
	L2n = vstack(L2n_all).astype('int8')
	_,n_nodes = L2n.shape
	start = time.time()
	print("distances less than 3 hops: ", sum(L2n <= 2))
	print('total landmark time is %.2f seconds' %(end - start))
	print("Savings landmark distance mtx to binary...")
	save(os.path.join(savedir, 'dist_mtx_landmarks2nodes'),L2n)	
	print(time.time() - start)
	print("Savings landmark distance mtx to text using pandas csv func...")
	filename = os.path.join(savedir, 'dist_mtx_landmarks2nodes.txt')
	os.system('touch ' + filename)
	L2n_pandas = pd.DataFrame(L2n,dtype='int8')
	L2n_pandas.astype('int8')
	print(time.time() - start)
	return L2n,L2n_pandas

def get_test_split(nonland_ids, node_list, n_test_points = 100000):
	'''
	get shortest path for random test points
	'''
	print("Generating random nonlandmarks...")
	n_nonland = len(nonland_ids.copy())
	n_random = min(n_test_points, n_nonland) # choose at most 20k points
	ri = rn.choice(n_nonland, n_random)
	rj = rn.choice(n_nonland, n_random)
	R = array([nonland_ids[ri],nonland_ids[rj]]).T
	R = vstack([tuple(row) for row in R]).astype(int)
	R0 = array([node_list[R[:,0]],node_list[R[:,1]]]).T
	splitR0 = split(R0,np)

	return R, R0, splitR0

def ssspfun(R0):
	paths = []
	start0 = time.time()
	for i,r in enumerate(R0):
		if i % 10 == 0: 
			text_to_append = "%.2f percent done, node %i out of %i" %(double(i)/len(R0), i, len(R0))
			print("%.2f percent done, node %i out of %i" %(double(i)/len(R0), i, len(R0)))
			print("%.2f seconds have elapsed so far..." %(time.time() - start0))
			os.system("echo '" + text_to_append + "\n' >> percent_done_4testing.txt")      
		sp = snap.GetShortPath(Graph, r[0].item(), r[1].item())
		paths.append(sp)
	print("Done!")
	paths = array(paths)
	return paths

def ssspfun_np(savedir,np):
	print("Starting parallel test data evaluation...")
	os.system('touch percent_done_4testing.txt')
	start = time.time()
	pool = mp.Pool(processes=np)
	temp_dist_list = pool.map(ssspfun,splitR0)
	end = time.time()
	print("Total time is %.2f seconds" %(end - start))
	paths = hstack(temp_dist_list)
	print("Savings test data...")
	savetxt(os.path.join(savedir, 'test_points.txt'),R,fmt='%i')
	savetxt(os.path.join(savedir, 'test_point_distances.txt'),paths,fmt='%i')

	return paths

print("Begin getting data for L-hydra+")

#######################################################################
parser = argparse.ArgumentParser()
parser.add_argument("-filename", type=str, help="name of original snap file (string)")
parser.add_argument("-location", type=str, help="source location of snap data file (string)")
parser.add_argument("-datadir", type=str, help="output location")
parser.add_argument("-nprocs", type=int, help="number of processors")
parser.add_argument("-nL", type=int, help="number of landmarks")
parser.add_argument("-nval", type=int, help="number of validation nodes")

args = parser.parse_args()
#######################################################################

# Choose random seed for selecting landmarks and nonlandmark testing pairs
rn = random.RandomState(233)

# generate undirected graph with power low exponent
nL = args.nL
nval = args.nval
np = args.nprocs

graphsrc = args.filename
dirloc = args.location
datafile = dirloc + graphsrc
filename4edges = dirloc + graphsrc # remapped edge list

Graph, conn_node_ids, V, E, V0 = preproc_graph(filename4edges)
savedir = args.datadir

# get landmark and nonlmark id's
land_ids0, node_list = get_landmarks_ids(Graph,nL,savedir)
land_ids, nonland_ids = save_id_data(node_list, land_ids0, savedir)

# get shortest paths to landmarks only
L2n, L2n_pandas = get_land_D_mtx_np(land_ids0,savedir,np)

# get test pairs
R, R0, splitR0 = get_test_split(nonland_ids, node_list, nval)
Rdist = ssspfun_np(savedir,np)
