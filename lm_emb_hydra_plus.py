import lm_emb_hydra_functions as lm_emb_func
import numpy as np 
import matplotlib.pyplot as mpl
import os, pdb, time, pickle, sys


class LHydraPlus:
	'''
	Implements the L-hydra+ (landmarked hyperbolic distance recovery and approximation) method 
	for embedding high-dimensional data points (represented by their distance matrix \code{D}) 
	into low-dimensional hyperbolic space.
	
	Parameters
	----------
	name : Char 
		Name of the network to embed.
	outdir : Char
		Directory for output data.
	nlandmark: Int
		Number of landmark nodes
	nodes_per_run : Int
		Number of nodes per run of nonlandmark embedding
	'''
	
	def __init__(self, name, outdir, nL, nodes_per_run):
		print("Start HydraPlus-Embedding")
		self.name = name
		self.datadir = outdir
		self.nL = nL
		self.nnodes = np.int(np.loadtxt(self.datadir + 'nNodes.txt'))
		self.nl_split = int(np.ceil((self.nnodes-self.nL)/float(nodes_per_run)))
		self.metadata = {"nL":self.nL, "datadir":self.datadir, "nnodes":self.nnodes}
		if outdir != None and os.path.exists(os.path.join(outdir, 'metadata.pkl')):
			self.metadata = self.load_meta("metadata")
						
	def embed_landmarks(self,hdim,mpicc,nrepeat):
		'''
		performs landmark embedding in parallel using MPI
		'''
		MD = self.metadata
		self.metadata['hdim'] = hdim
		print("\nHyperbolic dimension for landmark embedding is %i." %(hdim))
		start_land = time.time()
		landmark_input = lm_emb_func.landmark_preproc(MD['nnodes'], MD['nL'], hdim, MD['datadir'])
		start = time.time()
		self.mpicc = mpicc
		os.system(mpicc + ' --oversubscribe -np ' + str(nrepeat) + ' python3 lm_emb_hydra_plus_landmarks.py' + landmark_input)
		# get best embedding 
		flandopt, land_coord, k, land_time, land_ind = lm_emb_func.gather_landmarks(MD['nnodes'], MD['nL'], hdim, MD['datadir'], nrepeat)
		self.land_coord = land_coord
		self.full_land_coord = self.get_full_coordinates(self.land_coord)
		print("Complete landmark time is ", time.time() - start_land)
		print("Best embedding error is %f" %(flandopt))
		return land_coord, 1.0*k, self.full_land_coord
		
	def embed_nonlandmarks(self, nprocs2):
		'''
		performs nonlandmark embedding using MPI as well
		'''
		MD = self.metadata
		nodes = MD['nnodes']
		landmarks = MD['nL']
		datadir = MD['datadir']
		dim = MD['hdim']
		print("\nHyperbolic dimension for non-landmark embedding is %i." %(dim))
		start_nonland = time.time()
		print("Number of batches of nonlandmarks is %i" %(self.nl_split))
		for split in range(self.nl_split): # nonlandmark splits
			nonland_input = lm_emb_func.nonlandmark_preproc(nodes, landmarks, dim, split, datadir)
			os.system(self.mpicc + ' -np ' + str(nprocs2) + ' python3 lm_emb_hydra_plus_nonlandmarks.py' + nonland_input)
			# gather solutions per split
			lm_emb_func.gather_nonlandmarks_split(nodes, landmarks, dim, datadir, split, nprocs2)
		C, nonland_avg_rel_error, hypy_nonland_time = lm_emb_func.gather_nonlandmarks(nodes, landmarks, dim, datadir, self.nl_split)
		tot_land_time = time.time() - start_nonland
		print("Total nonland time: ", tot_land_time)
		
	def validate(self, savedir, nprocs3):
		'''
		validate and save validation meta data in savedir
		'''
		MD = self.metadata
		nodes = MD['nnodes']
		landmarks = MD['nL']
		datadir = MD['datadir']
		dim = 10
		# run validation test (must create /results directory)
		print("begin validation tests...")
		val_input = lm_emb_func.validation_preproc(nodes, landmarks, dim, nprocs3, datadir, savedir)
		os.system('python3 lm_emb_validation.py' + val_input)
	
	def load_meta(self, name):
	    with open(os.path.join(self.datadir, name + ".pkl"), 'rb') as f:
	    	return pickle.load(f)
			
	def get_full_coordinates(self,coordinates):
		''' 
		Given a 2d array of d-dim hyper coordinates
		return the full d+1-dim coordinate for plotting
		'''
		u_dp1 = np.sqrt(np.sum(coordinates**2,1)+1)
		full_coord = np.hstack((coordinates,np.atleast_2d(u_dp1).T))
		return full_coord
