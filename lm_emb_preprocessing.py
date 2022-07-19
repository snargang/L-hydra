import lm_emb_hydra_functions as lm_emb_func
import numpy as np 
import numpy.random as random
import matplotlib.pyplot as mpl
import multiprocessing as mp
import pandas as pd
import os, pdb, time, pickle, sys, argparse, snap

class Preprocessing:

	def __init__(self, name, indir, outdir, nprocs, nL, nVal):
		print("Start Preprocessing")
		self.name = name # network name
		self.location = indir # directory for input data (contains edge list from snap)
		self.datadir = outdir # directory for output data
		self.nprocs = nprocs # number of processes for parallel embedding
		self.nL = nL # number of landmarks
		self.nVal = nVal # number of validation pairs
	
	def ReMapNodes(self, filename):
		'''
		Remap the nodes given an edge list file
		into consecutive numbers. This is necessary
		for SNAP data sets where indices could be greater
		than the actual number of nodes.
		Requires pandas and multiprocessing package.
		'''
		graphsrc = filename # original file from SNAP
		nprocs = self.nprocs # processes
		dirloc = self.location
		datafile = dirloc + graphsrc
		newfilename = dirloc + graphsrc + '_remapped' # remapped edge list

		edge_list, nodes, n_edges = self.load_graph_data(datafile)

		# create dictionary map from old nodes to new nodes
		global D
		D = {}
		for i,n in enumerate(nodes):
			D[n] = i

		# remap function requires dictionary D as a global
		start = time.time()
		new_edge_list = self.remap_edge_list(edge_list, nprocs)
		end = time.time()
		print("Total time for remapping is %f seconds" %(end-start))

		# save data
		print('writing to text file...')
		np.savetxt(newfilename,new_edge_list,fmt='%i',delimiter='\t')

		self.edge_list_file = filename + '_remapped'
		self.edge_list_location = self.location
		
	def GenAdjMat(self, filename):
		'''
		Compute the landmarks given the edge list along
		with distances from nodes to each landmark
		'''
		self.metadata = {"nL":self.nL, "nVal":self.nVal, "datadir":self.datadir}

		os.system('python lm_emb_preprocessing_gen_adj.py' + ' -filename=' + filename  \
											+ ' -location=' + self.location  \
											+ ' -datadir=' + self.datadir   \
											+ ' -nprocs=' + str(self.nprocs) \
											+ ' -nL=' + str(self.nL) \
											+ ' -nval=' + str(self.nVal))
											
		self.nnodes = np.int(np.loadtxt(self.datadir + 'nNodes.txt'))
		self.metadata["nnodes"] = self.nnodes
		self.save_meta(self.metadata, "metadata")
		
	def set_main_env(self, maxprocs4seed, split, nodes_per_run):
		'''
		Split the nonlandmark embedding into serial chunks

		If split = True, then user has to define nodes_per_run. Split
		defines how many serial chunks to perform the nonlandmark
		embedding. Each chunk is then performed in parallel. E.g., 
		if nnodes = 100 and nodes per run is 2, then the first 50
		nodes are embedding in parallel, followed by the next 50.

		Most data is written to the datadir so that there is minimal
		sharing of data and each function can be run at different times.  
		'''
		# set hyperbolic dimension and processors for nonland embedding
		nodes = self.metadata['nnodes']
		landmarks = self.metadata['nL']
		datadir = self.metadata['datadir']
		self._nprocs1 = maxprocs4seed # parallel landmark procs

		if split==True:
			self.nl_split = int(np.ceil((nodes-landmarks)/float(nodes_per_run)))
			# preprocess data for landmark and nonlandmark embedding
			os.system('mkdir ' + os.path.join(datadir, 'temp'))
			lm_emb_func.preprocdata(nodes, landmarks, nodes_per_run, datadir, maxprocs4seed=100*self._nprocs1, cleanup=True)
		
	def save_meta(self, obj, name):
	    with open(os.path.join(self.datadir, name + ".pkl"), 'wb') as f:
	    	pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

	def load_meta(self, name):
	    with open(os.path.join(self.datadir, name + ".pkl"), 'rb') as f:
	    	return pickle.load(f)
	
	def split(self, seq, procs):
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
    	
	def load_graph_data(self, filename):
		print('Loading data...')
		e0 = pd.read_csv(filename,sep='\t',header=None,skiprows=4,dtype='int32')
		e1 = np.array(e0[0],dtype='int32')   
		e1.shape = (len(e1),1)
		e2 = np.array(e0[1],dtype='int32')
		e2.shape = (len(e2),1)   
		edge_list = np.hstack((e1,e2)).astype('int32')
		n_edges = len(edge_list)
		nodes = np.unique(edge_list.flatten()) # get number of nodes
		print('Done loading data!')

		return edge_list, nodes, n_edges

	def get_new_edge_list(self, edge_list_temp):
		''' 
		renumber edge list using dictionary of indices
		'''
		new_edge_list = np.zeros(edge_list_temp.shape,dtype='int32')
		for j,r in enumerate(edge_list_temp):
			if j % 100000 == 0: print("%.2f percent done" %(float(j)/len(edge_list_temp)))
			new_edge_list[j,0] = D[r[0]]
			new_edge_list[j,1] = D[r[1]]
		return new_edge_list

	def remap_edge_list(self, edge_list, nprocs=4):
		''' 
		re-map edge list to have consecutive numbers (in parallel)
		create a dictionary which maps node # to index 
		note that the node number might be higher than the total number of 
		unique n_nodes, which is why we are doing this
		'''
		sp = self.split(range(len(edge_list)),nprocs)
		edge_split = [edge_list[s,:] for s in sp]
		pool = mp.Pool(processes=nprocs)
		new_edge_lists = pool.map(self.get_new_edge_list,edge_split)
		new_edge_list = np.vstack(new_edge_lists)

		return new_edge_list