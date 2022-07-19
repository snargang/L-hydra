from lm_emb_preprocessing import Preprocessing
from lm_emb_hydra import LHydra
from lm_emb_hydra_plus import LHydraPlus

'''
Parameters that need to be provided by the user:

Parameters for Preprocessing
----------------------------
name : Char 
	Name of the network to embed.
indir : Char
	Directory for input data from snap.
outdir : Char
	Directory for output data.
nprocs : Int
	Number of processes for parallel embedding during L-hydra+.
	(Necessary to part hyperbolic coordinates into chunks.)
nlandmarks: Int
	Number of landmarks
nvalidation: Int
	Number of validation nodes
maxprocs4seed: Int
	Number of procs for parallel landmark
split: Boolean
	If split = True, the user has to define nodes_per_run.
	Split defines how many serial chunks to perform the nonlandmark
	embedding.
nodes_per_run: Int
	Number of nodes per run of nonlandmark embedding

Parameters for L-hydra / L-hydra+
---------------------------------
curvature : Float, optional
	Embedding curvature. The default is 1.
dim : Int, optional
	Embedding dimension. The default is 2.
alpha : Float, optional
	Adjusts the hyperbolic curvature. The default is 1.1.
equi_adj : Float, optional
	Parameter for equi-angular adjustment; must be a real number between zero and
	one; only used if dim is 2. The default is 0.5.
	
Parameters for Parallelization of L-Hydra+
------------------------------------------
nrepeat: Int
	Number of repetitions for landmark embedding
	Best result is used to continue with nonlandmark embedding
'''

# Parameters for Preprocessing
name = "amazon"
snapdir = "/home/h4/s8550570/data/amazon/snap/"
savedir = "/home/h4/s8550570/data/amazon/"
nprocs = 4
nlandmarks = 100
nvalidation = 1000
maxprocs4seed = 8
split = True
nodes_per_run = 100000
# Parameters for L-hydra / L-hydra+
curvature = 1
dim = 2
alpha = 1
equi_adj = 0.5
# Parameters for Parallelization of L-Hydra+
nrepeat = 8

# Run preprocessing
preproc = Preprocessing(name, snapdir, savedir, nprocs, nlandmarks, nvalidation)
filename_remapping = name + '_edges'
preproc.ReMapNodes(filename_remapping)
filename_adjmat = name + '_edges_remapped'
preproc.GenAdjMat(filename_adjmat)
preproc.set_main_env(maxprocs4seed, split, nodes_per_run)
  
# Landmark Embedding with input from Hydra
# Run Hydra
hydra = LHydra(name, snapdir, savedir, nprocs)
hydra.hydra_landmark(curvature, dim, alpha, equi_adj)
  	   
# Run Hydra Plus
hydra_plus = LHydraPlus(name, savedir, nlandmarks, nodes_per_run)
hydra_plus.embed_landmarks(dim, 'mpirun', nrepeat)
hydra_plus.embed_nonlandmarks(nprocs)
hydra_plus.validate(savedir,nprocs)

