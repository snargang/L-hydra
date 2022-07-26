import numpy as np 
import os, math, warnings

from scipy.linalg import eigh
from scipy.optimize import minimize

class LHydra:
	'''
	Implements the L-hydra (landmarked hyperbolic distance recovery and approximation) method 
	for embedding high-dimensional data points (represented by their distance matrix \code{D}) 
	into low-dimensional hyperbolic space.
	
	Parameters
	----------
	name : Char 
		Name of the network to embed.
	indir : Char
		Directory for input data from snap.
	outdir : Char
		Directory for output data.
	nprocs : Int
		Number of processes for parallel embedding during L-hydra+.
		(Necessary to part hyperbolic coordinates into chunks.)
	'''
	
	def __init__(self, name, indir, outdir, nprocs):
		print("Start Hydra-Embedding")
		self.name = name
		self.snapdir = indir
		self.datadir = outdir
		self.nprocs = nprocs
		
	def hydra_landmark(self, dim=2, curvature=1.0, alpha=1.1, equi_adj=0.5, polar=False, isotropic_adj=True, hydra=False, lorentz=False):
		''' 
		Wrapper function for different variants of the L-hydra method.
		
		If the curvature is 'None', L-hydra tries to find the optimal curvature.
		Otherwise, L-hydra is run with fixed curvature.
		'''
		if curvature != None:
			return self.hydra_landmark_fixed_curvature(curvature, dim, alpha, equi_adj, polar, isotropic_adj, hydra, lorentz)
		else:
			# setup control parameter for curvature optimization
			hydra=False
			lorentz = False
			
			# initial value for curvature 
			curvature = 1
			
			eps = np.finfo(np.double).eps
			D_land = np.load(os.path.join(self.datadir, 'D_land.npy')).astype('int')
			k_bounds = [eps,max(1,(8/D_land.max())**2)] # a priori bounds for curvature
			k_min = minimize(
				fun=self.hydra_landmark_fixed_curvature,
				args=(dim, alpha, equi_adj, polar, isotropic_adj, hydra, lorentz),
				x0=curvature,
				method="BFGS",
				bounds=k_bounds,
				options={"disp": True, "maxiter": 1000}
			)
			k1_objective = self.hydra_landmark_fixed_curvature(1, dim, alpha, equi_adj, polar, isotropic_adj, hydra, lorentz)
			k_optimal = k_min.x[0]
			if k1_objective < k_min.fun: 
				k_optimal = 1 # make sure that returned result is never worse than unit curvature
			return self.hydra_landmark_fixed_curvature(k_optimal, dim, alpha, equi_adj, polar, isotropic_adj, hydra, lorentz)
				
	def hydra_landmark_fixed_curvature(self, curvature=1.0, dim=2, alpha=1.1, equi_adj=0.5, polar=False, isotropic_adj=True, hydra=False, lorentz=False):
		'''
		Parameters
		----------
		curvature : Float, optional
			Embedding curvature. The default is 1.
		dim : Int, optional
			Embedding dimension. The default is 2.
		alpha : Float, optional
			Adjusts the hyperbolic curvature. Values larger than one yield a
			more distorted embedding where points are pushed to the outer
			boundary (i.e. the ideal points) of hyperblic space. The
			interaction between code{curvature} and code{alpha} is non-linear.
			The default is 1.1.
		equi_adj : Float, optional
			Equi-angular adjustment; must be a real number between zero and
			one; only used if dim is 2. Value 0 means no ajustment, 1
			adjusts embedded data points such that their angular coordinates
			in the Poincare disc are uniformly distributed. Other values
			interpolate between the two extremes. Setting the parameter to non-
			zero values can make the embedding result look more harmoniuous in
			plots. The default is 0.5.
		polar:
			Return polar coordinates in dimension 2. (This flag is
			ignored in higher dimension).
		isotropic_adj :
			Perform isotropic adjustment, ignoring Eigenvalues
			(default: TRUE if dim is 2, FALSE else)
		hydra:
			Return radial and spherical coordinates (default: FALSE)
		lorentz:
			Return raw Lorentz coordinates (before projection to
			hyperbolic space) (default: FALSE)
			
		Yields
		------
		X : float array
			Hyperbolic coordinates in chosen space
		'''
		
		# load distance matrices and landmark indices (as computed during pre-processing)
		D_land = np.load(os.path.join(self.datadir, 'D_land.npy')).astype('int') # distances between landmark nodes
		D_nonland = np.load(os.path.join(self.datadir, 'D_nonland.npy')).astype('int') # distances between landmark and non-landmark nodes

		nodes = np.loadtxt(os.path.join(self.datadir, 'nNodes.txt'),dtype='int32') # number of network nodes
		land_ids = np.loadtxt(os.path.join(self.datadir, 'landmark_ids.txt'),dtype='int32') # indices of landmark nodes
		ind = ~np.in1d(range(nodes),land_ids)
		nonland_ids = np.arange(nodes)[ind] # indices of non-landmark nodes

		# sanitize/check input
		if any(np.diag(D_land) != 0):  # non-zero diagonal elements are set to zero
			np.fill_diagonal(D_land, 0)
			warnings.warn("Diagonal of input matrix D_land has been set to zero")
			
		if dim > len(D_land):
			raise RuntimeError(
				f"Hydra cannot embed {len(D_land)} points in {dim}-dimensions. Limit of {len(D_land)}."
			)
		
		if not np.allclose(D_land, np.transpose(D_land)):
			warnings.warn(
				"Input matrix D_land is not symmetric.\
				Lower triangle part is used."
			)

		if dim > 2:
			# set default values in dimension > 2
			isotropic_adj = False
			if polar:
				warnings.warn("Polar coordinates only valid in dimension two")
				polar = False
			if equi_adj != 0.0:
				warnings.warn("Equiangular adjustment only possible in dimension two.")

			
		# convert distance matrix to 'hyperbolic Gram matrix'
		A_land = np.cosh(np.sqrt(abs(curvature))*D_land)
		A_nonland = np.cosh(np.sqrt(abs(curvature))*D_nonland)
		nlm = len(land_ids)
		nnonlm = nodes - nlm
		
		# check for large/infinite values
		A_max = np.amax(A_land)
		if A_max > 1e8:
			warnings.warn(
				"Gram Matrix contains values > 1e8. Rerun with smaller\
				curvature parameter or rescaled distances."
			)
		if A_max == float("inf"):
			warnings.warn(
				"Gram matrix contains infinite values.\
				Rerun with smaller curvature parameter or rescaled distances."
			)

		# Eigendecomposition of A
		# compute leading Eigenvalue and Eigenvector
		lambda0, x0 = eigh(A_land, subset_by_index=[nlm-1, nlm-1])
		# compute lower tail of spectrum
		w, v = eigh(A_land, subset_by_index=[0,dim-1])
		idx = w.argsort()[::-1]
		spec_tail = w[idx] # Last dim Eigenvalues
		X_land_raw = v[:,idx] # Last dim Eigenvectors
		
		x0 = x0 * np.sqrt(lambda0) # scale by Eigenvalue
		if x0[0]<0:
			x0 = -x0 # Flip sign if first element negative

		# no isotropic adjustment: rescale Eigenvectors by Eigenvalues
		if not isotropic_adj:
			if np.array([spec_tail > 0]).any():
				warnings.warn(
					"Spectral Values have been truncated to zero. Try to use\
					lower embedding dimension"
				)
				spec_tail[spec_tail > 0] = 0
			X_land_raw = np.matmul(X_land_raw, np.diag(np.sqrt(np.maximum(-spec_tail,0))))
			
		X_nonland = np.matmul(np.transpose(A_nonland), np.c_[(x0/lambda0), -(X_land_raw/abs(spec_tail))])
		
		X_raw = np.zeros((nodes, dim))
		X_raw[land_ids,:] = X_land_raw
		X_raw[nonland_ids,:] = X_nonland[:,1:(dim+1)]
		
		x0_full = np.zeros((nodes, 1))
		x0_full[land_ids,0] = x0[:,0]
		x0_full[nonland_ids,0] = X_nonland[:,0]
		x_min = x0_full.min()
		
		
		if hydra:
			# Calculate radial coordinate
			s = np.sqrt(np.sum(X_raw ** 2, axis=1))
			directional = X_raw / s[:, None]  # convert to directional coordinates
			r = np.sqrt((alpha*x0_full - x_min)/(alpha*x0_full + x_min)) ## multiplicative adjustment (scaling)
			X = self.poincare_to_hyper(r=r, directional=directional)
		else:
			X = X_raw
			
		# Calculate polar coordinates if dimension is 2
		if dim == 2:
			# calculate polar angle
			theta = np.arctan2(X[:, 0], -X[:, 1])

			# Equiangular adjustment
			if equi_adj > 0.0:
				angles = [(2 * x / nodes - 1) * math.pi for x in range(0, nodes)]
				theta_equi = np.array(
					[x for _, x in sorted(zip(theta, angles))]
				)  # Equi-spaced angles
				# convex combination of original and equi-spaced angles
				theta = (1 - equi_adj) * theta + equi_adj * theta_equi
				# update directional coordinate
				directional = np.array([np.cos(theta), np.sin(theta)]).transpose()
				
		if lorentz:
			X_lorentz = np.concatenate((x0_full, X), axis=1)
		
		np.save(os.path.join(self.datadir, 'xstart'), X)
		print('*** DIMENSIONS OF XSTART ' + str(X.shape[0]) + ' and ' + str(X.shape[1]))
		self.split_xstart(nodes, nlm, self.datadir, self.nprocs)
		
		#stress = self.get_stress(curvature, X_raw[land_ids,:], D_land)
		
		return X
		
	def split_xstart(self, nodes, landmarks, dataloc, nprocs):
		'''
		Splits (hyperbolic) coordinates into nprocs chunks of equal size
		'''
		node_ids = np.array_split(np.arange(nodes - landmarks),nprocs)
		print("Loading hydra results...")
		xstart_full = np.load(os.path.join(dataloc, 'xstart.npy'))
		nonland_ids = np.loadtxt(os.path.join(dataloc, 'nonlandmark_ids.txt'), dtype='int32')
		xstart = xstart_full[nonland_ids,:]
		print("Length of starting point:" + str(len(xstart)))
		for ii,nn in enumerate(node_ids):
			print("Splitting %i out of %i" %(ii+1,len(node_ids)))
			xstart_tmp = xstart[nn,:]
			np.save(os.path.join(dataloc, 'xstart_split_' + str(ii).zfill(3)), xstart_tmp)
	
	def get_stress(self, curvature, x, dist):
		'''
		Calculate stress of embedding from given coordinates
		'''
		X = np.matmul(x, x.T)
		u_tilde = np.sqrt(X.diagonal() + 1)
		H = X - np.outer(u_tilde, u_tilde)
		D = 1 / np.sqrt(abs(curvature)) * np.arccosh(np.maximum(-H, 1))
		np.fill_diagonal(D, 0)
		y = 0.5 * np.sum((D - dist) ** 2)
		return y
		
	def poincare_to_hyper(self, r, directional):
		'''
		Convert coordinates in the Poincare ball to reduced hyperbolic coordinates
		'''
		X = (directional.transpose()*(2*r/(1-r**2))).transpose()
		return X
