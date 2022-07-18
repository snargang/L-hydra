# L-hydra

Implementation of L-hydra and L-hydra+

L-hydra embeds networks into hyperbolic space, which requires only the distance measurements to a few 'landmark nodes'. L-hydra+ uses the result of L-hydra as an initial condition for stress minimization of the embedding compared to the original pairwise distances between points. A detailed description of both algorithms can be found in our paper 'Strain-Minimizing Hyperbolic Network Embeddings with Landmarks', see https://arxiv.org/pdf/2207.06775.pdf. See the tests for some examples.


## Acknowledgement

The implementation of L-hydra+ is based on the source code of the HyPy algorithm from Kenny Chowdhary and Tamara Kolda (https://doi.org/10.1093/comnet/cnx034).
