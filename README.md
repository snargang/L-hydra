# L-hydra

Implementation of L-hydra and L-hydra+

L-hydra embedds networks into hyperbolic space, which requires only the distance measurements to a few 'landmark nodes'. It minimizes the stress of the embedding compared to the original pairwise distances between points. A detailed description on the algorithm can be found in our paper 'Strain-Minimizing Hyperbolic Network Embeddings with Landmarks', see https://arxiv.org/pdf/2207.06775.pdf. See the tests for some examples.


## Acknowledgement

The implementation of L-hydra+ is based on the source code of the HyPy algorithm from Kenny Chowdhary and Tamara Kolda (https://doi.org/10.1093/comnet/cnx034).
