"""
Classifying individuals in a sample into populations via LDA
(using variational inference with mean field assumption to approximiate inference)
Usage: TODO...specify input, returns
Author: Ruixi Lin
Date: 09-19-2019
Version: v0.1
References:
[1] D.M. Blei, A.Y. Ng, and M.I. Jordan. Latent Dirichlet allocation. JMLR, 3:993–1022, 2003
"""

import scipy
from scipy import io
from scipy import special
import numpy as np
import pandas
import datetime

CONVERGENCE_THRESHOLD = 1e-3

data = scipy.io.loadmat('proj2_data.mat')
individuals = data['data']	# (100, 200)
beta = data['beta_matrix']	# (200, 4) beta is already learned

# Parameters
alpha = 0.1 # hyperparameter

M = individuals.shape[0] # 100
N = individuals.shape[1] # 200 (doc length is a fixed length in this setting; can be variable length for each doc)
K = beta.shape[1]		 # 4

# Genotype vocab
vocab = list(set([x for s in individuals for x in s]))  # [0, 1, 2, 3, 4, 5, 6]

# Initialization
# Create a word-topic probability matrix (phi) and doc-topic probability matrix (gamma)
# TODO: the update has nothing to do with individual[m]'s information...bug here
# TODO: repeat for convergence
def lda():
	for n in range(N):
		for i in range(K):
			phi[n][i] = np.log(beta[n][i]*np.exp(scipy.special.digamma(gamma[i]))) # take log to avoid underflow or overflow
		phi[n] = phi[n] / np.linalg.norm(phi[n])
	gamma = alpha + np.sum(phi, axis=0)
	return phi, gamma


# Trial 0: run LDA variational inference for individual 1
'''
phi = np.full((N, K), 1/K)
gamma = np.full((K,), alpha+N/K)
for n in range(N):
	for i in range(K):
		phi[n][i] = beta[n][i]*np.exp(scipy.special.digamma(gamma[i]))
	phi[n] = phi[n] / np.linalg.norm(phi[n])

gamma = alpha + np.sum(phi, axis=0)

N_0 = list(set(individuals[0])) # [0, 1, 2, 3]
phi_out = np.zeros((len(N_0), K))
for n in N_0:
	indices = np.where(individuals[0] == n)[0]
	phi_out[n] = np.sum(phi[indices], axis=0)

# Write phi_out to file
np.savetxt('phi1.out', phi_out)
print('phi1.out saved to current directory')
'''

# Iterate through LDA Inference for all individuals
# In the language of text, the optimizing parameters (γ∗(w),φ∗(w)) are document-specific
# I.e., the program iterates within each individual, no global updates across individuals
e_phi = np.infty
e_gamma = np.infty
theta = np.zeros((M, K))	# doc-topic assignment
phi = np.full((N, K), 1/K)
gamma = np.full((K,), alpha+N/K)

iter = 0
for alpha in [0.1]:#[0.01, 0.1, 1.0, 10.0]:
	while e_phi > CONVERGENCE_THRESHOLD and e_gamma > CONVERGENCE_THRESHOLD: 
		# Update
		for m in range(M):
			phi_new, gamma_new = lda() # TODO: what does it mean by each value of ... is less than threshold??
			theta[m] = gamma_new
		e_phi = np.abs(phi, phi_new)
		e_gamma = np.abs(gamma, gamma_new)
		iter += 1
	print('Iterations to converge when alpha = %f:'%(alpha), iter)
	print('Theta ', theta)




















