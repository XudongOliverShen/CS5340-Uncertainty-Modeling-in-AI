"""
Classifying individuals in a sample into populations via LDA
(using variational inference with mean field assumption to approximiate inference)
Usage: TODO...specify input, returns
Author: Ruixi Lin & Xudong Shen
Date: 09-19-2019
Version: v0.1
References:
[1] D.M. Blei, A.Y. Ng, and M.I. Jordan. Latent Dirichlet allocation. JMLR, 3:993–1022, 2003
"""

import numpy as np
from scipy import io
from scipy.special import digamma
from tqdm import tqdm


def data_preprocess(d):
    
    '''
    data_preprocess function transform data in matrix D to LDA form
    misconceptions: we cannot directly apply LDA VI on rows in matrix D!
                    the number of genotype present in individuals is not fixed,
                    i.e., in the example of documents, each document has different number of words.
                    D(i,j) represent the number of OCCURANCE of genotype j in individual i
                    for example, if D(38,155) = 2, 38-th individual(a document) has 2 155-th genotype loci(words).
    
    args:
        d: row in D, length V
    returns:
        w: inference-ready form
    '''
    
    # preprocess data
    V = d.shape[0]
    w = []
    for locus in range(V):
        if d[locus] != 0:
            for i in range(d[locus]):
                w.append(locus)
    w = np.array(w)
    return w

def LDA_inference(w, alpha, beta):
    
    '''
    Latent Dirichlet Allocation inference step, also knowed as E-step in EM algorithm
    
    args:
        w: (N) vector, an individual's genotype loci, N varies between individuals
        alpha: (4) vector,
        beta: (200,4) matrix, ancestor_population-genotype_locus (topic-word) distribution
    returns:
        phi: (N,4) matrix, learned genotype_loci-ancestor_population (word-topic) distribution
        gamma: (4) vector, individual-ancestor_population (document-topic) distribution
    '''
    
    # constants
    epsilon = 1e-3 # convergence threshold
    max_iter = 1000
    
    # get dimensions
    N = w.shape[0]
    V, K = beta.shape # V, vocabulary size; K, number of ancestor populations (topics)
    
    # initialize variables
    phi = 1/K * np.ones([N,K]) # genotype_loci-ancestor_population (word-topic) distribution
    gamma = alpha+N/K # individual-ancestor_population (document-topic) distribution
    phi_new = np.copy(phi)
    log_phi_new = np.zeros(K)
    gamma_new = np.copy(gamma)
    
    # inference
    # we update in log space
    iter_count = 0
    while True:
        psi = digamma(gamma) - digamma(sum(gamma))
        for n in range(N):
            for i in range(K):
                log_phi_new[i] = np.log(beta[w[n], i]) + psi[i]
                # a little trick to calculate log normalizing term
                # log(a+b) = log(a) + log(1 + exp(log(b) - log(a)))
                if i == 0:
                    norm = log_phi_new[i]
                else:
                    norm = norm + np.log(1 + np.exp(log_phi_new[i] - norm))
            log_phi_new = log_phi_new - norm # normalize in log space
            phi_new[n,:] = np.exp(log_phi_new)
        gamma_new = alpha + np.sum(phi_new, axis=0)
        iter_count += 1
        
        # calculate erorr, then update
        E1 = abs(phi_new - phi)
        E2 = abs(gamma_new - gamma)
        phi = np.copy(phi_new)
        gamma = np.copy(gamma_new)
        
        # check convergence
        if (E1<epsilon).all() & (E2<epsilon).all():
            return phi, gamma, iter_count
        elif iter_count > max_iter:
            raise ValueError('Warning! did not converge after %d iterations. Please modify and try again.' % max_iter)


#%%
if __name__ == '__main__':
    
    # Parameters
    alpha = 0.1 * np.ones(4) # individual-ancestor_population (document-topic) distribution
    
    # load data
    data = io.loadmat('proj2_data.mat')
    D = data['data']	# (100, 200), 100 individuals, each represented by a vocabulary of N=200 genotype loci
    beta = data['beta_matrix']	# (200, 4) beta is already learned (word-topic distribution)
    
    # initialize dimensions
    M = D.shape[0] # M = 100, number of individuals (documents)
    V = D.shape[1] # V = 200, number of genotype loci (vocabulary size)
    K = beta.shape[1] # K = 4, number of ancestor populations (topics)
    
    #%%
    # Task 1:
    #   run LDA variational inference for individual 1
    #   returns phi1 (a matrix of size n1 x K), save as phi1.out
        
    print('running task 1/3...')
    
    # preprocess data
    w = data_preprocess(D[0,:])
    
    # inference
    phi, gamma, _ = LDA_inference(w, alpha, beta)
    
    # save
    np.savetxt('phi1.out', phi)
    print('phi1.out saved to current directory.')
    print('finished task 1/3. Press enter to continue...')
    wait = input()
    
    #%%
    print('running task 2/3...')
    
    # initialize empty variables
    Theta = np.zeros([M, K])
    iterations = np.zeros(M) # store number of iterations for each individual
    
    # loop over all individuals
    for i in tqdm(range(M)):
        w = data_preprocess(D[i, :]) # data preprocess
        _, Theta[i, :], iterations[i] = LDA_inference(w, alpha, beta) # LDA inference
    
    # save
    np.savetxt('Theta.out', Theta)
    print('Theta.out saved to current directory.')
    print('finished task 2/3. Press enter to continue...')
    wait = input()
    
    #%%
    print('running task 3/3...')
    
    # initialize empty variables
    Theta = np.zeros([3, M, K])
    iterations = np.zeros([3, M]) # store number of iterations for each individual
    
    for n in range(3):
        
        print('running experiment %d/3' % n+1)
        alpha_list = [0.01, 1, 10]
        alpha = alpha_list[n] * np.ones(4)
        
        for i in tqdm(range(M)):
            w = data_preprocess(D[i,:]) # data preprocess
            _, Theta[n, i, :], iterations[n, i] = LDA_inference(w, alpha, beta) # LDA inference
    
    # save
    np.savez('alpha_experiment.npz', Theta, iterations)
    print('alpha_experiment.npz saved to current directory.')
    print('finished task 3/3.')