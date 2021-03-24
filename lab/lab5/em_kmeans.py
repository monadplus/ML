import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# given centers mu, assign to each example (row of X) its closest center
# inputs are matrix 'X' of size N x 2 with coordinates of input points, and
#            matrix 'mu' of size K x 2 with coordinates of K cluster centers
# output is a vector 'res' of length N,
#            with component res[i] an integer in the range 0 .. K indicating closest center
def assign_nearest(X, mu):
    res = np.zeros(X.shape[0], dtype=np.uint8)   # will hold cluster assignments
    for i in range(X.shape[0]):
        distance_to_centers = [np.linalg.norm(mu[k,:] - X[i,:]) for k in range(mu.shape[0])]
        res[i] = np.argmin(distance_to_centers)
    return res

##################################
#### class KMeans

class K_Means(object):
    """implements EM for a mixture of Gaussians in 2D data"""

    def __init__(self, data, K, init='random'):
        self.data = data
        self.K = K
        self.N = data.shape[0]

        self.xmin = self.data[:,0].min() - 0.1
        self.xmax = self.data[:,0].max() + 0.1
        self.ymin = self.data[:,1].min() - 0.1
        self.ymax = self.data[:,1].max() + 0.1

        # variables
        self.labels = np.zeros((self.N, 1), dtype=np.uint8)  # cluster assignments
        self.mu = np.zeros((self.K, 2))         # cluster centers

        if init == 'random':
            print(f'initializing to random input points')
            ind = choice(range(self.N), self.K, replace=False)
            self.mu = self.data[ind, :]

        elif init == 'adversarial':
            a_ = self.data[self.data[:,0].argmin(),:]
            b_ = self.data[self.data[:,0].argmax(),:]
            c_ = self.data[self.data[:,1].argmin(),:]
            d_ = self.data[self.data[:,1].argmax(),:]
            extreme = np.vstack((a_, b_, c_, d_))
            assert extreme.shape[0] >= self.K, f'not enough extreme points'
            ind = choice(extreme.shape[0], self.K, replace=False)
            self.mu = extreme[ind, :]

        else: # using k-means++ to initialize
            idx = choice(range(self.N), 1)
            self.mu[0,:] = self.data[idx,:]
            for k in range(1, self.K):
                # choose proportionally according to min distance to current centers
                closest = assign_nearest(self.data, self.mu[:k,:])
                # compute square distance
                dist2 = np.array([np.sum(np.square(self.data[i,:] - self.mu[closest[i],:]))
                                  for i in range(self.N)])
                selected = choice(np.arange(self.N), p=dist2 / np.sum(dist2))
                self.mu[k,:] = self.data[selected, :]


    def step(self, n_iterations = 1):
        """ performs `n_iterations` steps of kmeans """

        for _ in range(n_iterations):
            # 1. recompute assignments
            self.labels = assign_nearest(self.data, self.mu)

            # 2. recompute centers
            for k in range(self.K):
                self.mu[k,:] = np.mean(self.data[self.labels==k,:], axis=0)


##################################
## EM for Gaussian mixtures

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from collections import Counter
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
from numpy.random import choice
from scipy.stats.distributions import chi2

from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

def draw_ellipse(mu, cov, confint=75):
    """eig decomp of linear transf given by mu, sigma at conf. level given by chisq, default is 75% conf.int."""
    lambdas, U  = np.linalg.eig(cov)
    idx = lambdas.argsort()[::-1]
    lambdas = lambdas[idx]
    U = U[:, idx]
    assert lambdas[0] >= lambdas[1], f'eigenvalues not sorted: {lambdas, U}'

    chisq = chi2.ppf(confint/100, df=2)            
    deg = (np.arctan2(U[1,0], U[0,0])*180/np.pi + 360) % 360
            
    ell = Ellipse((0, 0), width=chisq, height=chisq, fill=None)
    transf = transforms.Affine2D()\
                .scale(np.sqrt(lambdas[0]), np.sqrt(lambdas[1]))\
                .rotate_deg(deg)\
                .translate(mu[0], mu[1])

    return ell, transf

class Gm_EM(object):
    """implements EM for a mixture of Gaussians in 2D data"""

    def __init__(self, data, K, init='kmeans'):
        """ init can be 'random' or 'kmeans' or some mu's directly """
        self.data = data
        self.K = K
        self.N = data.shape[0]

        self.xmin = self.data[:,0].min() - 0.1
        self.xmax = self.data[:,0].max() + 0.1
        self.ymin = self.data[:,1].min() - 0.1
        self.ymax = self.data[:,1].max() + 0.1

        # variables
        self.gamma = np.zeros((self.N, self.K))
        self.pi = np.zeros(self.K)              # mixing coefficients
        self.mu = np.zeros((self.K, 2))         # centers of K gaussians
        self.sigma = np.zeros((self.K, 2, 2))   # covariances of K gaussians

        if init == 'random_points':
            ind = choice(range(self.N), self.K, replace=False)
            self.mu = self.data[ind, :]
            labels = assign_nearest(self.data, self.mu)

        elif init == 'adversarial':
            a_ = self.data[self.data[:,0].argmin(),:]
            b_ = self.data[self.data[:,0].argmax(),:]
            c_ = self.data[self.data[:,1].argmin(),:]
            d_ = self.data[self.data[:,1].argmax(),:]
            extreme = np.vstack((a_, b_, c_, d_))
            assert extreme.shape[0] >= self.K, f'not enough extreme points'
            ind = choice(extreme.shape[0], self.K, replace=False)
            self.mu = extreme[ind, :]
            labels = assign_nearest(self.data, self.mu)

        elif init == 'kmeans':
            kmeans = KMeans(n_clusters=K, max_iter=10, n_init=1, init='k-means++')
            kmeans.fit(self.data)
            # init centers, partition
            self.mu = kmeans.cluster_centers_
            labels = kmeans.labels_

        else:  # init contains centers!
            assert init.shape[0] == K, f'K= {K} does not match length of mus given {init.shape}'
            self.mu = init.copy()
            labels = assign_nearest(self.data, self.mu)

        # init mixing coeff.
        ctr = Counter(labels)
        self.pi = np.array([ctr[k] + 1 for k in range(self.K)]).reshape(-1, 1) # some Laplace smoothing
        self.pi = normalize(self.pi, norm='l1', axis=0)

        # init covariance matrices.
        for k in range(self.K):
            self.sigma[k,:,:] = np.cov(self.data[labels == k, :], rowvar=False) + np.identity(2)

    def step(self, n_iterations = 1):
        """ performs `n_iterations` steps of EM """

        for _ in range(n_iterations):
            # 1. recompute soft assignments.. using logsumexp trick to avoid underflow/overflow..
            log_gamma = np.zeros((self.N, self.K))
            for k in range(self.K):
                log_gamma[:, k] = np.log(self.pi[k]) \
                    + np.log(multivariate_normal.pdf(self.data, mean=self.mu[k,:], cov=self.sigma[k,:,:]))
            log_gamma_norm = logsumexp(log_gamma, axis=1)
            self.gamma = np.exp(log_gamma - log_gamma_norm[..., np.newaxis])

            # 2. recompute Gaussian mixture
            self.pi = normalize(self.gamma.sum(axis=0).reshape(-1, 1), norm='l1', axis=0)
            for k in range(self.K):
                self.mu[k,:] = np.average(self.data, weights=self.gamma[:,k], axis = 0)
                self.sigma[k,:,:] = np.cov(self.data, rowvar=False, aweights = self.gamma[:,k])

        return

#x = np.arange(20)/20
#for val in x:
#    print(f'val = {val}, chisq={chi2.ppf(val, df=2)}')
