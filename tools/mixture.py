#!/usr/bin/python3

import numpy as np
from scipy.stats import multivariate_normal

class K_means(object):
    """
    K-means clustering class
    """
    def __init__(self, K):
        self.K = K
        self.C = None
        self.A = None
        self.distorsion = []
        return
    def _random_centroid(self, datax):
        """
        Draw centroids chosen uniformly at random among the data
        """
        np.random.shuffle(datax)
        return datax[:self.K].reshape(self.K, 1, -1)
    def _one_pass(self, datax):
        """
        One pass of the K_means alternative minimization by computing the distance matrix
        1. Assign each point to the closet centroid
        2. Compute distorsion for each pass and add it to the histogram
        """
        # Compute the distance matrix from each point to each centroid
        distance_matrix = np.linalg.norm(np.tile(datax, (self.K, 1, 1)) - self.C, axis = 2)
        # Get the argmin of the distance for all centroids
        self.A = distance_matrix.argmin(axis = 0)
        # Get the current value of distorsion
        ## Store the minimum distance
        min_dist = distance_matrix.min(axis = 0)
        self.distorsion.append((min_dist ** 2).sum())
        # Recompute the centroids
        self.C = np.array([(lambda a : datax[a].sum(axis = 0)/a.sum())(self.A == k)\
                           for k in range(self.K)]).reshape(self.K, 1, -1)
    def fit(self, datax):
        """
        """
        # Initialisation
        ## Of centroids
        self.C = self._random_centroid(datax)
        ## Stop condition
        stop = False
        while (stop == False):
            self._one_pass(datax)
            if (len(self.distorsion) > 1):
                stop = True if self.distorsion[-2] - self.distorsion[-1] < 1 else False
        return self.C, self.A
    def predict(self, datax):
        """
        Predict cluster given current centroids
        """
        # Compute the distance matrix from each point to each centroid
        distance_matrix = np.linalg.norm(np.tile(datax, (self.K, 1, 1)) - self.C, axis = 2)
        return distance_matrix.argmin(axis = 0)

class GM(object):
    """
    Gaussian Mixture (EM) method for clustering
    """
    def __init__(self, K, covariance_type = "general"):
        self.K = K
        self.mu = None
        self.sigma = None
        self.pi = None
        self.q = None
        self.L = []
        self.covariance_type = covariance_type

        # Check the covariance model is defined
        if self.covariance_type not in ['general', 'isotropic']:
            raise ValueError("Invalid value for 'covariance_type': %s "
                             "'covariance_type' should be in "
                             "['general', 'isotropic']" % self.covariance_type)
    def _e_step(self, x):
        """
        Expectation step
        """
        gaussiens = np.array([multivariate_normal(self.mu[i].reshape(-1),
                                         self.sigma[i]
                                        ).pdf(x) for i in range(self.K)])
        self.q = gaussiens * self.pi
        self.q = self.q / self.q.sum(axis = 0)
        # Likelihood
        self.L.append(np.log((gaussiens * self.pi).sum(axis = 0)).sum())
        self.q = self.q.reshape(self.K, len(x), 1)
        return
    def _m_step(self, x):
        """
        Maximization step
        """
        self.mu = (x * self.q).sum(axis = 1) / self.q.sum(axis = 1)
        self.mu = self.mu.reshape(self.K, 1, -1)
        if self.covariance_type == "isotropic":
            self._isotropic_sigma(x)
        elif self.covariance_type == "general":
            self._general_sigma(x)
        self.pi = self.q.sum(axis = 1) / len(x)
        return
    def _initialization(self, x):
        """
        Initializes mu, sigma and pi
        """
        self.mu, A = K_means(self.K).fit(x)
        self.q = np.zeros((self.K, len(x), 1))
        for i,j in np.ndenumerate(A):
            self.q[j, i] = 1

        if self.covariance_type == "isotropic":
            self._isotropic_sigma(x)
        elif self.covariance_type == "general":
            self._general_sigma(x)
        self.pi = self.q.sum(axis = 1) / len(x)
    def fit(self, x):
        """
        """
        # Initialisation
        ## Of centroids
        self._initialization(x)
        stop = False
        while (stop == False):
            self._e_step(x)
            self._m_step(x)
            if (len(self.L) > 2 and np.abs(self.L[-2] - self.L[-1]) < 10e-3):
                stop = True
        return
    def _isotropic_sigma(self, x):
        """
        Update sigma covariance matrix given an isotropic model
        """
        self.sigma = 0.5 * (((np.linalg.norm(x - self.mu, axis = 2) ** 2\
                        * self.q.reshape(self.K, -1)).sum(axis = 1)\
                        / (self.q.reshape(self.K, -1).sum(axis = 1))).reshape(self.K, 1, 1))
        self.sigma = self.sigma * np.identity(self.mu.shape[-1])
        return
    def _general_sigma(self, x):
        """
        Update sigma covariance matrix given a general model
        """
        self.sigma = np.matmul(np.transpose(x-self.mu, (0, 2, 1)),
                               (x-self.mu) * self.q) / self.q.sum(axis = 1)\
                                .reshape(self.K, 1, 1)
        return

def gen_arti(centerx = 1,
     centery = 1,
     sigma = 0.1,
     nbex = 1000,
     data_type = 0,
     epsilon = 0.02,
     cov1 = None,
     cov2 = None):
    """
    Generateur de donnees,
    :param centerx: centre des gaussiennes
    :param centery:
    :param sigma: des gaussiennes
    :param nbex: nombre d'exemples
    :param data_type: 0: melange 2 gaussiennes, 1: melange 4 gaussiennes, 2:echequier
    :param epsilon: bruit dans les donnees
    :return: data matrice 2d des donnnes,y etiquette des donnnees
    """
    if cov1 is None:
        cov1 = np.diag([sigma, sigma])
    if cov2 is None:
        cov2 = np.diag([sigma, sigma])
    if data_type == 0:
        # melange de 2 gaussiennes
        xpos = np.random.multivariate_normal([centerx, centerx],
                                   cov1,
                                   int(nbex/2))
        xneg = np.random.multivariate_normal([-centerx, -centerx],
                                   cov2,
                                   int(nbex/2))
        data = np.vstack((xpos, xneg))
        y = np.hstack((np.ones(int(nbex/2)), -np.ones(int(nbex/2))))
    if data_type == 1:
        # melange de 4 gaussiennes
        xpos = np.vstack((
        np.random.multivariate_normal([centerx, centerx],
                                      cov1,
                                      int(nbex/4)),
        np.random.multivariate_normal([-centerx, -centerx],
                                      cov2,
                                      int(nbex/4))
        ))
        xneg = np.vstack((
        np.random.multivariate_normal([-centerx, centerx],
                                      cov1,
                                      int(nbex/4)),
        np.random.multivariate_normal([centerx, -centerx],
                                      cov2,
                                      int(nbex/4))
        ))
        data = np.vstack((xpos, xneg))
        y = np.hstack((np.ones(int(nbex/2)), -np.ones(int(nbex/2))))
    if data_type == 2:
        # echiquier
        data = np.reshape(np.random.uniform(-4, 4, 2*nbex), (nbex,2))
        y = np.ceil(data[:,0]) + np.ceil(data[:,1])
        y = 2 * (y%2) - 1
        # un peu de bruit
        data[:,0] += np.random.normal(0, epsilon, nbex)
        data[:,1] += np.random.normal(0, epsilon, nbex)
        # on melange les donnees
        idx = np.random.permutation((range(y.size)))
        data = data[idx,:]
        y = y[idx]
    return data,y
