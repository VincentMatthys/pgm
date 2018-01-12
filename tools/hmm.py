import numpy as np
# Logsumexp function
from scipy.misc import logsumexp
# gaussian pdf class
from scipy.stats import multivariate_normal

class HMM(object):
    """
    HMM handmade class
    """

    def __init__(self, u, nbr_states, emission, A = None, pi = None):
        """
        Parameters:
        -----------
        u:                   array-like, shape (T,d)
                             T observations of d dimension
        nbr_states:          positive integer
                             number of hidden states
        emission:            array of variables
                             emission probability distributions
        A:                   array
                             state transition matrix
        pi:                  array-like, shape(nbr_states,)
                             initial state distribution
        """
        # Observations
        self.u = u
        # Model constants
        self.T = u.shape[0]
        self.d = u.shape[1]
        self.M = nbr_states
        # Model parameters
        self.pi = pi
        self.A = A
        self.O = emission
        if len(self.O) != self.M:
            raise ValueError("Number of states {} and number of emission {}"
            "distributions do not match.".format(self.M, len(self.O)))
        if A is not None:
            if A.shape[0] != A.shape[1] or not (np.round(A.sum(0), 3) == [1] * nbr_states).all():
                raise ValueError("A is not a valid transition matrix")
        if pi is not None:
            if len(pi) != nbr_states:
                raise ValueError("Number of states {} and length of distribution of initial states {}"
                                 "do not match".format(self.M, len(self.pi)))
        # Variables for recursions (inference and expectation step)
        self.alphas = None
        self.betas = None
        self.alphas_norm = None
        self.gammas = None
        self.xis = None
        # Complete log-likelihood
        self.likelihood = []
        return

    def _check_parameters(self):
        """
        Checking needed parameters for inference tasks
        """
        if self.A is None:
            raise ValueError("State transition matrix is empty")
        elif self.pi is None:
            raise ValueError("Initial state distribution is unknown")
        return
    
    def _alpha_beta_recusion(self, check_value = True):
        """
        Checking if alpha or beta recursion has to be done
        """
        if self.alphas is None:
            self._alpha_recursion()
        elif self.betas is None:
            self._beta_recursion()

    def _alpha_recursion(self):
        """
        Alpha recursion
        ---------------
        """
        # Can't do inference if A or pi is missing
        self._check_parameters()
        alphas = np.zeros((self.M, self.T))
        # Initialization of alphas
        with np.errstate(divide='ignore'):
            alphas[:, 0] = np.log(self.pi * np.array([self.O[i].pdf(self.u[0]) for i in range(self.M)]))
            # For the t_th observation
            for t in range(1, self.T):
                # Compute the path for the i_th state
                for i in range(self.M):
                    alphas[i, t] = np.log(self.O[i].pdf(self.u[t]))\
                                 + logsumexp(alphas[:, t - 1], b = self.A[:, i])
        self.alphas = alphas
        return

    def _alpha_norm_recursion(self):
        """
        Alpha normalized recursion
        """
        # Can't do inference if A or pi is missing
        self._check_parameters()
        alphas_norm = np.zeros((self.M, self.T))
        # Initialization of alphas
        alphas_norm[:, 0] = self.pi * np.array([self.O[i].pdf(self.u[0]) for i in range(self.M)])
        alphas_norm[:, 0] /= alphas_norm[:, 0].sum()
        # For the t_th observation
        for t in range(1, self.T):
            # Compute the path for the i_th state
            for i in range(self.M):
                alphas_norm[i, t] = self.O[i].pdf(self.u[t]) * (alphas_norm[:, t - 1] * self.A[:, i]).sum()
            alphas_norm[:, t] /= alphas_norm[:, t].sum()
        self.alphas_norm = alphas_norm
        return

    def _beta_recursion(self):
        """
        Beta recursion
        --------------
        """
        # Can't do inference if A or pi is missing
        self._check_parameters()
        betas = np.zeros((self.M, self.T))
        # Initialization of betas (to 0 because we take the log)
        betas[:, self.T - 1] = [0] * self.M
        # For the t_th observation
        for t in range(self.T - 2, -1, -1):
            # Compute the path for the i_th state
            for i in range(self.M):
                betas[i, t] = logsumexp(betas[:, t + 1],
                                        b = self.A[i, :] * [self.O[q].pdf(self.u[t + 1]) for q in range(self.M)])
        self.betas = betas
        return
    def _smoothing(self, t, check = True):
        """
        Inference task: smoothing
        Find the distribution of the hidden state at t given the actual observations
        """
        # Compute alphas and betas if needed
        if check == True:
            self._alpha_beta_recusion()
        return self.alphas[:, t] + self.betas[:, t] - logsumexp(self.alphas[:, -1])

    def _smoothing_all(self, check = True):
        """
        Inference task: smoothing
        Find the distribution of the hidden state at t given the actual observations
        """
        if check:
            self._alpha_beta_recusion()
        
        self.smoothing_all = np.array([self._smoothing(t, check = False) for t in range(self.T)]).T
        if (np.round(np.exp(self.smoothing_all).sum(0), 3) != 1).any():
            raise ValueError("Numerical approximations failed, please be careful with next results")

    def _gamma_recursion(self):
        """
        Gamma recursion
        Using normalized alpha recursion
        ---------------
        """
        # Can't do inference if A is missing
        self._check_parameters()
        # Use normalized alphas to avoid numerical issues
        self._alpha_norm_recursion()
        # Initialization
        gammas = np.zeros((self.M, self.T))
        gammas[:, -1] = self.alphas_norm[:, -1]
        # For the t_th observation
        for t in range(self.T - 2, -1, -1):
            # Compute the path for the i_th state
            for i in range(self.M):
                gammas[i, t] = float(np.array([self.alphas_norm[i, t] * self.A[i, j] * gammas[j, t + 1] \
                               / (self.alphas_norm[:, t] * self.A[:, j]).sum() for j in range(self.M)]).sum())
        self.gammas = gammas
        return

    def _xi_recursion(self):
        """
        Xi recursion
        Using normalized alpha recursion
        """
        # Can't do inference if gamma has not been computed yet
        if self.gammas is None:
            self._gamma_recursion()
        # Can't do inference if alphas has not been computed yet too
        if self.alphas is None:
            self._alpha_recursion()
        # Initialization
        xis = np.zeros((self.T - 1, self.M, self.M))
        # for the t_th observation
        with np.errstate(divide='ignore'):
            for t in range(self.T - 1):
                for i in range(self.M):
                    for j in range(self.M):
                        xis[t, i, j] = np.log(np.exp(logsumexp(self.alphas[:, t]) - logsumexp(self.alphas[:, t + 1])))\
                                       + np.log(self.alphas_norm[i, t]) + np.log(self.O[j].pdf(self.u[t + 1]))\
                                       + np.log(self.gammas[j, t + 1] * self.A[i, j]) \
                                       - np.log(self.alphas_norm[j, t + 1])
        self.xis = np.exp(xis)
        return

    def _e_step(self):
        """
        Expectation step
        ----------------
        """
        self._alpha_recursion()
        # Compute incomplete log_likelihood
        self._log_likelihood_incomplete()
        # Compute complete log_likelihood
#         self._log_likelihood_complete()
        self._alpha_norm_recursion()
        self._gamma_recursion()
        self._xi_recursion()

        return

    def _m_step(self):
        """
        Maximization step
        -----------------
        """
        # Pi update
        self.pi = self.gammas[:, 0]
        # A update
        self.A = self.xis.sum(0) / self.gammas.sum(1).reshape(-1, 1)
        # Emission parameters update
        params = {"mu" : np.array([(self.gammas[k, :].reshape(-1, 1) * self.u).sum(0)\
                                   / self.gammas[k, :].sum() for k in range(self.M)]),
                  "sigma" : np.array([(self.gammas[k, :].reshape(1, -1) * ((self.u - self.O[k].mean).T))\
                       .dot((self.u - self.O[k].mean)) / self.gammas[k, :].sum() for k in range(self.M)])}
        self.O = np.array([multivariate_normal(params["mu"][key], params["sigma"][key])
                           for key in range(params["mu"].shape[0])])
        return

    def _log_likelihood_incomplete(self):
        """
        Compute incomplete log-likelihood
        """
        print("--------------------\nLog-likelihood\n")
        print(self.alphas[:, -1])
        self.likelihood.append(logsumexp(self.alphas[:, -1]))
#         self.likelihood.append(self.alphas[-1, -1])
        return 

    def _log_likelihood_complete(self):
        """
        Compute the complete log-likelihood
        """
        
        l = np.array([[np.log(self.O[k].pdf(self.u[t])) * self.gammas[k, t] + 1e-10 for k in range(self.M)]\
                  for t in range(self.T)]).sum() \
               + (np.log(self.A) * self.xis.sum(0)).sum() \
               + (np.log(self.pi) * self.gammas[:, 0]).sum()
        self.likelihood.append(l)
        return

    def _EM(self, A, pi, emission):
        """
        Learning parameters with Expectation-Maximization algorithm
        (Only works for gaussian emission)
        -----------------------------------------------------------
        
        """
        # Initial start
        self.A = A
        self.pi = pi
        self.O = emission
        # Stop condition
        stop = False
        while (stop == False):
            self._e_step()
            self._m_step()
            print (self.likelihood[-1])
            if (len(self.likelihood) > 2 and np.abs(self.likelihood[-2] - self.likelihood[-1]) < 1e-3):
                stop = True
        return