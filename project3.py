import random
import time

import numpy as np
import scipy.stats
import pandas as pd


def k_means(data, k, eps=1e-4, mu=None):
    """ Run the k-means algorithm
    data - an nxd ndarray
    k - number of clusters to fit
    eps - stopping criterion tolerance
    mu - an optional kxd ndarray containing initial centroids

    returns: a tuple containing
        mu - a kxd ndarray containing the learned means
        cluster_assignments - an n-vector of each point's cluster index
    """
    n, d = data.shape
    if mu is None:
#         print(data[:3])
#         print(random.sample(range(data.shape[0]), k))
#         print(data[random.sample(range(data.shape[0]), k)])
        # randomly choose k points as initial centroids
        mu = data[random.sample(range(data.shape[0]), k)]
    cluster_assignments = np.zeros(n)
    cost = float('inf')
    new_cost = 0
    while abs(new_cost - cost) > eps:
        cost = new_cost
        new_cost = 0
        new_mu = np.zeros((k, d))
        mu_count = np.zeros(k)
        for i in range(0, n):
            distance = np.linalg.norm(mu-data[i], axis=1)
            label = np.argmin(distance)
            cluster_assignments[i] = label
            new_cost += distance[label]
            new_mu[label] += data[i]
            mu_count[label] += 1
        mu = new_mu / mu_count.reshape((k, 1))
    #print("k =", str(k) + ": cost is", new_cost)
    return (mu, cluster_assignments)

class MixtureModel(object):
    def __init__(self, k):
        self.k = k
        self.params = {
            'pi': np.random.dirichlet([1]*k),
        }

    def __getattr__(self, attr):
        if attr not in self.params:
            raise AttributeError()
        return self.params[attr]

    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)

    def e_step(self, data):
        """ Performs the E-step of the EM algorithm
        data - an NxD pandas DataFrame

        returns a tuple containing
            (float) the expected log-likelihood
            (NxK ndarray) the posterior probability of the latent variables
        """
        raise NotImplementedError()

    def m_step(self, data, p_z):
        """ Performs the M-step of the EM algorithm
        data - an NxD pandas DataFrame
        p_z - an NxK numpy ndarray containing posterior probabilities

        returns a dictionary containing the new parameter values
        """
        raise NotImplementedError()

    def fit(self, data, eps=1e-4, verbose=True, max_iters=50): # max_iters=100
        """ Fits the model to data
        data - an NxD pandas DataFrame
        eps - the tolerance for the stopping criterion
        verbose - whether to print ll every iter
        max_iters - maximum number of iterations before giving up

        returns a boolean indicating whether fitting succeeded

        if fit was successful, sets the following properties on the Model object:
          n_train - the number of data points provided
          max_ll - the maximized log-likelihood
        """
        last_ll = np.finfo(float).min
        start_t = last_t = time.time()
        i = 0
        while True:
            i += 1
            if i > max_iters:
                return False
            ll, p_z = self.e_step(data)
            new_params = self.m_step(data, p_z)
            self.params.update(new_params)
            if verbose:
                dt = time.time() - last_t
                last_t += dt
                print('iter %s: ll = %.5f  (%.2f s)' % (i, ll, dt))
                last_ts = time.time()
            if abs((ll - last_ll) / ll) < eps:
                break
            last_ll = ll

        setattr(self, 'n_train', len(data))
        setattr(self, 'max_ll', ll)
        self.params.update({'p_z': p_z})

        print('max ll = %.5f  (%.2f min, %d iters)' %
              (ll, (time.time() - start_t) / 60, i))

        return True


class GMM(MixtureModel):
    def __init__(self, k, d):
        super(GMM, self).__init__(k)
        self.params['mu'] = np.random.randn(k, d)

    def e_step(self, data):
        """ Performs the E-step of the EM algorithm
        data - an NxD pandas DataFrame

        returns a tuple containing
            (float) the expected log-likelihood
            (NxK ndarray) the posterior probability of the latent variables
        """
        #print(type(data)) # actually not a Pandas DataFrame: <class 'numpy.ndarray'>
        K = self.k
        N, D = data.shape
        normals = np.zeros((N, K))
        for i in range(0, N):
            x = data[i]
            for j in range(0, K):
                normals[i][j] = scipy.stats.multivariate_normal.pdf(x, mean=self.mu[j], 
                                                                cov=self.sigsq[j])
        normals *= self.pi.reshape((K, 1)).transpose()
        row_sums = np.sum(normals, axis=1)
        l = np.sum(np.log(row_sums))
        probabilities = normals / row_sums.reshape(N, 1)
        return (l, probabilities)
    
    def m_step(self, data, p_z):
        """ Performs the M-step of the EM algorithm
        data - an NxD pandas DataFrame
        p_z - an NxK numpy ndarray containing posterior probabilities

        returns a dictionary containing the new parameter values
        """
        K = self.k
        N, D = data.shape
        n_j = np.sum(p_z, axis=0) # column sum
        new_pi = n_j / N
        p_z_T = p_z.transpose()
        new_mu = (np.array([np.sum(data * p_z_T[j].reshape(N, 1), axis=0) for j in range(0, K)]) 
                  / n_j.reshape((K, 1)))
        new_sigsq = np.array([np.dot(p_z_T[j], np.sum((data - new_mu[j]) ** 2, axis=1)) 
                             for j in range(0, K)]) / n_j / D # (np.linalg.norm(data - new_mu[j], axis=1) ** 2)
            # new_sigsq = np.array([sum(p_z_T[j].reshape(N, 1) 
            #                           * np.sum((data - new_mu[j]) ** 2, axis=1).reshape(N, 1)) 
            #                      for j in range(0, K)]) / n_j.reshape(K, 1) / D
                # print(np.sum((data - new_mu[1]) ** 2, axis=1))
                # print(p_z_T[1].reshape(N, 1) * np.sum((data - new_mu[1]) ** 2, axis=1).reshape(N, 1))
                # print(p_z_T[1].reshape(N, 1).shape)
                # print(np.sum((data - new_mu[1]) ** 2, axis=1).reshape(N, 1).shape)
                # print((p_z_T[1].reshape(N, 1) * np.sum((data - new_mu[1]) ** 2, axis=1).reshape(N, 1)).shape)
        return {
            'pi': new_pi,
            'mu': new_mu,
            'sigsq': new_sigsq,
        }

    def fit(self, data, *args, **kwargs):
        self.params['sigsq'] = np.asarray([np.mean(data.var(0))] * self.k)
        return super(GMM, self).fit(data, *args, **kwargs)


class CMM(MixtureModel):
    def __init__(self, k, ds):
        """d is a list containing the number of categories for each feature"""
        super(CMM, self).__init__(k)
        self.params['alpha'] = [np.random.dirichlet([1]*d, size=k) for d in ds]
#         print(ds) # Debug use: examine categories

    def e_step(self, data):
        """ Performs the E-step of the EM algorithm
        data - an NxD pandas DataFrame

        returns a tuple containing
            (float) the expected log-likelihood
            (NxK ndarray) the posterior probability of the latent variables
        """
        #print(pd.get_dummies(data.iloc[:, 0], dummy_na=True))
        K = self.k
        N, D = data.shape
        p_x_zapi = np.ones((N, K))
        for d in range(0, D):
            dumb = pd.get_dummies(data.iloc[:, d], dummy_na=True)
            dumb0 = dumb.iloc[:, :-1]
            dumb1 = dumb.iloc[:, -1]
            p_x_zapi *= (np.dot(dumb0, self.alpha[d].transpose()) 
                         + np.dot(dumb1.reshape(N, 1), np.ones((1, K))))
        p_xz_api = np.multiply(p_x_zapi, self.pi)
        p_z = p_xz_api / np.sum(p_xz_api, axis=1).reshape(N, 1)
        log_pi = np.log(self.pi)
#         print(p_xz_api) # DEBUG!
        ll = np.sum(p_z * log_pi) + np.sum(p_z * np.log(p_x_zapi+1e-4)) ######
        return (ll, p_z)

    def m_step(self, data, p_z):
        """ Performs the M-step of the EM algorithm
        data - an NxD pandas DataFrame
        p_z - an NxK numpy ndarray containing posterior probabilities

        returns a dictionary containing the new parameter values
        """
        K = self.k
        N, D = data.shape
        new_pi = np.sum(p_z, axis=0) / N # K by 1
        new_alpha = []
        #print(p_z)
        for d in range(0, D):
            dumb = pd.get_dummies(data.iloc[:, d], dummy_na=True)
            dumb0 = dumb.iloc[:, :-1] # N by nd
            dumb1 = dumb.iloc[:, -1] # N by 1
            alpha_d = np.dot(p_z.transpose(), dumb0)
            denom = np.sum(p_z - np.multiply(p_z, dumb1.reshape(N, 1)), axis=0)
            alpha_d = alpha_d / denom.reshape((K, 1)) # denom[:, np.newaxis] or np.expand_dims(denom, axis=1)
            #print("alpha_d\n", alpha_d)
            new_alpha.append(alpha_d) 
        return {
            'pi': new_pi,
            'alpha': new_alpha,
        }

    @property
    def bic(self):
        """ Computes the Bayesian Information Criterion for the trained model.
            Note: 'n_train' and 'max_ll' set during @see{fit} may be useful
            
            BIC(M) = l - 0.5 * p * log(n), 
            where l is the highest log-likelihood of the data 
                      under the parameters of the current model, 
                  p is the number of adjustable parameters, 
                  and n is the number of data points
        """
        num_alpha = 0
        for array in self.alpha:
            num_alpha += array.shape[0] * (array.shape[1] - 1)
            # ! elements of vector sum to 1: last parameter not free !
        return self.max_ll - 0.5 * (num_alpha + self.k - 1) * np.log(self.n_train)
