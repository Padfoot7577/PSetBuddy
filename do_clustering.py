import random
import time

import numpy as np
import scipy.stats
import pandas as pd


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

