import numpy as np
import scipy as sp
import pymc3 as pm
from sklearn.preprocessing import StandardScaler

import config

# import theano
# import theano.tensor as tt


class Model(object):
    """Base class on top of which every concrete model is built."""
    def __init__(self, X, Y, **kwargs):
        self.X = X
        self.Y = Y

        self.n_sample = 1000

        self.N = X.shape[0]
        self.DX = X.shape[1]

        self.create_model()

    def create_model(self):
        raise NotImplementedError

    def infer_model(self, n=10000, inference=None):
        """Based on a PyMC3 model, use ADVI to infer the posterior given the data."""
        with self.model:
            self.inference = inference
            if not inference:
                self.inference = pm.ADVI()
            approx = pm.fit(n=n, method=self.inference, progressbar=False)
            self.trace = approx.sample(draws=self.n_sample)

    def eval_model(self):
        """Computes the log probabilities for each sample of the parameters from the posterior."""
        model = self.model
        trace = self.trace

        data_dict = {'X': self.X, 'Y': self.Y}

        variable_names = list(map(str, model.vars))

        logxp = model.X.logp
        logyp = model.Y.logp
        logps = np.zeros(self.n_sample)
        for i in range(self.n_sample):
            point = trace.point(i)
            point.update(data_dict)
            logps[i] = logxp(point) + logyp(point)

        return logps


class Causal(Model):
    """Model X as multivariate normally distributed and Y via a probabilistic linear regression from X to Y."""
    def __init__(self, X, Y, **kwargs):
        super().__init__(X, Y, **kwargs)

    def create_model(self):
        self.model = pm.Model()
        self.model_x()
        self.model_w()
        self.model_y()

    def model_x(self):
        """X is normally distributed."""
        with self.model:
            mux = pm.Normal('mux', 0, 1)
            sdx = pm.Lognormal('sdx', 0, 1)

            X = pm.Normal('X', mu=mux, sd=sdx, observed=self.X)

    def model_w(self):
        """Weights from X to Y are normally distributed."""
        with self.model:
            sdw = pm.Lognormal('sdw', 1)
            w = pm.Normal('w', 0, sdw, shape=[self.DX, 1])

    def model_y(self):
        """Model Y via regression onto X."""
        with self.model:
            muy = pm.Deterministic('muy',
                                   pm.math.dot(self.model.X, self.model.w))
            sdy = pm.Lognormal('sdy', 1)
            Y = pm.Normal('Y', mu=muy, sd=sdy, observed=self.Y)


class Confounded(Model):
    """Model (X, Y) by doing PPCA, i.e. jointly inferring a latent variable Z and doing linear regression from Z towards X,Y."""
    def __init__(self, X, Y, DZ, **kwargs):
        self.DZ = DZ

        super().__init__(X, Y, **kwargs)

    def create_model(self):
        self.model = pm.Model()

        self.model_z()
        self.model_w()
        self.model_x()
        self.model_y()

    def model_z(self):
        """Z is normally distributed."""
        with self.model:
            muz = pm.Normal('muz', 0, 10)
            sdz = pm.Lognormal('sdz', 0, 1)
            Z = pm.Normal('Z', mu=muz, sd=sdz, shape=[self.N, self.DZ])

    def model_w(self):
        """The weights from Z to (X,Y) are normally distributed."""
        with self.model:
            sdw = pm.Lognormal('sdw', 1)
            wx = pm.Normal('wx', 0, sdw, shape=[self.DZ, self.DX])
            wy = pm.Normal('wy', 0, sdw, shape=[self.DZ, 1])

    def model_x(self):
        """Model X via regression onto Z."""
        with self.model:
            mux = pm.Deterministic('mux',
                                   pm.math.dot(self.model.Z, self.model.wx))
            sdx = pm.Lognormal('sdx', 0, 1)
            X = pm.Normal('X', mu=mux, sd=sdx, observed=self.X)

    def model_y(self):
        """Model X via regression onto Z."""
        with self.model:
            muy = pm.Deterministic('muy',
                                   pm.math.dot(self.model.Z, self.model.wy))
            sdy = pm.Lognormal('sdy', 0, 1)
            Y = pm.Normal('Y', mu=muy, sd=sdy, observed=self.Y)


def compare_models(X,
                   Y,
                   DZ,
                   Caus=Causal,
                   Conf=Confounded,
                   normalize=True,
                   advi_iter=config.advi_iter):
    """Runs both both the X -> Y and X <- Z -> Y models on the data and returns the scores obtained for both."""

    if normalize:
        X = StandardScaler().fit(X).transform(X)
        Y = Y.reshape(-1, 1)
        Y = StandardScaler().fit(Y).transform(Y)

    Cs = Caus(X, Y)
    Cf = Conf(X, Y, DZ)

    Cs.infer_model(n=advi_iter)
    Cf.infer_model(n=advi_iter)

    cse = Cs.eval_model()
    cfe = Cf.eval_model()

    cs_mean = sp.special.logsumexp(cse) - np.log(Cs.n_sample)
    cf_mean = sp.special.logsumexp(cfe) - np.log(Cf.n_sample)

    return cs_mean, cf_mean
