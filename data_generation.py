import numpy as np

import os
import argparse

import itertools


class Generator():
    def __init__(self, N, DX, DZ, distributions):
        self.N = N
        self.DX = DX
        self.DZ = DZ
        self.distributions = distributions

    def _generate_cf(self):
        """Generates confounded data. Uses the distributions in self.distributions
        to sample from the desired distributions.
        First generates the confounder, then the weights and then X, Y from these."""

        Z = self.distributions['z'](size=[self.N, self.DZ])

        wx = self.distributions['w'](size=[self.DX, Z.shape[1]])
        wy = self.distributions['w'](size=[1, Z.shape[1]])

        epsx = np.random.randn(self.N, self.DX)
        epsy = np.random.randn(self.N, 1)

        X = np.dot(Z, wx.T) + epsx
        Y = np.dot(Z, wy.T) + epsy

        return X, Y, Z

    def _generate_cs(self):
        """Generates causal data. Uses the distributions in self.distributions
        to sample from the desired distributions.
        First generates X, then weights and Y from these. Z is generated independently
        of everything else."""
        X = self.distributions['x'](size=[self.N, self.DX])
        w = self.distributions['w'](size=[1, X.shape[1]])
        eps = np.random.randn(self.N, 1)

        Y = np.dot(X, w.T) + eps
        Z = np.random.randn(self.N, self.DZ)

        return X, Y, Z

    def generate(self):
        """Runs both the internal generators and checks that the dimensions are the same."""
        X_cf, Y_cf, Z_cf = self._generate_cf()
        X_cs, Y_cs, Z_cs = self._generate_cs()

        assert X_cs.shape == X_cf.shape
        assert Y_cs.shape == Y_cf.shape
        assert Z_cs.shape == Z_cf.shape

        return [(X_cf, Y_cf, Z_cf), (X_cs, Y_cs, Z_cs)]


def generate_data(n_samples=200, N=500, dx=(1, 5, 10), dz=(5, ), data='dr'):
    """Generates samples of a specific size from the desired dimensions of X and Z.
    Inputs:
    n_samples: Number of samples to generate for each pair of dimensions and combination
    of distributions.
    N: Number of data points in each sample.
    dx: List of dimensions of X.
    dz: List of dimensions of Z.
    data: Data for the decision rate (dr) or heatmap (heat) plots? Defaults to dr."""
    distributions = [
        ('n', lambda size: np.random.normal(0, 1, size=size)),
        ('u', lambda size: np.random.uniform(-1, 1, size=size)),
        ('ln', lambda size: np.random.lognormal(0, 1, size=size)),
        ('lap', lambda size: np.random.laplace(0, 1, size=size)),
    ]
    for DX, DZ in itertools.product(dx, dz):
        for ns in range(n_samples):
            # We sample look only at the cases where all the
            # distributions are the same. Other combinations give
            # similar results.
            for dist in distributions:
                dist_label, dist = dist
                dists = dict(
                    zip(('z', 'x', 'w'), [dist, dist, distributions[0][1]]))
                gen = Generator(N, DX, DZ, dists)
                samples = gen.generate()
                for i, (X, Y, Z) in enumerate(samples):
                    save_dir = f'data/synthetic/{data}/{dist_label}/'
                    if i == 0:
                        save_dir += 'cf/'
                    else:
                        save_dir += 'cs/'
                    save_file = save_dir + f'n{N}_dx{DX}_dz{DZ}[{ns}].csv'
                    XY = np.hstack([X, Y])
                    assert XY.shape == (N, DX + 1)
                    os.makedirs(save_dir, exist_ok=True)
                    np.savetxt(save_file, XY, delimiter=',')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', type=str, default='dr')
    args = parser.parse_args()
    data = args.data

    np.random.seed(1)

    # We need different data for the two different plots.
    if data == 'dr':
        dx = (1, 3, 6, 9, 12)
        dz = (3, )
        n_samples = 1000
    elif data == 'heat':
        dx = (2, 4, 6, 8, 10)
        dz = (2, 4, 6, 8, 10)
        n_samples = 200
    else:
        raise NotImplementedError

    generate_data(n_samples=n_samples, dx=dx, dz=dz, data=data)
