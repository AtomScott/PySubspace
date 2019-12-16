"""
PCA by minimizing the Quadratic Discriminant Function
====================================================================

This example plots an animated gif showing how we can perform principle 
component analysis (PCA) by minimizing the Quadratic Discriminant Function.
"""

from scipy import linalg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
from matplotlib.animation import FuncAnimation

def dataset_fixed_cov():
    '''Generate 1 Gaussians samples with the same covariance matrix'''
    n, dim = 300, 2
    np.random.seed(0)
    C = np.array([[0., -0.23], [0.83, .23]])
    X = np.dot(np.random.randn(n, dim), C) + [2., 2.]
    return X

def calc_J(X, vec):
    j = 0
    m = X.mean(axis=0)
    for v in X:
        j += (v-m).T @ vec
    return j

def normalize(vec):
    return vec / np.linalg.norm(vec)

def is_normalized(vec):
    return np.linalg.norm(vec) == 1.

def make_vector(deg):
    return np.array([np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))])


if __name__ == "__maian__":
    # Generate dataset
    X = dataset_fixed_cov()

    # Calculate a bad estimation of phi
    phi = make_vector(-45)
    assert is_normalized(phi)

    # Calculate the direction that maximises the variance
    # with eigen decomposition
    eig_vals, eig_vecs = np.linalg.eig(X.T@X + X.mean())
    target_phi = eig_vecs[0]
    assert is_normalized(target_phi)
    
    fig, ax = plt.subplots()
    J = calc_J(X, phi)
    ln, = ax.quiver(*[0,0], *phi.T, scale=2)

    def init():
        # Plot direction of Variance
        ax.quiver(*[0,0], *phi.T, scale=2)
        # Plot datset
        ax.scatter(*X.T, c='blue', label='Target dataset')
        # Plot mean
        ax.scatter(*X.mean(axis=0), c='red', label='Mean')
        # Plot origin
        ax.scatter(*[0,0], c='black', label='Origin')
        plt.legend()
        plt.title(f'J={J}')
        return ln,

    def update(frame):
        if frame % 2 ==0:
            ln.set_data(*target_phi)
        else:
            ln.set_data(*phi)
        return ln,    

    ani = FuncAnimation(fig, update, frames=np.linspace(0, 2*np.pi, 128),
                    init_func=init, blit=True)
    plt.show()