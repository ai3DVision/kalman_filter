import numpy as np
from learn_kalman import learn_kalman
import scipy.io as io
from sample_lds import sample_lds
from kalman_filter import kalman_smoother

"""
Run Kevin Murphy's particle example explained here:
http://www.cs.ubc.ca/~murphyk/Software/Kalman/kalman.html
"""


def run_learning_demo(filename):
    ss = 4  # state size
    os = 2  # observation size
    A = np.matrix('1 0 1 0; 0 1 0 1; 0 0 1 0; 0 0 0 1')
    C = np.matrix('1 0 0 0; 0 1 0 0')
    Q = 0.1 * np.identity(ss)
    R = 1 * np.identity(os)
    init_x = np.matrix([10, 10, 1, 0]).T
    init_V = 10 * np.identity(ss)
    T = 100

    # Use mat file to reproduce Murphy's example
    # (Or uncomment below to generate new data)
    mat = io.loadmat(filename)
    x = mat['x']
    y = mat['y']

    # Generate new data
    # np.random.seed(0)
    # x, y = sample_lds(A, C, Q, R, init_x, T)

    # Initializing the params to sensible values is crucial.
    # Here, we use the true values for everything except F and H,
    # which we initialize randomly (bad idea!)
    # Lack of identifiability means the learned params. are often far from the true ones.
    # All that EM guarantees is that the likelihood will increase.

    # Use murphy data for reproducibility
    A1 = mat['F1']
    C1 = mat['H1']

    # Or generate new starting values
    # A1 = np.random.randn(ss,ss)
    # C1 = np.random.randn(os,ss)

    Q1 = Q
    R1 = R
    initx1 = init_x
    initV1 = init_V
    max_iter = 10
    F2, H2, Q2, R2, initx2, initV2, LL = learn_kalman(y, A1, C1, Q1, R1, initx1, initV1, max_iter=max_iter, smoother=kalman_smoother)


if __name__ == '__main__':
    run_learning_demo('data/murphy_learning_data.mat')
