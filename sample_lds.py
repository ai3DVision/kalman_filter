import numpy as np


def sample_lds(A, C, Q, R, init_x, T):

    # x(t+1) = A*x(t) + w(t),  w ~ N(0, Q),  x(0) = init_state
    # y(t) =   C*x(t) + v(t),  v ~ N(0, R)
    #
    # Input:
    # A(:,:,i) - the transition matrix for the i'th model
    # C(:,:,i) - the observation matrix for the i'th model
    # Q(:,:,i) - the transition covariance for the i'th model
    # R(:,:,i) - the observation covariance for the i'th model
    # init_x(:,i) - the initial mean for the i'th model
    # T - the num. time steps to run for
    #
    # Output:
    # x(:,t)    - the hidden state vector at time t.
    # y(:,t)    - the observation vector at time t.

    os, ss = C.shape
    state_noise_samples = np.random.multivariate_normal(mean=np.zeros([ss]), cov=Q, size=T).T
    obs_noise_samples = np.random.multivariate_normal(mean=np.zeros([os]), cov=R, size=T).T

    x = np.zeros([ss, T])
    y = np.zeros([os, T])

    x[:, 0] = init_x.reshape(-1)
    y[:, 0] = np.matmul(C, x[:, 0]) + obs_noise_samples[:, 0]

    for t in xrange(1, T):
        x[:, t] = np.matmul(A, x[:, t-1]) + state_noise_samples[:, t]
        y[:, t] = np.matmul(C, x[:, t])  + obs_noise_samples[:, t]

    return x, y
