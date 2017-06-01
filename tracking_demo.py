import numpy as np
import scipy.io as io
# import sample_lds
from kalman_filter import kalman_filter, kalman_smoother
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab


"""
Run Kevin Murphy's particle example explained here:
http://www.cs.ubc.ca/~murphyk/Software/Kalman/kalman.html
"""


def run_tracking_demo(filename, plot=True):
    ss = 4  # state size
    os = 2  # observation size
    A = np.matrix('1 0 1 0; 0 1 0 1; 0 0 1 0; 0 0 0 1')
    C = np.matrix('1 0 0 0; 0 1 0 0')
    Q = 0.1 * np.identity(ss)
    R = 1 * np.identity(os)
    init_x = np.matrix([10, 10, 1, 0]).T
    init_V = 10 * np.identity(ss)
    T = 15

    # Generate new data
    # np.random.seed(0)
    # x, y = sample_lds(A, C, Q, R, init_x, T)

    # Use mat file to reproduce Murphy's example
    mat = io.loadmat(filename)
    x = mat['x']
    y = mat['y']

    xfilt, Vfilt, VVfilt, loglik = kalman_filter(y, A, C, Q, R, init_x, init_V)
    xsmooth, Vsmooth, VVsmooth, loglik = kalman_smoother(y, A, C, Q, R, init_x, init_V)

    dfilt = x[[0,1],:] - xfilt[[0,1],:]
    mse_filt = np.sqrt(np.sum(dfilt**2))
    print "mse_filt: %f" % mse_filt

    dsmooth = x[[0,1],:] - xsmooth[[0,1],:]
    mse_dsmooth = np.sqrt(np.sum(dsmooth**2))
    print "mse_dsmooth: %f" % mse_dsmooth

    # Plot 
    if plot:
        f, (ax1, ax2) = plt.subplots(1, 2, sharex=True,sharey=True)
        ax1.plot(x[0,:], x[1,:], 'ks-')
        ax1.plot(xfilt[0,:], xfilt[1,:], 'rx:')
        ax1.set_title('Kalman Filter')
        plot_2d_contours(ax1, xfilt, Vfilt)

        ax2.plot(x[0,:], x[1,:], 'ks-')
        ax2.plot(xsmooth[0,:], xsmooth[1,:], 'rx:')
        ax2.set_title('Kalman Smoother')
        plot_2d_contours(ax2, xsmooth, Vsmooth)

        plt.show()


def plot_2d_contours(ax, xfit, vfit):
    delta = 0.025
    [_,d,T] = vfit.shape 
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x = np.arange(xlim[0],xlim[1], delta)
    y = np.arange(ylim[0],ylim[1], delta)
    X, Y = np.meshgrid(x, y)
    for t in range(T):
        s2d = np.sqrt(vfit[0:2,0:2,t])
        x0 = xfit[0,t] 
        x1 = xfit[1,t] 
        Z = mlab.bivariate_normal(X, Y,sigmax=s2d[0,0], sigmay=s2d[1,1], mux=x0, muy=x1, sigmaxy=s2d[0,1])
        ax.contour(X, Y, Z, 3)


if __name__ == '__main__':
    run_tracking_demo('data/murphy_tracking_data.mat', plot=False)