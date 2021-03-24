#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation

from scipy.stats import wishart
from numpy.random import uniform, choice, multivariate_normal

from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

from em_kmeans import Gm_EM, draw_ellipse

import argparse
from progress.bar import Bar

class DemoEM(object):
    def __init__(self, data, K, init, step_t, frames):
        self.step_t = step_t
        self.em = Gm_EM(data, K, init)
        self.clusters = labels

        self.bar = Bar('Processing', max=frames)

        # Setup the figure and axes...
        self.fig, self.ax = plt.subplots()
        # Then setup FuncAnimation.
        self.ani = FuncAnimation(self.fig, self.update, frames=frames, interval=200,
                                 blit=False, repeat=False)

    def draw(self):
        """drawing of the scatter plot."""
        self.ax.scatter(self.em.data[:,0], self.em.data[:,1], s=10, alpha=0.2, c=self.clusters)
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.axis([self.em.xmin, self.em.xmax, self.em.ymin, self.em.ymax])

        self.ax.scatter(self.em.mu[:,0], self.em.mu[:,1],
                                    s=50, cmap="plasma", c=range(self.em.K),
                                   edgecolor = 'black')

        for k in range(self.em.K):
            ell, transf = draw_ellipse(mu=self.em.mu[k,:], cov=self.em.sigma[k,:,:], confint=80)
            ell.set_transform(transf + self.ax.transData)
            self.ax.add_patch(ell)


    def update(self, i):
        """Update the scatter plot"""
        if i:
            self.em.step(self.step_t)
            self.bar.next()

        self.ax.clear()
        self.draw()

def gen_data(K=3, Nk=[400, 300, 125, 100, 75], random_state=123, normalize=True):
    np.random.seed(random_state)

    # for generating random centers
    center = np.array((0,0))
    dispersion = 10

    mu_k = multivariate_normal(center, np.identity(2)*dispersion, K)
    sd_k = uniform(0.1, 2, size=K)
    data, labels = make_blobs(n_samples=Nk[:K], n_features=2, centers=mu_k, cluster_std=sd_k)

    # rotate data in each cluster using random covariance matrices..
    sigmas = wishart.rvs(2, scale=np.identity(2), size=K) + np.identity(2)

    x = data[labels==0] @ sigmas[0]
    for k in range(1,K):
        x = np.vstack((x, data[labels==k] @ sigmas[k]))
    labels = np.sort(labels)

    if normalize:
        x = StandardScaler().fit_transform(x)

    return x, labels

parser = argparse.ArgumentParser()
parser.add_argument("--normalize", help="normalize generated data", action="store_false")
parser.add_argument('--data_clusters', default=3, help="nr of clusters in generated data", type=int)
parser.add_argument('--em_clusters', default=3, help="nr. of clusters to be discovered by EM", type=int)
parser.add_argument('--random_state', default=123, help="random_seed", type=int)
parser.add_argument('--nr_frames', default=30, help="how many frames", type=int)
parser.add_argument('--step_t', default=1, help="how many EM iterations in each frame", type=int)
parser.add_argument('--fname', default="em.mp4")
parser.add_argument('--init_mode', default="adversarial",
    help="random initialization, choices are 'random_points' or 'adversarial' or 'kmeans'")
args = parser.parse_args()

assert args.data_clusters <= 5, f'maximum nr. of true clusters is 5, you gave {args.data_clusters}'

x, labels = gen_data(K=args.data_clusters, random_state=args.random_state, normalize=args.normalize)


print(args)
demo = DemoEM(x, args.em_clusters, args.init_mode, args.step_t, args.nr_frames)
demo.ani.save(args.fname)
demo.bar.finish()
