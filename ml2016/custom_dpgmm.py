from __future__ import division
import logging

import numpy as np
import numpy.matlib
import math, random, sys
from numpy.linalg import inv, det
from scipy.special import gamma
from scipy.stats import wishart, multivariate_normal

from time import time

from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import KFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from ml2016.util import BaseClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomDPGMM(BaseClassifier):

    def new_cluster(self, X, a, B, c, m):
        """

        x is a np.matrix
        s is a float
        a is a float
        c is a float
        B is a np.matrix
        m is a np.matrix
        """
        d = float(X.shape[0])
        s = float(X.shape[1])
        x_mean = np.mean(X, axis=1)
        m_p = c/(s+c) * m + 1/(s+c) * np.sum(X, axis=1)
        c_p = s + c
        a_p = a + s

        sum_B = np.matlib.zeros((d,d))
        for i in np.arange(s):
            sum_B = sum_B + (X - x_mean) * (X - x_mean).T
        B_p = B + sum_B + s/(a*s + 1) * (x_mean - m) * (x_mean - m).T

        sigma_p = wishart.rvs(df=a_p, scale=inv(B_p))
        mean_p = multivariate_normal.rvs(mean=m_p, cov=inv(c_p * sigma_p)) #transpose omitted
        return mean_p, sigma_p

    def marginal(self, X, a, B, c, m):
        d = float(X.shape[0])
        px1 = (c / (math.pi * (1 + c)))**(d/2)
        px2 = det(B + c/(1+c) * (X - m) * (X - m).T)**(-(a+1)/2) / det(B)**(-a/2)
        px3 = math.exp(np.sum( math.log(gamma( (a+1)/2 + (1 - np.arange(1.0, d))/2 )) - math.log(gamma( a/2 + (1 - np.arange(1.0, d))/2 )) ))
        px = px1 * px2 * px3
        return px

    def discrete_dist(self, psi):
        psi = psi / np.sum(psi)
        rand_i = random.random()
        for i in np.arange(psi.shape[1]):
            psi_cum = np.sum(psi[:i])
            if rand_i < psi_cum:
                index = i
                break
        return index

    def DP_GMM(self, X, T):
        X = X.todense()
        for i in np.arange(X.shape[0]):
            for j in np.arange(X.shape[1]):
                X[i,j] += 1e-3 * random.random()
        print X
        d = float(X.shape[0])
        N = float(X.shape[1])
        c = 1.0/10.0
        a = d
        B = c * d * np.cov(X.T)
        alpha = 1.0
        m = np.mean(X, axis=1)

        m_mean = [None] * int(N)
        m_sigma = [None] * int(N)
        m_n = np.zeros((1, int(N))) # number of points in each cluster

        m_c = np.ones((1, int(N)))
        print B
        print B.shape
        sigma = wishart.rvs(df=a, scale=inv(B))
        m_sigma[0] = sigma
        m_mean[0] = multivariate_normal.rvs(mean=m, cov=inv(c * sigma)) #transpose omitted
        num_clusters = 1
        m_n[0] = N

        num_cluster_list = np.zeros((1, T))
        for t in np.arange(T):
            print t
            for i in np.arange(N):
                m_psi = np.zeros((1, num_clusters + 1))

                # (a) for all clusters with points in them besides x_i
                for j in np.arange(num_clusters):
                    nj = m_n[j]
                    if m_c[i] == j:
                        nj = nj - 1

                    # if only x_i in the cluster, then m_psi prob = 0
                    if nj > 0:
                        mean_j = m_mean[j]
                        sigma_j = m_sigma[j]
                        m_psi_val = multivariate_normal.pdf(x=X[:, i], mean=mean_j, cov=inv(sigma_j)) * nj / (alpha + N - 1)
                        m_psi[j] = m_psi_val

                # (b) for new cluster
                m_psi[num_clusters + 1] = alpha / (alpha + N - 1) * self.marginal(X[:, i], a, B, c, m)

                # (c) normalize m_psi and sample from discrete distribution
                m_psi = m_psi / sum(m_psi)

                # remove this point from the cluster's count
                m_n[m_c[i]] -= 1
                m_c[i] = self.discrete_dist(m_psi) # cluster assignment for x_i
                m_n[m_c[i]] += 1

                if m_c[i] == num_clusters + 1:
                    # generate a new cluster
                    num_clusters = num_clusters + 1
                    mean_p, sigma_p = self.new_cluster(X[:, i], a, B, c, m)
                    m_mean[num_clusters] = mean_p
                    m_sigma[num_clusters] = sigma_p

                # (d) remove clusters with no points, reindex remaining clusters
                m_n = np.zeros((1, N))
                m_c_temp = m_c
                for j in np.arange(num_clusters):
                    indices = []
                    for i, item in m_c:
                        if item == j:
                            indices.append(i)
                    count = float(len(indices))
                    m_n[j] = count
                    X_sub = X[:, indices]
                    mean_p, sigma_p = self.new_cluster(X_sub, a, B, c, m)
                    m_mean[j] = mean_p
                    m_sigma[j] = sigma_p