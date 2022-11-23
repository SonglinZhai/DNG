# -*- coding: utf-8 -*-
# @Author: Songlin.Zhai
# @Date:   2022-05-20
# @Contact: slinzhai@gmail.com

import warnings
warnings.filterwarnings('ignore')

import numpy as np
from sklearn.utils import check_array
from sklearn.linear_model import LassoLarsIC, LinearRegression
from sklearn.decomposition import FastICA
from scipy.optimize import linear_sum_assignment
from joblib import parallel_backend, Parallel, delayed


class Decompose(object):
    """
    Implementation of structure decompostion algorithm
    """
    def __init__(self, max_iter=1000, whiten=True, algorithm='deflation'):
        super().__init__()
        """
        Construct a ICA-based model.
        """
        self.ica = FastICA(
            max_iter=max_iter, whiten=whiten, algorithm=algorithm#'parallel')
        )
    
    def estimate_inherit_factor(self, X, predictors, target, gamma=1.0):
        """Predict inherit factor based on Adaptive Lasso.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        predictors : array-like, shape (n_predictors)
            Indices of predictor variable.
        target : int
            Index of target variable.

        Returns
        -------
        coef : array-like, shape (n_features)
            Coefficients of predictor variable.
        """
        lr = LinearRegression()
        lr.fit(X[:, predictors], X[:, target])
        weight = np.power(np.abs(lr.coef_), gamma)
        reg = LassoLarsIC(criterion="bic")
        reg.fit(X[:, predictors] * weight, X[:, target])
        return reg.coef_ * weight
    
    def estimate_source(self, X, max_iter=1000, adj_mx=True):
        """Fit the model to X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where ``n_samples`` is the number of samples
            and ``n_features`` is the number of features.
        
        max_iter : int, optional (default=1000)
            The maximum number of iterations of FastICA.
        
        adj_mx : bool, optional (default=True)
            Whether estimate the adjacent matrix
            if False, will return the mixing matrix
        Returns
        -------
        source vector and mixing (or adjacent) matrix
        """
        X = check_array(X)
        self.ica.max_iter = max_iter
        source = np.asarray(self.ica.fit_transform(X))

        if adj_mx:
            W_ica = self.ica.components_
            # obtain a permuted W_ica
            _, col_index = linear_sum_assignment(1 / np.abs(W_ica))
            PW_ica = np.zeros_like(W_ica)
            PW_ica[col_index] = W_ica
            # obtain a vector to scale
            D = np.diag(PW_ica)[:, np.newaxis]
            # estimate an adjacency matrix
            W_estimate = PW_ica / D
            B_estimate = np.eye(len(W_estimate)) - W_estimate
        else: B_estimate = self.ica.mixing_
        return source, B_estimate

    def estimate_adj_mx_parallel(self, X, node_order=None, backend='loky', n_jobs=4):
        """Estimate adjacency matrix by node order.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        
        backend : str
            - 'loky': single-host, process-based parallelism (used by default),
            - 'threading': single-host, thread-based parallelism,
            - 'multiprocessing': legacy single-host, process-based parallelism.

        Returns
        -------
        self : np.array
            adjacency matrix between features
        """
        # Parallel compute the adjacency matrix
        n_feats = X.shape[1]
        if node_order == None:
            node_order = list(range(0, n_feats))
        with parallel_backend(backend=backend, n_jobs=n_jobs):
            res = Parallel()(delayed(self.estimate_inherit_factor)\
                (X,node_order[:i],node_order[i])\
                for i in range(1, n_feats))
        B = np.zeros([n_feats, n_feats], dtype="float64")
        for i in range(1, n_feats):
            B[node_order[i], node_order[:i]] = res[i-1]
        return B
    
    def estimate_adj_mx_deflation(self, X, node_order=None):
        """Estimate adjacency matrix by node order.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        self : np.array
            adjacency matrix between features
        """
        n_feats = X.shape[1]
        if node_order == None:
            node_order = list(range(0, n_feats))
        B = np.zeros([n_feats, n_feats], dtype="float64")
        for i in range(1, n_feats):
            target = node_order[i]
            predictors = node_order[:i]
            # target is exogenous variables if predictors are empty
            if len(predictors) != 0:
                B[target, predictors] = self.estimate_inherit_factor(X, predictors, target)
        return B


if __name__ == "__main__":
    # node direction: x0 --> x1 --> x3 --> x2
    #
    x0 = np.random.uniform(size=256)
    x1 = 2.0 * x0 + np.random.uniform(size=256)
    x3 = 4.0 * x1 + np.random.uniform(size=256)
    x2 = 3.0 * x3 + np.random.uniform(size=256)
    #
    x = np.zeros(shape=(4, 256))
    x[0] = x0
    x[1] = x1
    x[2] = x3
    x[3] = x2
    
    model = Decompose()
    adj_mx = model.estimate_adj_mx_deflation(x.transpose(1,0))
    print(adj_mx)

    adj_mx = model.estimate_adj_mx_parallel(x.transpose(1,0))
    print(adj_mx)

    last_adj = model.estimate_inherit_factor(x.transpose(1,0), [0,1,2], -1)
    print(last_adj)
    
    source, mix_mx = model.estimate_source(x.transpose(1,0))
    mix_mx = np.linalg.pinv(np.identity(mix_mx.shape[0])-mix_mx)
    print(mix_mx)
    print(mix_mx.shape)
    print(np.matmul(mix_mx, source.T))
    print(np.matmul(mix_mx, source.T).shape)
