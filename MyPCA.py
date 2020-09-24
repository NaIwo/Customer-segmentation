import sys
from singledispatchmethod import singledispatchmethod
import numpy as np

class MyPCA():
    def __init__(self, n_components):
        if n_components > 1 and type(n_components) is not int:
            raise NotImplementedError('Cannot interpret given value.')
        self.n_components = n_components
        self.U = None
        self.s = None
        self.Vt = None
        self.explained_variance_ratio_ = None
        self.epsilon = sys.float_info.epsilon
        
    def fit(self, X, y = None):
        local_mean = X.mean(axis = 0)
        if (local_mean > self.epsilon).all():
            X = X - local_mean
        self.U, self.s, self.Vt = np.linalg.svd(X)
        
    def __compute_explained_variance_ratio_(self, data):
        explained_variance_ = (self.s ** 2) / (data.shape[0] - 1)
        total_var = sum(explained_variance_)
        explained_variance_ratio_ = explained_variance_ / total_var
        return explained_variance_ratio_
    
    @singledispatchmethod
    def __compute_W(self, arg, data):
        raise TypeError("Wrong type has been passing. Given type: {}, but avaliable {} or {}. ".format(type(arg), int, float))
    
    @__compute_W.register
    def _(self, arg : int, data):
        self.explained_variance_ratio_ = self.__compute_explained_variance_ratio_(data)[:arg]
        return self.Vt.T[:,:arg]
    
    @__compute_W.register
    def _(self, arg : float, data):
        explained_variance_ratio_ = self.__compute_explained_variance_ratio_(data)
        cum_sum = np.cumsum(explained_variance_ratio_)
        
        self.n_components = np.argmax(cum_sum >= self.n_components) + 1
        
        self.explained_variance_ratio_ = explained_variance_ratio_[:self.n_components]
        return self.Vt.T[:,:self.n_components]
        
    def transform(self, X, y = None):
        
        W = self.__compute_W(self.n_components, X)
        return np.array(X.dot(W))
            
    
    def fit_transform(self, X, y = None):
        self.fit(X)
        return self.transform(X)