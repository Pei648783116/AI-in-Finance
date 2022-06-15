# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 16:43:47 2019

@author: ThinkPad
Extend the TimeSeriesSplit object and override the split method
"""
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples

class fwTimeSeriesSplit(TimeSeriesSplit):        
    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like, with shape (n_samples,)
            Always ignored, exists for compatibility.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        n_folds = n_splits + 1
        if n_folds > n_samples:
            raise ValueError(
                ("Cannot have number of folds ={0} greater"
                 " than the number of samples: {1}.").format(n_folds,
                                                             n_samples))
        indices = np.arange(n_samples)
        test_size = (n_samples // n_folds)
        test_starts = range(test_size + n_samples % n_folds,
                            n_samples, test_size)
        lookback = test_starts[0]
        for test_start in test_starts:
            if self.max_train_size and self.max_train_size < test_start:
                if test_start < lookback:
                    lookback = test_start-1
                yield (indices[(test_start - lookback):test_start],
                       indices[test_start:test_start + test_size])
            else:
                yield (indices[test_start-lookback:test_start],
                       indices[test_start:test_start + test_size])
