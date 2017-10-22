""" A Kernel Ridge Regression module.
"""
from __future__ import print_function
import random
import numpy as np
from scipy.optimize import minimize, fmin_bfgs

class KernelRegression(object):
    """ Class object for performing kernel ridge regression including
        nested k-fold cross-validation for error estimation and
        optimization of hyperparameters.

        -- Input --

        data_values: list
            Values of all known data.

        k: int
            k-fold cross-validation.

        feature_matrix: array
            Matrix with data features.
            If no specific indices are given for training (known data) and
            testing (unknown data), it is assumed that unknown data are
            appended in the end.
            Not necessary if similarity matrix is provided.

        similarity_matrix: array
            Symmetric matrix with data similarities.
            If no specific indices are given for training (known data) and
            testing (unknown data), it is assumed that unknown data are
            appended in the end.

        comp: comparator object
            Object that calculates similarity with get_similarity.
            Not necessary if a similarity matrix is provided.

        stratify: boolean
            Data are distributed evenly (as opposed to randomly) into folds 
            for cross-validation. Assumes that data are sorted after
            ascending value.

        nested: boolean
            Optimize hyperparameter via nested cross-validation.
    """
    def __init__(self, data_values, k=5,
                 feature_matrix=None, similarity_matrix=None,
                 comp=None, stratify=True, nested=True, bias='avg'):
        self.data_values = np.array(data_values)
        self.k = k
        self.feature_matrix = feature_matrix
        self.similarity_matrix = similarity_matrix
        self.comp = comp
        self.sigma = None

        self.nested = nested
        self.stratify = stratify
        assert similarity_matrix is not None or comp is not None

        self.kernel_matrix = None
        if feature_matrix != None:
            self.n_data = len(feature_matrix)
        else:
            self.n_data = len(similarity_matrix)
        self.bias = bias
        self.alpha1 = None
        self.alpha2 = None

    def _kernel_(self, sigma, s):
        """ Kernel.
        """
        c = -0.5/sigma**2
        return np.exp(c*s)

    def set_kernel_matrix(self, sigma=None, idx=None):
        """ Construct a symmetric matrix with all kernel values.
        """
        if sigma is None:
            sigma = self.sigma
        if self.similarity_matrix is None:
            similarity_matrix = np.zeros((self.n_data, self.n_data))
            for no, i in enumerate(idx):
                similarity_matrix[i][idx[no+1:]] = np.apply_along_axis(self.comp.get_similarity,
                                                                 0,
                                                                 [self.feature_matrix[idx[no+1:]]],
                                                                 self.feature_matrix[i])
            self.similarity_matrix = similarity_matrix + similarity_matrix.T

        self.kernel_matrix = self._kernel_(sigma, self.similarity_matrix)

    def _update_kernel_matrix_(self,idx):
        """ Updates the current kernel matrix
        """
        c = -0.5/self.sigma**2
        f = lambda x: np.exp(c*x)
        mask = np.ones(len(self.similarity_matrix), dtype=bool)
        mask[idx] = False
        k_row = f(self.similarity_matrix[idx][mask])
        self.kernel_matrix = np.insert(self.kernel_matrix, idx, k_row, axis=0)
        k_col = np.insert(k_row, idx, 1.)
        self.kernel_matrix = np.insert(self.kernel_matrix, idx, k_col, axis=1)


    def add_data(self, feature, value=None, sort=True):
        """ Add data and update kernel matrix.
        """
        assert self.feature_matrix is not None
        
        if sort == True:
            idx = np.searchsorted(self.data_values, value)
        else:
            idx = self.n_data

        if value is not None:
            self.data_values = np.insert(self.data_values, idx, value)
            self.alpha1 = None
            self.alpha2 = None

        #s_row = np.apply_along_axis(self.comp.get_similarity,
        #                            0,
        #                            [self.feature_matrix],
        #                            feature)

        s_row = np.zeros(len(self.feature_matrix))
        for i in range(len(self.feature_matrix)):
            s_row[i] = self.comp.get_similarity(self.feature_matrix[i],feature)


        self.similarity_matrix = np.insert(self.similarity_matrix, idx,
                                           s_row, axis=0)
        s_col = np.insert(s_row, idx, 0.)
        self.similarity_matrix = np.insert(self.similarity_matrix, idx,
                                           s_col, axis=1)

        if self.kernel_matrix is not None:
            self._update_kernel_matrix_(idx)

        self.feature_matrix = np.insert(self.feature_matrix, idx, feature, axis=0)

        self.n_data += 1

    def remove_unknown_data(self):
        """ Remove all unknown data (data with no value).
        """
        idx = range(len(self.data_values), self.n_data)
        if len(idx) == 0:
            return

        self.feature_matrix = np.delete(self.feature_matrix, idx, axis=0)
        self.similarity_matrix = np.delete(self.similarity_matrix, idx, axis=0)
        self.similarity_matrix = np.delete(self.similarity_matrix, idx, axis=1)
        self.kernel_matrix = np.delete(self.kernel_matrix, idx, axis=0)
        self.kernel_matrix = np.delete(self.kernel_matrix, idx, axis=1)

        self.n_data -= len(idx)

    def predict_values(self, new_features=[], train_idx=None, test_idx=None, sigma=None, L=1e-5, bias='avg'):
        """ Prediction of new data values.

            -- Input --

            new_features: list
                Predict the values of data with these features.

            train_idx: list
                Training data indices (default: all known data).

            test_idx: list
                Test data indices (default: all data not used for
                training).

            sigma: float
                Gaussian kernel width. Uses optimised sigma from
                cross-validation by default.

            L: float
                Regularization parameter.
            
            bias: str:{'avg','high','low'} or int
                Bias for the regression.
                'avg':  The average value of the training data will be used as bias.
                'high': The largest value in the training data will be used as bias.
                'low':  The smallest value in the training data will be used as bias. 
                (default: 'avg')
        """
        for f in new_features:
            self.add_data(f, sort=False)

        if sigma is None or sigma == self.sigma:
            sigma = self.sigma
            new_sigma = False
        else:
            self.sigma = sigma
            new_sigma = True
        if train_idx is None:
            train_idx = range(len(self.data_values))
        if test_idx is None:
            test_idx = [i for i in range(self.n_data) if i not in train_idx]

        train_values = self.data_values[train_idx]

        if self.kernel_matrix is None or new_sigma:
            all_idx = np.hstack([train_idx, test_idx])
            self.set_kernel_matrix(sigma, all_idx)

        # Prediction (with mean value reference)
        train_avg = np.mean(train_values)
        train_values -= train_avg

        # get alphhas
        C = np.array([self.kernel_matrix[i][train_idx] for i in train_idx])
        kT = np.array([self.kernel_matrix[i][train_idx] for i in test_idx])
        if (self.alpha1 is not None) and (len(self.data_values) == len(train_idx)):
            alpha1 = self.alpha1
            alpha2 = self.alpha2
        else:
            alpha1 = np.linalg.solve(C+np.eye(len(C))*L,train_values)
            alpha2 = np.linalg.solve(C+np.eye(len(C))*L,kT.T)
        if (self.alpha1 is None) and (len(self.data_values) == len(train_idx)):
            self.alpha1 = alpha1
            self.alpha2 = alpha2

        # Predicted value
        if bias != 'avg':
            biased_tv = train_values + train_avg
            if bias == 'high':
                bias = np.max(biased_tv)
            elif bias == 'low':
                bias = np.min(biased_tv)
            biased_tv -= bias
            alpha_b = np.linalg.solve(C+np.eye(len(C))*L,biased_tv)
            pred_values = np.dot(kT,alpha_b) + bias
        else:
            pred_values = np.dot(kT,alpha1) + train_avg

        # Predicted error
        kTalpha2 = np.sum((kT*alpha2.T),-1)
        theta0 = np.dot(train_values,alpha1)/float(len(train_idx))
        pred_errors = np.sqrt(abs(theta0*(1-kTalpha2)))

        self.remove_unknown_data()
        return pred_values, pred_errors

    def cross_validation(self, sigma0, L=1e-5, train_idx=None, opt_sigma=True, bias='avg'):
        """ Performs k-fold cross-validation.

            -- Input --

            sigma0: float
                Gaussian kernel width (starting guess if opt_sigma=True).

            L: float
                Regularization parameter.

            train_idx: list
                Training data indices (default: all known data).

            opt_sigma: boolean
                Optimize sigma via cross-validation.

            bias: str:{'avg','high','low'} or int
                Bias for the regression.
                'avg':  The average value of the training data will be used as bias.
                'high': The largest value in the training data will be used as bias.
                'low':  The smallest value in the training data will be used as bias.
                (default: 'avg')  
        """
        if train_idx is None:
            train_idx = range(self.n_data)

        # Create k subsets
        if self.stratify:
            # Distribute data evenly into subsets (stratification)
            subsets = []
            n_buckets = int(len(train_idx)/self.k)
            buckets = np.array_split(train_idx, n_buckets)
            for h in range(n_buckets):
                random.shuffle(buckets[h])
            buckets = np.hstack(buckets)
            for fold in range(self.k):
                subsets.append(list(buckets[fold::self.k]))
        else:
            random.shuffle(train_idx)
            subsets = np.array_split(train_idx, self.k)
            subsets = [list(i) for i in subsets]

        # Test loop
        if self.nested:
            MAE, sigma, apv, ape = self._nested_test_loop_(sigma0, L, subsets,
                                                           opt_sigma, bias)
        else:
            MAE, sigma, apv, ape = self._test_loop_(sigma0, L, subsets,
                                                    opt_sigma, bias)

        all_subsets = np.hstack(subsets)
        all_pred_values = [x for (y, x) in sorted(zip(all_subsets, apv))]
        all_pred_errors = [x for (y, x) in sorted(zip(all_subsets, ape))]
        self.kernel_matrix = None
        self.sigma = sigma
        return MAE, sigma, all_pred_values, all_pred_errors

    def _nested_test_loop_(self, sigma0, L, subsets, opt_sigma, bias):
        """ k-fold cross-validation with _test_loop_ acting as a nested
            (k-1)-fold cross-validation loop.
        """
        all_MAE = []
        all_sigma = []
        all_pred_values = []
        all_pred_errors = []

        # k-fold cross-validation
        for ki in range(self.k):
            if opt_sigma:
                print('Fold {}'.format(ki+1))
            test_set = subsets[ki]
            val_sets = [subsets[m] for m in range(self.k) if m != ki]

            if opt_sigma:
                sigma = self._test_loop_(sigma0, L, val_sets, opt_sigma, bias)[1]
            else:
                sigma = sigma0

            all_sigma.append(abs(sigma))

            self.set_kernel_matrix(sigma, np.hstack(subsets))
            train_idx = [m for m in sum(subsets, []) if m not in test_set]
            pred_values, pred_errors = self.predict_values([],
                                                           train_idx,
                                                           test_set,
                                                           sigma,
                                                           L,
                                                           bias)
            # Errors
            test_values = self.data_values[test_set]
            MAE = np.mean(abs(pred_values-test_values))

            all_pred_values.append(pred_values)
            all_pred_errors.append(pred_errors)
            all_MAE.append(MAE)

        MAE = np.mean(all_MAE)
        sigma = np.mean(all_sigma)
        all_pred_values = np.hstack(all_pred_values)
        all_pred_errors = np.hstack(all_pred_errors)
        return MAE, sigma, all_pred_values, all_pred_errors

    def _test_loop_(self, sigma0, L, subsets, opt_sigma, bias):
        """ Cross-validation test loop.
        """
        if opt_sigma:
            res = minimize(self._objective_function_, sigma0,
                             args=(L, subsets, bias),
                             method='BFGS',
                             options={'gtol':1e-3,'maxiter':5})
#            print(res)
            sigma = res['x'][0]
        else:
            sigma = sigma0

        RMSE, MAE, apv, ape = self._inner_loop_(sigma, L, subsets, bias)

        return MAE, sigma, apv, ape

    def _inner_loop_(self, sigma, L, subsets, bias):
        """ Inner loop of the cross-validation.
        """
        self.set_kernel_matrix(sigma, np.hstack(subsets))

        all_RMSE = []
        all_MAE = []
        all_pred_values = []
        all_pred_errors = []

        if self.nested:
            n = self.k - 1
        else:
            n = self.k

        # n-fold cross-validation
        for ni in range(n):
            validation_set = subsets[ni]
            train_idx = [m for m in sum(subsets, [])
                         if m not in validation_set]
            pred_values, pred_errors = self.predict_values([],
                                                           train_idx,
                                                           validation_set,
                                                           sigma,
                                                           L,
                                                           bias)

            test_values = self.data_values[validation_set]
            MAE = np.mean(abs(pred_values-test_values))
            RMSE = np.sqrt(np.mean((pred_values-test_values)**2))

            all_RMSE.append(RMSE)
            all_MAE.append(MAE)
            all_pred_values.append(pred_values)
            all_pred_errors.append(pred_errors)

        RMSE = np.mean(all_RMSE)
        MAE = np.mean(all_MAE)
        all_pred_values = np.hstack(all_pred_values)
        all_pred_errors = np.hstack(all_pred_errors)

        print('sigma = {} (RMSE = {})'.format(sigma, RMSE))
        return RMSE, MAE, all_pred_values, all_pred_errors

    def _objective_function_(self, sigma, L, subsets, bias):
        """ Objective function for optimization of the hyperparameter.
        """
        sigma = sigma[0]
        RMSE, MAE, apv, ape = self._inner_loop_(sigma, L, subsets, bias)

        return RMSE
