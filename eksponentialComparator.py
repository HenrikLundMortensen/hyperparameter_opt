import numpy as np
from scipy.spatial.distance import euclidean


class eksponentialComparator():
    def __init__(self, featureMat=None, **kwargs):
        self.featureMat = featureMat
        if 'sigma' in kwargs:
            self.sigma = kwargs['sigma']

    def set_args(self, **kwargs):
        if 'sigma' in kwargs:
            self.sigma = kwargs['sigma']

    def get_similarity_matrix(self, featureMat=None):
        if featureMat is not None:
            self.featureMat = featureMat
        else:
            print("You need to supply a feature matrix")

        self.similarityMat = np.array([[self.single_comparison(f1, f2, self.sigma)
                                        for f2 in self.featureMat]
                                       for f1 in self.featureMat])
        return self.similarityMat

    def get_similarity_vector(self, fnew):

        self.similarityVec = np.array([self.single_comparison(fnew, f, self.sigma)
                                       for f in self.featureMat])
        return self.similarityVec

    def single_comparison(self, feature1, feature2, sigma=None):
        if sigma is None:
            sigma = self.sigma
        d = euclidean(feature1, feature2)
        return np.exp(-1/(2*sigma**2)*d)

    def get_jac(self, fnew, kappa=None, featureMat=None):
        """
        Calculates tor jacobian of the similarity vector 'k' with respect
        to the feature vector 'f' of the new data-point.
        ie. calculates dk_df.
        Using the chain rule: dk_df = dk_dd*dd_df , where d is the distance measure
        """
        if kappa is None:
            kappa = self.similarityVec.copy()
        if featureMat is None:
            featureMat = self.featureMat.copy()

        dk_dd = -1/(2*self.sigma**2)*kappa.reshape((kappa.shape[0], 1))
        dd_df = np.array([-(f - fnew)/euclidean(f, fnew) for f in featureMat])

        dk_df = np.multiply(dk_dd, dd_df)
        return dk_df

    def get_Hess_single(self, f1, f2):
        """
        Calculated the hessian of the kernel function with respect to
        the two features f1 and f2.
        ie. calculates: d^2/df1df2(kernel)
        """
        # kernel between the two features
        kernel = self.single_comparison(f1, f2)

        d = np.linalg.norm(f2 - f1)
        dd_df1 = -(f2-f1)/d
        dd_df2 = -dd_df1

        print('f1=\n', f1)
        print('f2=\n', f2)
        print('d=', d)
        print('dd_df1=\n', dd_df1)
        
        d2d_df1df2 = np.array([f2 * f1_i for f1_i in f1])/d**3

        Hess = -1/self.sigma**2 * (-1/self.sigma**2 * kernel * dd_df1 * dd_df2 +
                                   kernel * d2d_df1df2)
        print('d double deriv:\n', d2d_df1df2)
        print('Hess=\n', Hess)
        return Hess
        
