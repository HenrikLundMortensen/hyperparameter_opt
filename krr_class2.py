import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from doubleLJ import doubleLJ
from fingerprintFeature import fingerprintFeature
from eksponentialComparator import eksponentialComparator
from gaussComparator import gaussComparator

class krr_class():
    def __init__(self, featureCalculator=None, comparator=None, reg=1e-5, **comparator_kwargs):
        self.featureCalculator = featureCalculator
        self.comparator = comparator
        self.comparator.set_args(**comparator_kwargs)
        self.reg = reg

        assert self.comparator is not None

    def fit(self, data_values, featureMat=None, positionMat=None, reg=None):
        self.data_values = data_values

        # calculate features from positions if they are not given
        if featureMat is not None:
            self.featureMat = featureMat
        elif positionMat is not None and self.featureCalculator is not None:
            self.featureMat = self.featureCalculator.get_featureMat(positionMat)
        else:
            print("You need to set the feature matrix or both the position matrix and a feature calculator")

        if reg is not None:
            self.reg = reg
        self.similarityMat = self.comparator.get_similarity_matrix(self.featureMat)

        self.beta = np.mean(data_values)

        A = self.similarityMat + self.reg*np.identity(self.data_values.shape[0])

        #self.alpha = np.linalg.inv(A).dot(self.data_values-self.beta)
        self.alpha = np.linalg.solve(A, self.data_values - self.beta)

    def predict_energy(self, fnew=None, pos=None):
        if fnew is not None:
            self.fnew = fnew
        else:
            self.pos = pos
            assert self.featureCalculator is not None
            self.fnew = self.featureCalculator.get_singleFeature(x=self.pos)

        self.similarityVec = self.comparator.get_similarity_vector(self.fnew)

        return self.similarityVec.dot(self.alpha) + self.beta

    def predict_force(self, pos=None, fnew=None, inew=None):
        if pos is not None:
            self.pos = pos
            assert self.featureCalculator is not None
            self.fnew = self.featureCalculator.get_singleFeature(self.pos)
            self.similarityVec = self.comparator.get_similarity_vector(self.fnew)

        df_dR = self.featureCalculator.get_featureGradient(self.pos, self.fnew, self.inew)
        dk_df = self.comparator.get_jac(self.fnew)
        
        kernelDeriv = np.dot(dk_df, df_dR)
        return -(kernelDeriv.T).dot(self.alpha)

    def cross_validation(self, data_values, featureMat, k=3, reg=None):
        Ndata = data_values.shape[0]
        permutation = np.random.permutation(Ndata)
        data_values = data_values[permutation]
        featureMat = featureMat[permutation]

        Ntest = int(np.floor(Ndata/k))
        FVU = np.zeros(k)
        for ik in range(k):
            [i_train1, i_test, i_train2] = np.split(np.arange(Ndata),
                                                    [Ntest * ik, Ntest * (ik+1)])
            i_train = np.r_[i_train1, i_train2]
            self.fit(data_values[i_train], featureMat[i_train], reg=reg)
            FVU[ik] = self.get_FVU_energy(data_values[i_test], featureMat[i_test])
        return np.mean(FVU)

    def cross_validation_EandF(self, energy, force, featureMat, indexMat, positionMat, k=3, reg=None):
        Ndata, Ndf = positionMat.shape
        permutation = np.random.permutation(Ndata)
        energy = energy[permutation]
        force = force[permutation]
        featureMat = featureMat[permutation]
        indexMat = indexMat[permutation]
        positionMat = positionMat[permutation]
        
        Ntest = int(np.floor(Ndata/k))
        FVU_energy = np.zeros(k)
        FVU_force = np.zeros((k, Ndf))
        for ik in range(k):
            [i_train1, i_test, i_train2] = np.split(np.arange(Ndata),
                                                    [Ntest * ik, Ntest * (ik+1)])
            print('index:', ik)
            i_train = np.r_[i_train1, i_train2]
            self.fit(energy[i_train], featureMat[i_train], reg=reg)
            FVU_energy[ik] = self.get_FVU_energy(energy[i_test], featureMat[i_test])
            FVU_force[ik, :] = self.get_FVU_force(force[i_test], positionMat[i_test],
                                                  featureMat[i_test], indexMat[i_test])
        return np.mean(FVU_energy), np.mean(FVU_force, axis=0)

    def gridSearch(self, data_values, featureMat=None, positionMat=None, k=3, disp=False, **GSkwargs):
        if positionMat is not None and self.featureCalculator is not None:
            featureMat = self.featureCalculator.get_featureMat(positionMat)
        else:
            assert featureMat is not None
        sigma_array = GSkwargs['sigma']
        reg_array = GSkwargs['reg']
        Nsigma = len(sigma_array)
        Nreg = len(reg_array)
        best_args = np.zeros(2).astype(int)
        FVU_min = None
        for i in range(Nsigma):
            self.comparator.set_args(sigma=sigma_array[i])
            for j in range(Nreg):
                FVU = self.cross_validation(data_values, featureMat, k=k, reg=reg_array[j])
                if disp:
                    print('FVU:', FVU,'params: (', sigma_array[i],',', reg_array[j], ')')
                if FVU_min is None or FVU < FVU_min:
                    FVU_min = FVU
                    best_args = np.array([i, j])
        sigma_best = sigma_array[best_args[0]]
        reg_best = reg_array[best_args[1]]
        # Train with best parameters using all data
        self.comparator.set_args(sigma=sigma_best)
        self.fit(data_values, featureMat, reg=reg_best)
        return FVU_min, {'sigma': sigma_best, 'reg': reg_best}

    def get_FVU_energy(self, data_values, featureMat=None, positionMat=None):
        if featureMat is None:
            assert positionMat is not None
            featureMat = self.featureCalculator.get_featureMat(positionMat)
        Epred = np.array([self.predict_energy(f) for f in featureMat])
        error = Epred - data_values
        MSE = np.mean((Epred - data_values)**2)
        var = np.var(data_values)
        return MSE / var 

    def get_FVU_force(self, force, positionMat, featureMat=None, indexMat=None):
        if featureMat is None or indexMat is None:
            featureMat, indexMat = self.featureCalculator.get_featureMat(positionMat)
        Fpred = np.array([self.predict_force(positionMat[i], featureMat[i], indexMat[i])
                          for i in range(force.shape[0])])
        MSE_force = np.mean((Fpred - force)**2, axis=0)
        var_force = np.var(force, axis=0)        
        return MSE_force / var_force
    

def createData(Ndata, theta):
    # Define fixed points
    x1 = np.array([-1, 0, 1, 2])
    x2 = np.array([0, 0, 0, 0])

    # rotate ficed coordinates
    x1rot = np.cos(theta) * x1 - np.sin(theta) * x2
    x2rot = np.sin(theta) * x1 + np.cos(theta) * x2
    xrot = np.c_[x1rot, x2rot].reshape((1, 8))

    # Define an array of positions for the last pointB
    # xnew = np.c_[np.random.rand(Ndata)+0.5, np.random.rand(Ndata)+1]
    x1new = np.linspace(0.5, 2, Ndata)
    x2new = np.ones(Ndata)

    # rotate new coordinates
    x1new_rot = np.cos(theta) * x1new - np.sin(theta) * x2new
    x2new_rot = np.sin(theta) * x1new + np.cos(theta) * x2new

    xnew_rot = np.c_[x1new_rot, x2new_rot]

    # Make X matrix with rows beeing the coordinates for each point in a structure.
    # row example: [x1, y1, x2, y2, ...]
    X = np.c_[np.repeat(xrot, Ndata, axis=0), xnew_rot]
    return X


if __name__ == "__main__":

    eps, r0, sigma = 1.8, 1.1, np.sqrt(0.02)

    Ndata = 20
    reg = 1e-7  # expKernel: 0.005 , gaussKernel: 1e-7
    sig = 30  # expKernel: 0.3 , gaussKernel: 0.13

    theta = 0*np.pi

    # Set feature parameters
    
    
    X = createData(Ndata, theta)
    featureCalculator = fingerprintFeature()
    G = featureCalculator.get_featureMat(X)

    # Calculate energies for each structure
    E = np.zeros(Ndata)
    F = np.zeros((Ndata, 2*5))
    for i in range(Ndata):
        E[i], grad = doubleLJ(X[i], eps, r0, sigma)
        F[i, :] = -grad

    Gtrain = G[:-1]
    Etrain = E[:-1]
    beta = np.mean(Etrain)

    # Train model
    comparator = gaussComparator(sigma=sig)
    krr = krr_class(comparator=comparator, featureCalculator=featureCalculator)
    krr.fit(Etrain, Gtrain, reg=reg)

    """ gridSearch
    GSkwargs = {'reg': np.logspace(-7, -4, 10), 'sigma': np.logspace(-1, 2, 20)}
    print(Etrain.shape, Gtrain.shape)
    MAE, params = krr.gridSearch(Etrain, Gtrain, disp=True, **GSkwargs)
    print('sigma', params['sigma'])
    print('reg', params['reg'])
    """

    Npoints = 1000
    Etest = np.zeros(Npoints)
    Epredict = np.zeros(Npoints)
    Fpredx = np.zeros(Npoints)
    Ftestx = np.zeros(Npoints)
    Xtest0 = X[-1]
    Xtest = np.zeros((Npoints, 10))

    Gtest = np.zeros((Npoints, G.shape[1]))
    
    delta_array = np.linspace(-3.5, 0.5, Npoints)
    for i in range(Npoints):
        delta = delta_array[i]
        Xtest[i] = Xtest0
        pertub = np.array([delta, 0])
        pertub_rot = np.array([np.cos(theta) * pertub[0] - np.sin(theta) * pertub[1],
                               np.sin(theta) * pertub[0] + np.cos(theta) * pertub[1]])
        Xtest[i, -2:] += pertub_rot

        Gtest[i] = featureCalculator.get_singleFeature(Xtest[i])
        
        Etest[i], gradtest = doubleLJ(Xtest[i], eps, r0, sigma)
        Ftest = -gradtest
        Epredict[i] = krr.predict_energy(pos=Xtest[i])
        Ftestx[i] = np.cos(theta) * Ftest[-2] + np.cos(np.pi/2 - theta) * Ftest[-1]

        # Fpred = krr.predict_force()
        # Fpredx[i] = np.cos(theta) * Fpred[-2] + np.cos(np.pi/2 - theta) * Fpred[-1]

    dx = delta_array[1] - delta_array[0]
    Ffinite = (Epredict[:-1] - Epredict[1:])/dx

    plt.figure(1)
    # plt.plot(delta_array, Ftestx, color='c')
    # plt.plot(delta_array, Fpredx, color='y')
    # plt.plot(delta_array[1:]-dx/2, Ffinite, color='g')
    plt.plot(delta_array, Etest)
    plt.plot(delta_array, Epredict, color='r')
    plt.scatter(np.linspace(-1.5, 0, Ndata), np.zeros(Ndata)-12, s=10)
    plt.figure(3)
    plt.plot(np.arange(G.shape[1]), G.T)

    similarityMat = comparator.get_similarity_matrix(G)
    plt.figure(4)
    plt.pcolor(1-similarityMat, antialiaseds=True)
    plt.colorbar()

    fig = plt.figure(5)
    ax = fig.gca(projection='3d')

    # Make data
    Xax = np.linspace(0, 100, G.shape[1])
    Yax = np.linspace(-3.5, 0.5, Npoints)
    Xax, Yax = np.meshgrid(Xax, Yax)
    
    # Plot the surface.
    surf = ax.plot_surface(Xax, Yax, Gtest, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    plt.figure(6)
    plt.plot(delta_array, Gtest[:,20])
    
    # Plot first structure
    plt.figure(2)
    plt.plot(Xtest[:, -2], Xtest[:, -1], color='r')
    
    x = X[-1].reshape((5, 2))
    plt.scatter(x[:, 0], x[:, 1])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

    
