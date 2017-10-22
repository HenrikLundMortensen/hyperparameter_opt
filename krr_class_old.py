import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time
from sklearn.kernel_ridge import KernelRidge

def doubleLJ(x, *params):
    """
    Calculates total energy and gradient of N atoms interacting with a
    double Lennard-Johnes potential.
    
    Input:
    x: positions of atoms in form x= [x1,y2,x2,y2,...]
    params: parameters for the Lennard-Johnes potential
    Output:
    E: Total energy
    dE: gradient of total energy
    """
    eps, r0, sigma = params
    N = x.shape[0]
    Natoms = int(N/2)
    x = np.reshape(x, (Natoms, 2))

    E = 0
    dE = np.zeros(N)
    for i in range(Natoms):
        for j in range(Natoms):
            r = np.sqrt(sp.dot(x[i] - x[j], x[i] - x[j]))
            if j > i:
                E1 = 1/r**12 - 2/r**6
                E2 = -eps * np.exp(-(r - r0)**2 / (2*sigma**2))
                E += E1 + E2
            if j != i:
                dxij = x[i, 0] - x[j, 0]
                dyij = x[i, 1] - x[j, 1]

                dEx1 = 12*dxij*(-1 / r**14 + 1 / r**8)
                dEx2 = eps*(r-r0)*dxij / (r*sigma**2) * np.exp(-(r - r0)**2 / (2*sigma**2))

                dEy1 = 12*dyij*(-1 / r**14 + 1 / r**8)
                dEy2 = eps*(r-r0)*dyij / (r*sigma**2) * np.exp(-(r - r0)**2 / (2*sigma**2))

                dE[2*i] += dEx1 + dEx2
                dE[2*i + 1] += dEy1 + dEy2
    return E, -dE


class bob_features():
    def __init__(self, X):
        """
        --input--
        X:
        Contains data. Each row tepresents a structure given by cartesian coordinates
        in the form [x1, y1, ... , xN, yN]
        """
        self.X = X
        self.calc_featureMat()
    
    def calc_featureMat(self):
        Ndata = self.X.shape[0]
        Natoms = int(self.X.shape[1]/2)
        Nfeatures = int(Natoms*(Natoms-1)/2)
        G = np.zeros((Ndata, Nfeatures))
        I = []
        for n in range(Ndata):
            g, atomIndices = self.calc_singleFeature(self.X[n])
            G[n, :] = g
            I.append(atomIndices)
        self.G = G
        self.I = I
    
    def calc_singleFeature(self, x):
        """
        --input--
        x: atomic positions for a single structure in the form [x1, y1, ... , xN, yN]
        """
        Natoms = int(np.size(x, 0)/2)
        x = x.reshape((Natoms, 2))

        # Calculate inverse bond lengths
        g = np.array([1/np.linalg.norm(x[j]-x[i]) for i in range(Natoms) for j in range(i+1, Natoms)])
        
        # Make list of atom indices corresponding to the elements of the feature g
        atomIndices = [(i, j) for i in range(Natoms) for j in range(i+1, Natoms)]
        
        # Get indices that sort g in decending order
        sorting_indices = np.argsort(-g)
        
        # Sort features and atomic indices
        g_ordered = g[sorting_indices]
        atomIndices_ordered = [atomIndices[i] for i in sorting_indices]
        return g_ordered, atomIndices_ordered


def bobDeriv(pos, g, atomIndices):
    Nr = len(atomIndices)
    pos = pos.reshape((int(Nr/2), 2))
    Nfeatures = np.size(g, 0)
    
    atomIndices = np.array(atomIndices)
    # Calculate gradient of bob-feature
    gDeriv = np.zeros((Nfeatures, Nr))
    for i in range(Nfeatures):
        a0 = atomIndices[i, 0]
        a1 = atomIndices[i, 1]
        inv_r = g[i]
        gDeriv[i, 2*a0:2*a0+2] += inv_r**3*np.array([pos[a1, 0] - pos[a0, 0], pos[a1, 1] - pos[a0, 1]])
        gDeriv[i, 2*a1:2*a1+2] += -inv_r**3*np.array([pos[a1, 0] - pos[a0, 0], pos[a1, 1] - pos[a0, 1]])
    return gDeriv


def gaussKernel(g1, g2, sigma):
    d = np.linalg.norm(g2 - g1)
    return np.exp(-1/(2*sigma**2)*d)  # **2)


def kernelMat(G, sigma):
    Ndata = G.shape[0]
    K = np.zeros((Ndata, Ndata))
    for i in range(Ndata):
        for j in range(i, Ndata):
            K[i, j] = gaussKernel(G[i, :], G[j, :], sigma)
    K = K + np.triu(K, k=1).T
    return K


def kernelVec(gnew, G, sigma):
    kappa = np.array([gaussKernel(gnew, g, sigma) for g in G])
    return kappa


def kernelVecDeriv(pos, gnew, inew, G, sig):
    Ntrain = G.shape[0]
    # dvec = np.array([np.linalg.norm(g - gnew) for g in G])
    kappa = kernelVec(gnew, G, sig).reshape((Ntrain, 1))
    dd_dg = np.array([-np.sign(g - gnew) for g in G])
    dg_dR = bobDeriv(pos, gnew, inew)
    dd_dR = np.dot(dd_dg, dg_dR)
    dk_dd = -1/(2*sig**2)*kappa  # -1/sig**2*np.multiply(dvec, kappa)
    kernelDeriv = np.multiply(dk_dd, dd_dR)
    return kernelDeriv.T

    
def createData(Ndata, theta):
    # Define fixed points
    x1 = np.array([-1, 0, 1, 2])
    x2 = np.array([0, 0, 0, 0])

    # rotate ficed coordinates
    x1rot = np.cos(theta) * x1 - np.sin(theta) * x2
    x2rot = np.sin(theta) * x1 + np.cos(theta) * x2
    xrot = np.c_[x1rot, x2rot].reshape((1, 8))
    
    
    # Define an array of positions for the last pointB
    #xnew = np.c_[np.random.rand(Ndata)+0.5, np.random.rand(Ndata)+1]
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

    Ndata = 100
    lamb = 0.005
    sig = 0.3

    theta = 0.7*np.pi
    X = createData(Ndata, theta)
    features = bob_features(X)
    G = features.G

    # Calculate energies for each structure
    E = np.zeros(Ndata)
    F = np.zeros((Ndata, 2*5))
    for i in range(Ndata):
        E[i], F[i, :] = doubleLJ(X[i], eps, r0, sigma)
    
    Gtrain = G[:-1]
    Etrain = E[:-1]
    beta = np.mean(Etrain)

    # Train model
    K = kernelMat(Gtrain, sig)
    print("K[0,:]=\n", K[0, :])
    alpha = np.linalg.inv(K + lamb*np.identity(np.size(Etrain, 0))).dot(Etrain-beta)

    Npoints = 1000
    Etest = np.zeros(Npoints)
    Epredict = np.zeros(Npoints)
    Fpredx = np.zeros(Npoints)
    Ftestx = np.zeros(Npoints)
    Xtest0 = X[-1]
    Xtest = np.zeros((Npoints, 10))
    print(Xtest.shape)
    delta_array = np.linspace(-3.5, 0.5, Npoints)
    for i in range(Npoints):
        delta = delta_array[i]
        Xtest[i] = Xtest0
        pertub = np.array([delta, 0])
        pertub_rot = np.array([np.cos(theta) * pertub[0] - np.sin(theta) * pertub[1],
                               np.sin(theta) * pertub[0] + np.cos(theta) * pertub[1]])
        Xtest[i, -2:] += pertub_rot
        Gtest, Itest = features.calc_singleFeature(Xtest[i])
        kappa = kernelVec(Gtest, Gtrain, sig)
        if i == 0:
            print("kappa=\n", kappa)
        Etest[i], Ftest = doubleLJ(Xtest[i], eps, r0, sigma)
        Epredict[i] = kappa.dot(alpha) + beta
        Ftestx[i] = np.cos(theta) * Ftest[-2] + np.cos(np.pi/2 - theta) * Ftest[-1]
        
        kappaDeriv = kernelVecDeriv(Xtest[i], Gtest, Itest, Gtrain, sig)
        Fpred = -kappaDeriv.dot(alpha)
        Fpredx[i] = np.cos(theta) * Fpred[-2] + np.cos(np.pi/2 - theta) * Fpred[-1]

    dx = delta_array[1] - delta_array[0]
    Ffinite = (Epredict[:-1] - Epredict[1:])/dx

    plt.figure(1)
    plt.plot(delta_array, Ftestx, color='c')
    plt.plot(delta_array, Fpredx, color='y')
    plt.plot(delta_array[1:], Ffinite, color='g')
    plt.plot(delta_array, Etest)
    plt.plot(delta_array, Epredict, color='r')

    """
    Xtest = Xtest0.copy()
    Xtest[-2] -= 1.2
    gtest, itest = features.calc_singleFeature(Xtest)

    kappaDeriv = kernelVecDeriv(Xtest, gtest, itest, Gtrain, sig)

    a = alpha

    Fpred = -kappaDeriv.dot(a)
    E, Ftest = doubleLJ(Xtest, eps, r0, sigma)
    print(Fpred)
    print(Ftest)
    """

    # Plot first structure
    plt.figure(2)
    plt.scatter(Xtest[:, -2], Xtest[:, -1], color='r')
    plt.scatter(Xtest[0, -2], Xtest[0, -1], color='y')
    
    x = X[-1].reshape((5, 2))
    plt.scatter(x[:, 0], x[:, 1])
    
    plt.show()

    
