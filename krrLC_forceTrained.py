import numpy as np
import matplotlib.pyplot as plt
from krr_force_class import krr_force_class
from doubleLJ import doubleLJ
from bob_features import bob_features
from gaussComparator import gaussComparator
from eksponentialComparator import eksponentialComparator
from scipy.optimize import minimize, fmin_bfgs
import time

def makeConstrainedStructure(Natoms):
    boxsize = 1.5 * np.sqrt(Natoms)
    rmin = 0.9
    rmax = 1.5
    def validPosition(X, xnew):
        Natoms = int(len(X)/2) # Current number of atoms                                                                            
        if Natoms == 0:
            return True
        connected = False
        for i in range(Natoms):
            r = np.linalg.norm(xnew - X[2*i:2*i+2])
            if r < rmin:
                return False
            if r < rmax:
                connected = True
        return connected

    Xinit = np.zeros(2*Natoms)
    for i in range(Natoms):
        while True:
            xnew = np.random.rand(2) * boxsize
            if validPosition(Xinit[:2*i], xnew):
                Xinit[2*i:2*i+2] = xnew
                break
    return Xinit

def energyANDforceLC():
    np.random.seed(455)
    Ndata = 150
    Natoms = 7
    
    # parameters for potential
    eps, r0, sigma = 1.8, 1.1, np.sqrt(0.02)
    params = (eps, r0, sigma)

    # parameters for kernel and regression
    reg = 1e-7
    sig = 0.3
    
    X = np.array([makeConstrainedStructure(Natoms) for i in range(Ndata)])
    featureCalculator = bob_features()
    G, I = featureCalculator.get_featureMat(X)

    comparator = gaussComparator(sigma=sig)
    krr = krr_force_class(comparator=comparator, featureCalculator=featureCalculator)

    E = np.zeros(Ndata)
    F = np.zeros((Ndata, 2*Natoms))
    for i in range(Ndata):
        E[i], grad = doubleLJ(X[i], eps, r0, sigma)
        F[i] = -grad

    NpointsLC = 10
    Ndata_array = np.logspace(1,2,NpointsLC).astype(int)
    FVU_energy_array = np.zeros(NpointsLC)
    FVU_force_array = np.zeros((NpointsLC, 2*Natoms))
    for i in range(NpointsLC):
        N = int(3/2*Ndata_array[i])
        Esub = E[:N]
        Fsub = F[:N]
        Xsub = X[:N]
        Gsub = G[:N]
        Isub = I[:N]
        t0 = time.time()
        FVU_force_array[i,:] = krr.cross_validation(Fsub, Xsub, reg=reg)
        print('dt:', time.time() - t0)
        print(FVU_force_array[i])

    np.savetxt('LC_bob_FT.txt', np.c_[Ndata_array, FVU_force_array], delimiter='\t')
    plt.figure(2)
    plt.title('Force learning curve (Model trained on forces)')
    plt.loglog(Ndata_array, np.mean(FVU_force_array, axis=1))
    plt.xlabel('# training data')
    plt.ylabel('mean FVU')
    plt.ylim([10**(-2), 10**(-0.6)])
    plt.show()

if __name__ == "__main__":
    #energyLC()
    energyANDforceLC()
