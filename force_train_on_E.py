import numpy as np
import matplotlib.pyplot as plt
from krr_class import krr_class
from doubleLJ import doubleLJ
from bob_features import bob_features
from eksponentialComparator import eksponentialComparator
from scipy.optimize import minimize, fmin_bfgs

def makePerturbedGridStructure(Natoms):
    Nside = int(np.ceil(np.sqrt(Natoms)))
    r0 = 1.5
    x = np.array([[i, j] for i in range(Nside) for j in range(Nside)
                  if j*Nside + i+1 <= Natoms]).reshape(2*Natoms) * r0
    x += (np.random.rand(2*Natoms) - 0.5) * 0.6
    return x


def makeStructuresFromRelaxation(Ndata, Natoms, params):
    def LJenergy(x, *params):
        return doubleLJ(x, params[0], params[1], params[2])[0]
    def LJforce(x, *params):
        return -doubleLJ(x, params[0], params[1], params[2])[1]
    Nside = int(np.ceil(np.sqrt(Natoms)))
    r0 = 1.5
    x0 = np.array([[i, j] for i in range(Nside) for j in range(Nside)
                   if j*Nside + i+1 <= Natoms]).reshape(2*Natoms) * r0
    X = np.zeros((Ndata, 2*Natoms))
    k = 0
    while True:
        x = x0 = (np.random.rand(2*Natoms) - 0.5) * 0.6
        res = fmin_bfgs(LJenergy, x, LJforce, args=params,
                        gtol=1e-2,
                        retall=True,
                        full_output=True)
        x_evol = res[7]
        x_add = np.array(x_evol[0::3])
        Nadd = min(Ndata-k, x_add.shape[0])
        print('k:', k)
        X[k:k+Nadd, :] = x_add[:Nadd]
        if k+Nadd < Ndata:
            k += Nadd
        else:
            break
    return X

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


def relaxWithModel(x, model):
    def getEandF(pos):
        E = model.predict_energy(pos=pos)
        F = -model.predict_force()
        return E, F
    res = minimize(getEandF, x,
                   method="TNC",
                   jac=True,
                   tol=1e-1)
    return res.x, res.fun

    
def forceCurve(x, krr_class, coord):
    eps, r0, sigma = 1.8, 1.1, np.sqrt(0.02)
    unitvec = np.zeros(x.shape[0])
    unitvec[coord] = 1
    Npoints = 1000
    perturb_array = np.linspace(-5, 5, Npoints)
    Xtest = np.array([x + unitvec*pertubation for pertubation in perturb_array])
    curve = np.array([[perturb_array[i],
                       krr_class.predict_energy(pos=Xtest[i]),
                       krr_class.predict_force(pos=Xtest[i])[coord],
                       doubleLJ(Xtest[i], eps, r0, sigma)[0],
                       doubleLJ(Xtest[i], eps, r0, sigma)[1][coord]]
                      for i in range(Npoints)])  # if doubleLJ(Xtest[i], eps, r0, sigma)[0] < 0])

    F_fd = -np.diff(curve[:,1])/np.diff(curve[:,0])
    
    dx = perturb_array[1] - perturb_array[0]
    
    filter_high = curve[:,4] < 20
    filter_low = curve[:,4] > -20
    curve = curve[filter_low & filter_high]
    perturb_array = curve[:,0]
    Epred = curve[:,1]
    Fpredx = curve[:,2]
    E_LJ = curve[:,3]
    F_LJx = -curve[:,4]
    F_fd = F_fd[filter_low[:-1] & filter_high[:-1]]
    
    """
    Epred = np.array([krr_class.predict_energy(pos=xi) for xi in Xtest])
    Etest = np.array([doubleLJ(xi, eps, r0, sigma)[0] for xi in Xtest])
    plt.plot(pertub_array, Etest, color='r')
    plt.plot(pertub_array, Epred, color='b')
    """
    
    #pertub_array[pertub_array > 0] = 'nan'
    plt.scatter(perturb_array, E_LJ, color='r', s=1)
    plt.scatter(perturb_array, F_LJx, color='b', s=1)
    plt.scatter(perturb_array, Epred, color='y', s=1)
    plt.scatter(perturb_array, Fpredx, color='g', s=1)
    plt.scatter(perturb_array[:-1]+dx/2, F_fd, color='c', s=1)
    
    plt.show()

def relaxTest(Xtest, model, params):
    params = (1.8, 1.1, np.sqrt(0.02))
    Ntest, Ncoord = Xtest.shape
    Xrelax = np.zeros((Ntest, Ncoord))
    Erelax = np.zeros(Ntest)
    XrelaxLJ = np.zeros((Ntest, Ncoord))
    ErelaxLJ = np.zeros(Ntest)
    for i in range(Ntest):
        print('Progress: {}/{}'.format(i, Ntest))
        Xrelax[i,:], Erelax[i] = relaxWithModel(Xtest[i], model)
        res = minimize(doubleLJ, Xtest[i], params,
                       method="TNC",
                       jac=True,
                       tol=1e-4)
        XrelaxLJ[i,:] = res.x
        ErelaxLJ[i] = res.fun

    Erelax_struct = np.array([doubleLJ(x, params[0], params[1], params[2])[0] for x in Xrelax])
    print('[Erelax, ErelaxLJ]:\n', np.c_[Erelax, ErelaxLJ, Erelax_struct])
    MAE = np.mean(np.abs(ErelaxLJ - Erelax))
    print('MAE relax:', MAE)

def main():
    np.random.seed(455)
    Ndata = 100
    Natoms = 4

    # parameters for potential
    eps, r0, sigma = 1.8, 1.1, np.sqrt(0.02)
    params = (eps, r0, sigma)

    # parameters for kernel and regression
    lamb = 1e-5
    sig = 10
    
    # X = np.array([makePerturbedGridStructure(Natoms) for i in range(Ndata)])
    X = np.array([makeConstrainedStructure(Natoms) for i in range(Ndata)])
    featureCalculator = bob_features()
    G, I = featureCalculator.get_featureMat(X)
    
    E = np.zeros(Ndata)
    F = np.zeros((Ndata, 2*Natoms))
    for i in range(Ndata):
        E[i], F[i] = doubleLJ(X[i], eps, r0, sigma)

    Gtrain = G[:-20]
    Etrain = E[:-20]

    # Train model
    comparator = eksponentialComparator(sigma=sig)
    krr = krr_class(comparator=comparator, featureCalculator=featureCalculator)
    # GSkwargs = {'reg': [lamb], 'sigma': [sig]}
    GSkwargs = {'reg': np.logspace(-6, -3, 5), 'sigma': np.logspace(-1, 1, 5)}
    print(Etrain.shape, Gtrain.shape)
    MAE, params = krr.gridSearch(Etrain, Gtrain, **GSkwargs)
    print('sigma', params['sigma'])
    print('reg', params['reg'])

    print('MAE=', MAE)
    print('MAE using mean:', np.mean(np.fabs(E-np.mean(E))))
    print('Mean absolute energy:', np.mean(np.fabs(E)))

    forceCurve(X[-1], krr, 6)
    
    # print('E_unrelaxed\n', E[-20:])
    # Xtest = X[-20:]
    # relaxTest(Xtest, krr, params)
    
    """
    pos = np.reshape(pos, (Natoms, 2))
    plt.scatter(pos[:, 0], pos[:, 1])
    plt.show()
    """
    
if __name__ == "__main__":
    main()
