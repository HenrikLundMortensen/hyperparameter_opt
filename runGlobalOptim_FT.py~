import numpy as np
import matplotlib.pyplot as plt
from globalOptim import globalOptim
from scipy.optimize import minimize
from doubleLJ import doubleLJ, doubleLJ_energy, doubleLJ_gradient
from bob_features import bob_features
from eksponentialComparator import eksponentialComparator
from gaussComparator import gaussComparator
from krr_class import krr_class

def plotStructure(X, boxsize=None, color='b'):
    Natoms = int(X.shape[0]/2)
    if boxsize is None:
        boxsize = 1.5 * np.sqrt(Natoms)

    xbox = np.array([0, boxsize, boxsize, 0, 0])
    ybox = np.array([0, 0, boxsize, boxsize, 0])
    
    x = X[0::2]
    y = X[1::2]
    plt.plot(xbox, ybox, color='k')
    plt.scatter(x, y, color=color, s=8)
    plt.gca().set_aspect('equal', adjustable='box')

    
def testLocalMinimizer():
    def Efun(X):
        params = (1, 1.4, np.sqrt(0.02))
        return doubleLJ(X, params[0], params[1], params[2])
    optim = globalOptim(Efun, Natoms=20)
    optim.makeInitialStructure()
    Erel, Xrel = optim.relax()
    plotStructure(Xrel, color='r')
    print('E:', Erel)
    plt.show()



def main():
    def Efun(X):
        params = (1.5, 1, np.sqrt(0.02))
        return doubleLJ_energy(X, params[0], params[1], params[2])
    def gradfun(X):
        params = (1.5, 1, np.sqrt(0.02))
        return doubleLJ_gradient(X, params[0], params[1], params[2])
    optim = globalOptim(Efun, gradfun, Natoms=13, dmax=1.5, Niter=1000, Nstag=30, sigma=0.5, maxIterLocal=5)
    optim.runOptimizer()
    print('Esaved:\n', optim.Esaved[:optim.ksaved])
    print('best E:', optim.Ebest)
    plotStructure(optim.Xbest)
    plt.show()

def mainML():
    def Efun(X):
        params = (1.8, 1.1, np.sqrt(0.02))
        return doubleLJ_energy(X, params[0], params[1], params[2])
    def gradfun(X):
        params = (1.8, 1.1, np.sqrt(0.02))
        return doubleLJ_gradient(X, params[0], params[1], params[2])

    sig = 3
    reg = 1e-5
    comparator = gaussComparator(sigma=sig)
    featureCalculator = bob_features()
    krr = krr_class(comparator=comparator, featureCalculator=featureCalculator, reg=reg)
    optim = globalOptim(Efun, gradfun, krr, Natoms=7, dmax=2.5, Niter=200, Nstag=400, sigma=1, maxIterLocal=3)
    optim.runOptimizer()

    GSkwargs = {'reg': [reg], 'sigma': [sig]}
    MAE, params = krr.gridSearch(optim.Esaved[:optim.ksaved], positionMat=optim.Xsaved[:optim.ksaved], **GSkwargs)
    print('MAE:', MAE)
    print(optim.Esaved.T[:optim.ksaved])
    print('best E:', optim.Ebest)
    # Make into numpy arrays
    ktrain = np.array(optim.ktrain)
    EunrelML = np.array(optim.EunrelML)
    Eunrel = np.array(optim.Eunrel)
    FunrelML = np.array(optim.FunrelML)
    FunrelTrue = np.array(optim.FunrelTrue)
    
    dErel = np.fabs(np.array(optim.ErelML) - np.array(optim.ErelTrue))
    dErelTrue = np.fabs(np.array(optim.ErelML) - np.array(optim.ErelMLTrue))
    dE2rel = np.fabs(np.array(optim.ErelML) - np.array(optim.E2rel))
    dEunrel = np.fabs(EunrelML - Eunrel)
    dFunrel = np.fabs(FunrelML = FunrelTrue)
    
    np.savetxt('dErel_vs_ktrain2.txt', np.c_[ktrain, dErel, dErelTrue, EunrelML, Eunrel], delimiter='\t')

    print(FunrelTrue)
    print(FunrelTrue.shape)
    
    plt.figure(1)
    plt.title('Structure Example')
    plotStructure(optim.Xbest)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.figure(2)
    plt.title('Energies of relaxed struxtures')
    plt.loglog(ktrain, dErel, color='r')
    plt.loglog(ktrain, dErelTrue, color='b')
    plt.loglog(ktrain, dE2rel, color='g')
    plt.xlabel('NData')
    plt.ylabel('Energy')
    # plt.legend()
    plt.figure(3)
    plt.title('Energies of unrelaxed structures')
    plt.loglog(ktrain, dEunrel)
    plt.xlabel('NData')
    plt.ylabel('Energy')
    plt.figure(4)
    plt.title('Difference between ML and True force of unrelaxed structures')
    plt.loglog(ktrain, dFunrel)
    plt.xlabel('NData')
    plt.ylabel('Force')
    plt.show()
    
def mainTestLearning():
    def Efun(X):
        params = (1.8, 1.1, np.sqrt(0.02))
        return doubleLJ_energy(X, params[0], params[1], params[2])

    def gradfun(X):
        params = (1.8, 1.1, np.sqrt(0.02))
        return doubleLJ_gradient(X, params[0], params[1], params[2])

    sig = 3
    reg = 1e-5
    Natoms = 7
    comparator = gaussComparator(sigma=sig)
    featureCalculator = bob_features()
    krr = krr_class(comparator=comparator, featureCalculator=featureCalculator, reg=reg)
    
    Nruns = 5
    Nstructs = 1200
    optimData = np.zeros((Nruns, Nstructs, 2*Natoms))
    for i in range(Nruns):
        print('Data creation: {}/{}'.format(i, Nruns))
        optim = globalOptim(Efun, gradfun, krr, Natoms=Natoms, dmax=2.5,
                            Niter=200, Nstag=400, sigma=1, maxIterLocal=3)
        optim.runOptimizer()
        optimData[i,:] = optim.Xsaved[:Nstructs]
    print('optimData created with shape:', optimData.shape)
    optimData = np.array(optimData)
    
    Npoints = 1
    FVU_energy = np.zeros(Npoints)
    FVU_force = np.zeros((Npoints, 2*Natoms))
    FVU_force_finite = np.zeros((Npoints, 2*Natoms))
    Ntrain_array = np.logspace(3, 3, Npoints).astype(int)
    for i in range(Nruns):
        # Calculate all energies and forces for this run
        E = np.zeros(Nstructs)
        F = np.zeros((Nstructs, 2*Natoms))
        for s in range(Nstructs):
            E[s], grad = doubleLJ(optimData[i,s], 1.8, 1.1, np.sqrt(0.02))
            F[s,:] = -grad
        # Calculate LC
        for n in range(Npoints):
            print('i:{}/{} , n:{}/{}'.format(i,Nruns,n,Npoints))
            Ntrain = Ntrain_array[n]
            Ntest = 10  # int(max(10, np.round(Ntrain/5)))
            posTrain = optimData[i, :Ntrain]
            posTest = optimData[i, Ntrain:Ntrain+Ntest]
            krr.fit(E[:Ntrain], positionMat=posTrain)
            Etest = E[Ntrain:Ntrain+Ntest]
            Ftest = F[Ntrain:Ntrain+Ntest]
            FVU_energy[n] += krr.get_MAE_energy(Etest, positionMat=posTest)
            FVU_force[n,:] += krr.get_MAE_force(Ftest, positionMat=posTest)

            Ffinite = np.array([finiteDiff(krr, pos) for pos in posTest])
            FVU_force_finite[n,:] += calcFVU_force(Ftest, Ffinite)
            if n == 9:
                print(np.c_[Ftest[0].T, Ffinite[0].T])
    FVU_energy /= Nruns
    FVU_force /= Nruns
    FVU_force_finite /= Nruns

    np.savetxt('LC_search_bob_N7.txt', np.c_[Ntrain_array, FVU_energy, FVU_force], delimiter='\t')
    plt.figure(1)
    plt.title('Energy FVU vs training size (from search)')
    plt.loglog(Ntrain_array, FVU_energy)
    plt.xlabel('# training data')
    plt.ylabel('FVU')
    plt.figure(2)
    plt.title('Force FVU vs training size (from search)')
    plt.loglog(Ntrain_array, FVU_force)
    plt.xlabel('# training data')
    plt.ylabel('FVU')
    plt.figure(3)
    plt.title('finite difference Force FVU vs training size (from search)')
    plt.loglog(Ntrain_array, FVU_force_finite)
    plt.xlabel('# training data')
    plt.ylabel('FVU')
    plt.show()

def finiteDiff(MLmodel, pos, dx=1e-3):
    Ndf = pos.shape[0]
    identity = np.identity(Ndf)
    E0 = MLmodel.predict_energy(pos=pos)
    F = -np.array([MLmodel.predict_energy(pos=pos+dx*ei) - E0 for ei in identity]) / dx
    return F

def calcFVU_force(FtrueMat, FpredMat):
    MSE_force = np.mean((FtrueMat - FpredMat)**2, axis=0)
    var_force = np.var(FtrueMat, axis=0)
    return MSE_force / var_force


def energyANDforceLC_searchData():
    # np.random.seed(455)
    Ndata = 1500
    Natoms = 7

    # parameters for potential                                                                                                          
    eps, r0, sigma = 1.8, 1.1, np.sqrt(0.02)

    # parameters for kernel and regression                                                                                              
    reg = 1e-5
    sig = 3

    def Efun(X):
        params = (1.8, 1.1, np.sqrt(0.02))
        return doubleLJ_energy(X, params[0], params[1], params[2])

    def gradfun(X):
        params = (1.8, 1.1, np.sqrt(0.02))
        return doubleLJ_gradient(X, params[0], params[1], params[2])

    featureCalculator = bob_features()
    comparator = gaussComparator(sigma=sig)
    krr = krr_class(comparator=comparator, featureCalculator=featureCalculator)

    optim = globalOptim(Efun, gradfun, krr, Natoms=Natoms, dmax=2.5,
                        Niter=200, Nstag=400, sigma=1, maxIterLocal=3)
    optim.runOptimizer()
    X = optim.Xsaved[:Ndata]
    
    G, I = featureCalculator.get_featureMat(X)
    
    E = np.zeros(Ndata)
    F = np.zeros((Ndata, 2*Natoms))
    for i in range(Ndata):
        E[i], grad = doubleLJ(X[i], eps, r0, sigma)
        F[i] = -grad

    NpointsLC = 10
    Ndata_array = np.logspace(1,3,NpointsLC).astype(int)
    FVU_energy_array = np.zeros(NpointsLC)
    FVU_force_array = np.zeros((NpointsLC, 2*Natoms))
    for i in range(NpointsLC):
        N = int(3/2*Ndata_array[i])
        Esub = E[:N]
        Fsub = F[:N]
        Xsub = X[:N]
        Gsub = G[:N]
        Isub = I[:N]
        FVU_energy_array[i], FVU_force_array[i, :] = krr.cross_validation_EandF(Esub, Fsub, Gsub, Isub, Xsub, reg=reg)
        print(FVU_energy_array[i])
        #print(FVU_force_array[i])

    np.savetxt('LC_bob_N7_search2.txt', np.c_[Ndata_array, FVU_energy_array, FVU_force_array], delimiter='\t')
    plt.figure(1)
    plt.loglog(Ndata_array, FVU_energy_array)
    plt.figure(2)
    plt.loglog(Ndata_array, FVU_force_array)
    plt.show()


def mainEnergyAndForceCurve():
    #np.random.seed(455)
    Ndata = 1000
    Natoms = 7

    # parameters for potential
    eps, r0, sigma = 1.8, 1.1, np.sqrt(0.02)
    params = (eps, r0, sigma)

    # parameters for kernel and regression
    reg = 1e-5
    sig = 3

    def Efun(X):
        params = (1.8, 1.1, np.sqrt(0.02))
        return doubleLJ_energy(X, params[0], params[1], params[2])

    def gradfun(X):
        params = (1.8, 1.1, np.sqrt(0.02))
        return doubleLJ_gradient(X, params[0], params[1], params[2])

    featureCalculator = bob_features()
    comparator = gaussComparator(sigma=sig)
    krr = krr_class(comparator=comparator, featureCalculator=featureCalculator, reg=reg)

    optim = globalOptim(Efun, gradfun, krr, Natoms=Natoms, dmax=2.5,
                        Niter=200, Nstag=400, sigma=1, maxIterLocal=3)
    optim.runOptimizer()
    X = optim.Xsaved[:Ndata]

    # Calculate energy and forces
    E = np.zeros(Ndata)
    F = np.zeros((Ndata, 2*Natoms))
    for i in range(Ndata):
        E[i], grad = doubleLJ(X[i], eps, r0, sigma)
        F[i] = -grad

    Xtrain = X[:-1]
    Xtest = X[-1]
    Etrain = E[:-1]
    Etest = E[-1]
    krr.fit(Etrain, positionMat=Xtrain)

    Npoints = 1000
    dx_array = np.linspace(-0.05, 0.05, Npoints)
    dx_diff = dx_array[1]-dx_array[0]
    # choose coordinate to perturb
    i_perturb = 0
    ei = np.zeros(2*Natoms)
    ei[i_perturb] = 1

    # Calculate energy and forces of perturbed structures
    Etrue = np.zeros(Npoints)
    Ftrue = np.zeros((Npoints, 2*Natoms))
    for i in range(Npoints):
        Etrue[i], grad = doubleLJ(Xtest+dx_array[i]*ei, eps, r0, sigma)
        Ftrue[i] = -grad
    Ftrue0 = Ftrue[:,i_perturb]
    
    Epred = np.array([krr.predict_energy(pos=Xtest+dx*ei) for dx in dx_array])
    Fpred = np.array([krr.predict_force(pos=Xtest+dx*ei) for dx in dx_array])
    Fpred0 = Fpred[:,i_perturb]
    Ffinite0 = -(Epred[1:] - Epred[:-1]) / dx_diff
    
    plt.figure(1)
    plt.title('Energy with one coordinate varied')
    plt.plot(dx_array, Etrue, label='True energy')
    plt.plot(dx_array, Epred, label='Predicted energy')
    plt.xlabel('dx (perturbation of single coordinate)')
    plt.ylabel('Energy')
    plt.legend()
    
    plt.figure(2)
    plt.title('Force with one coordinate varied')
    plt.plot(dx_array, Ftrue0, label='True force')
    plt.plot(dx_array, Fpred0, label='Predicted force')
    plt.plot(dx_array[:-1]+dx_diff/2, Ffinite0, linestyle=':', label='finite difference force')
    plt.xlabel('dx (perturbation of single coordinate)')
    plt.ylabel('Energy')
    plt.legend()

    boxsize = 1.5 * np.sqrt(Natoms)

    xbox = np.array([0, boxsize, boxsize, 0, 0])
    ybox = np.array([0, 0, boxsize, boxsize, 0])
    x = Xtest[0::2]
    y = Xtest[1::2]
    # Perturbation line
    xline = np.array([x[0]-0.05*ei, x[0]+0.05*ei])
    yline = y[0]*np.ones(2)

    plt.figure(3)
    #plt.plot(xbox, ybox, color='k')
    plt.scatter(x, y, color='r', s=8)
    plt.plot(xline, yline)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.gca().set_aspect('equal', adjustable='box')
    
    plt.show()

            
if __name__ == '__main__':
    #mainML()
    #mainTestLearning()
    energyANDforceLC_searchData()
    #mainEnergyAndForceCurve()
