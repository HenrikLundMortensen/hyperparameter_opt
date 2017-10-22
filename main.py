import numpy as np
import time
import statsmodels.api as sm

from IPython import embed
from krr_class import krr_class
from krr_utils import getConstrainedStructureDataset,getEnergyAndForces
from bob_features import bob_features
from gaussComparator import gaussComparator
from eksponentialComparator import eksponentialComparator
from doubleLJ import doubleLJ
from matplotlib import pyplot as plt
from HyperOptimizer import HyperOptimizer

featureCalculator = bob_features()


np.random.seed(13)
Ndata = 800
Ntest = 800
Natoms = 3

# parameters for potential
eps, r0, sigma = 1.8, 1.1, np.sqrt(0.02)
params = (eps, r0, sigma)

# parameters for kernel and regression
reg = 1e-7
sig = 1

X = getConstrainedStructureDataset(Ndata,Natoms)
Xtest = getConstrainedStructureDataset(Ntest,Natoms)

G, I = featureCalculator.get_featureMat(X)
Gtest, Itest = featureCalculator.get_featureMat(Xtest)

E,F = getEnergyAndForces(doubleLJ,X,Natoms,params)
Etest,Ftest = getEnergyAndForces(doubleLJ,Xtest,Natoms,params)



# Initialize KRR class
krr = krr_class(featureCalculator=bob_features,
                comparator=gaussComparator(sigma=sig),
                Ntrain=Ndata)



# Get distance matrix
distmat = np.sqrt(krr.comparator.getDistMat(G,G))

# Set entries larger than threshold to zero
distmatflat = np.matrix.flatten(distmat)
distmatflat = distmatflat[np.nonzero(distmatflat)]
nonzerodistmat = np.reshape(distmatflat,(Ndata,Ndata-1))
distsort = np.sort(nonzerodistmat,axis=1)


bw = np.mean(np.mean(np.power(distsort[:,0:10],1/2),axis=1))
dens = sm.nonparametric.KDEMultivariate(data=G,
                                        var_type='c'*G.shape[1],
                                        bw=[bw]*G.shape[1])
pdf = dens.pdf(G)
sigma_guess = 0.4/np.mean(pdf)
# krr.sigmaVec = 1/np.sqrt(pdf)
print('sigma_guess = %g' %(sigma_guess))




tic = time.time()
res = krr.gridSearch(E,
                     featureMat=G,
                     sigma=np.linspace(0.1,1,30),
                     reg=[reg],
                     k=5)
gridtime = time.time() - tic
print(res)
krr.sigmaVec = np.ones(Ndata)*res[1]['sigma']
krr.comparator.sigma = res[1]['sigma']
krr.fit(E,featureMat=G)
# error_before = krr.get_FVU_energy(Etest,featureMat=Gtest)
Epred = krr.predict(G,Gtest)
error_before = (Etest-Epred).T@(Etest-Epred)

print('Error before = %f' %(error_before))

HO = HyperOptimizer(krr,
                    G=G,
                    E=E,
                    runs=30,
                    sig=sigma_guess,
                    hlr=0.001,
                    tol = 1e-6,
                    k=5)
# gridsigma = HO.gridSearch(np.linspace(0.1,0.4,30))
# print('gridsigma = %4.12f' %(gridsigma))

krr.fit(E,featureMat=G)


krr.sigmaVec = np.ones(Ndata)*res[1]['sigma']
tic = time.time()
HO.optimizeSigmaVec(method='constant_hlr')
hypertime = time.time() - tic














# error_after = krr.get_FVU_energy(Etest,featureMat=Gtest)
Epred = krr.predictSigmaVec(G,Gtest,krr.sigmaVec)
error_after =  (Etest-Epred).T@(Etest-Epred)
print('Error after = %f' %(error_after))

Nsig= 50
sigmalist = np.linspace(0.1,1,Nsig)
err = np.zeros(Nsig)
Epred = np.zeros(Ntest)
for i,sig in enumerate(sigmalist):
    krr.comparator.sigma = sig
    krr.fit(E,featureMat=G)
    # for j,gtest in enumerate(Gtest):
    #     Epred[j] = krr.predict_energy(fnew=gtest)
    Epred = krr.predict(G,Gtest)
    err[i] =(Etest-Epred).T@(Etest-Epred)
    # err[i] = krr.get_FVU_energy(Etest,featureMat=Gtest)
    print('Progress: %i/%i\r' %(i+1,Nsig),end='')
print('')


err_val = np.zeros(Nsig)
for i,sig in enumerate(sigmalist):
    krr.comparator.sigma = sig
    for ki in range(HO.k):
        Gtrain,Gval,Etrain,Eval,_ = HO._get_train_and_val(ki)
        krr.fit(Etrain,featureMat=Gtrain)
        Epred = np.zeros(Gval.shape[0])        
        # for j,gval in enumerate(Gval):
        #     Epred[j] = krr.predict_energy(fnew=gval)
        Epred = krr.predict(Gtrain,Gval)            
        err_val[i] +=(Eval-Epred).T@(Eval-Epred)
        # err_val[i] = krr.get_FVU_energy(Etest,featureMat=Gtest)
    print('Progress: %i/%i\r' %(i+1,Nsig),end='')        
    err_val[i]= np.mean(err_val[i])
print('')

krr.comparator.sigma = HO.optimal_sig
krr.fit(E,featureMat=G)
    


fig =plt.figure()
ax = fig.gca()
ax.set_yscale('log')
ax.plot(sigmalist,err,color='red')
ax.plot(sigmalist,err_val,color='blue')
ax.plot(krr.comparator.sigma,error_after,'ko')
ax.plot(res[1]['sigma'],error_before,'go')
ax.set_xlim([sigmalist[0],sigmalist[-1]])
fig.savefig('err_vs_sigma')



