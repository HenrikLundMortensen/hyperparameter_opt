import numpy as np
import argparse
from krr_class import krr_class
from krr_utils import *
from HyperOptimizer import HyperOptimizer

from doubleLJ import doubleLJ
from bob_features import bob_features
from gaussComparator import gaussComparator


np.random.seed(13)

parser = argparse.ArgumentParser()
parser.add_argument("--Ndata",type=int,help="Number of data points to be generated for training/validation",default=200)
parser.add_argument("--Ntest",type=int,help="Number of data points to be generated for testing",default=200)
parser.add_argument("--Natoms",type=int,help="Number of atoms in the system",default=3)
parser.add_argument("--hlr",type=float,help="Hyper learning rate",default=1)
parser.add_argument("--savefile",type=str,help="String with name of file to be saved",default=None)
parser.add_argument("--q",type=float,help="Exponential scaling constant in initial sigmaVec",default=10)
parser.add_argument("--m",type=int,help="Number of distances included in initial sigmaVec",default=10)

args = parser.parse_args()
Ndata = args.Ndata
Ntest = args.Ntest
Natoms = args.Natoms
hlr = args.hlr
savefile = args.savefile
m = args.m
q = args.q


# Parameters for double well LJ potential
eps, r0, sigma = 1.8, 1.1, np.sqrt(0.02)
params = (eps, r0, sigma)


# Initialize KRR class instance
krr = krr_class(featureCalculator=bob_features,
                comparator=gaussComparator(sigma=1), # Initialize with sigma = 1. It is changed after gridsearch/optimization
                Ntrain=Ndata)

# Generate coordinates for atoms
X = getConstrainedStructureDataset(Ndata,Natoms)
Xtest = getConstrainedStructureDataset(Ntest,Natoms)

# Generate corresponding features
G, I = bob_features().get_featureMat(X)
Gtest, Itest = bob_features().get_featureMat(Xtest)


# Get energies and forecs
E,F = getEnergyAndForces(doubleLJ,X,Natoms,params)
Etest,Ftest = getEnergyAndForces(doubleLJ,Xtest,Natoms,params)

# Initialize hyper parameter optimization class instance
HO = HyperOptimizer(krr,
                    G=G,
                    E=E,
                    runs=30,
                    sig=0.5,
                    hlr=hlr,
                    tol = 1e-6,
                    k=5)

sigma_arr = np.linspace(0.6,1,2)
# BestGSSigma = HO.gridSearch(sigma_arr,verbose=True) 
# print('BestGSSigma = %g' %(BestGSSigma))

HO.krr.sigmaVec = HO.initializeSigmaVec(m,q)
print(HO.krr.sigmaVec)


HO.optimizeSigmaVec(verbose=True)
