import numpy as np


def tmp(krr,G,Gtest,E,Etest):
    """
    """
    
    Nsig= 50
    sigmalist = np.linspace(0.1,1,Nsig)
    err = np.zeros(Nsig)
    Epred = np.zeros(1000)
    
    for i,sig in enumerate(sigmalist):
        krr.comparator.sigma = sig
        krr.fit(E,featureMat=G)
        for j,gtest in enumerate(Gtest):
            Epred[j] = krr.predict_energy(fnew=gtest)
        Epred = krr.predict(G,Gtest)
        err[i] =(Etest-Epred).T@(Etest-Epred)
        # err[i] = krr.get_FVU_energy(Etest,featureMat=Gtest)
        print('Progress: %i/%i\r' %(i+1,Nsig),end='')
    return err
