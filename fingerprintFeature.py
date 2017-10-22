import numpy as np
from scipy.spatial.distance import euclidean
from scipy.special import erf

class fingerprintFeature():
    def __init__(self, X=None, rcut=4, binwidth=0.05, sigma=0.2, nsigma=4):
        """
        --input--
        X:                                                                                                                                     Contains data. Each row tepresents a structure given by cartesian coordinates
        in the form [x1, y1, ... , xN, yN]
    
        Rcut:
        Cutoff radius
        
        deltaR:
        radial binsize
        
        sigma:
        standard deviation for gaussian smearing
        
        nsigma:
        The distance (as the number of standard deviations sigma) at                                                            
        which the gaussian smearing is cut off (i.e. no smearing beyond                                                         
        that distance). 
        """
        self.X = X
        self.rcut = rcut
        self.binwidth = binwidth
        self.sigma = sigma
        self.nsigma = nsigma

        # parameters for the binning:
        self.m = int(np.ceil(self.nsigma*self.sigma/self.binwidth))  # number of neighbour bins included.
        self.smearing_norm = erf(0.25*np.sqrt(2)*self.binwidth*(2*self.m+1)*1./self.sigma)  # Integral of the included part of the gauss.
        self.Nbins = int(np.ceil(rcut/binwidth))

        # Cutoff volume
        self.cutoffVolume = 4*np.pi*self.rcut**2

    def get_singleFeature(self, x):
        """
        --input--
        x: atomic positions for a single structure in the form [x1, y1, ... , xN, yN]
        """
        Natoms = int(x.shape[0])
        gamma = Natoms*(Natoms-1)/2
            
        R = self.radiusVector(x)
        # filter distances longer than rcut + nsigma*sigma
        R = R[R <= self.rcut + self.nsigma*self.sigma]
        # Number of atoms within this radius
        N_within = R.shape[0]
        
        fingerprint = np.zeros(self.Nbins)
        for deltaR in R:
            drbin = deltaR % self.binwidth
            rabove = int(drbin > 0.5*self.binwidth)
            rbin = int(np.floor(deltaR/self.binwidth))
            for i in range(-self.m-(1-rabove), self.m+1+rabove):
                newbin = rbin + i  # maybe abs() to make negative bins constibute aswell.
                if newbin < 0 or newbin >= self.Nbins:
                    continue
                
                
                c = 0.25*np.sqrt(2)*self.binwidth*1./self.sigma
                value = 0.5*erf(c*(2*i+1))-0.5*erf(c*(2*i-1))
                #value = 0.5*erf(c*(2*i+2*(1-drbin)))-0.5*erf(c*(2*i-2*drbin)) # to make the continous
                # divide by smearing_norm
                value /= self.smearing_norm
                value /= (4*np.pi*deltaR**2)/self.cutoffVolume * self.binwidth * 0.5*N_within*(N_within-1)
                fingerprint[newbin] += value
        return fingerprint

    def get_featureMat(self, X):
        """
        Calculated the feature matrix based on a position matrix 'X'.
        ---input---
        X:
        Position matrix with each row 'x' containing the atomic coordinates of
        a structure. x = [x0,y0,x1,y1, ...].
        """
        featureMat = np.array([self.get_singleFeature(x) for x in X])
        return featureMat

    def radiusMatrix(self, x):
        """
        Calculates the matrix consisting of all pairwise euclidean distances.
        """
        Ndim = 2
        Natoms = int(x.shape[0])
        x = x.reshape((Natoms, Ndim))
        R = np.array([[euclidean(xi, xj) for xj in x] for xi in x])
        return R
    
    def radiusVector(self, x):
        """
        Calculates the vector consisting of all pairwise euclidean distances.
        """
        Ndim = 2
        Natoms = int(x.shape[0]/2)
        Ndistances = int(Natoms*(Natoms-1)/2)
        x = x.reshape((Natoms, Ndim))
        Rvec = np.zeros(Ndistances)
        k = 0
        for i in range(Natoms):
            for j in range(i+1, Natoms):
                Rvec[k] = euclidean(x[i],x[j])
                k += 1
        return Rvec
