import numpy as np


class gaussFeature():
    def __init__(self, X=None, eta=4, Xi=[2], cutoffRadius=6, lamb=[-1,1], Rs=[0,2,4]):
        """
        ---Input---
        X:
        Matrix with each rows containing the coordinates of a structure
        
        eta:
        Determines the with of the gaussian in the radial feature. 
        Which determines the radial resolution of the feature.
        
        Xi:
        Determines the width the peaks in the angular distribution.
        
        cutoffRadius:
        Sets the cutoff radius.
        
        lamb:
        Determines the positions of the angular peaks.
        The defaule lamb=[-1,1] is probably fine.
        
        Rs:
        Determines the positions of the radial peaks.        
        """
        self.X = X
        self.cutoffRadius = cutoffRadius
        self.lamb=lamb
        self.Xi = Xi
        self.eta = eta
        self.Rs = Rs

        self.Nf_r = len(self.Rs)  # Radial
        self.Nf_a = len(self.lamb)*len(self.Xi)  # Angular
        self.Nf = self.Nf_r + self.Nf_a  # Total number of features
        
    def get_FeatureMat(self, X):
        Ndata = np.shape[0]
        Natoms = int(X.shape(1)/2)
        featureMat = np.zeros((Ndata*Natoms, self.Nf))
        for i in range(Ndata):
            featureMat[Natoms*i:Natoms*(i+1), :] = getFeatureMat_singleStructure(X[i])
        
    def getFeatureMat_singleStructure(self, pos):
        Natoms = int(pos.shape[0]/2)
        featureMat_singleStruct = nparray([getFeature_singleAtom(pos, i) for i in range(Natoms)])
        return featureMat_singleStruct
        
    def getFeature_singelAtom(self, pos, index):
        """
        Calculate the feature vector of atom 'index' in the structure given
        by the coordinateset 'pos'. 
        where pos = [x0,y0,x1,y1, ...]
        """
        # Initialize featurevector.
        featureVector = np.zeros(self.Nf_r+self.Nf_a)

        Natoms = int(pos.shape[0]/2)
        x0, y0 = pos[2*index:2*(index+1)]

        # Calculate all radial features
        for s in range(Nf_r):
            Rs = self.Rs[s]
            f1 = 0
            for j in range(Natoms):     # Calculate radial part
                if j == index:
                    continue
                x, y = pos[2*j:2(j+1)]
                Rij = np.sqrt((x0 - x)**2 + (y0 - y)**2)
                if Rij <= self.Rc:
                    f1 += np.exp(- eta * (Rij - Rs)**2 / self.Rc**2) * self.cutOffFunction(Rij)
            featureVector[s] = f1
            
        # Calculate all angular features
        for p in range(Nf_a):
            f2 = 0
            for j in range(Natoms):
                if j == i:
                    continue
                for k in range(j+1, Natoms):
                    if k == i:
                        continue
                    # Calculate the distances between atoms
                    RijVec = pos[2*j:2*(j+1)] - pos[2*index:2*(index+1)]
                    Rij = np.linalg.norm(RijVec)
                    
                    RikVec = pos[2*k:2*(k+1)] - pos[2*index:2*(index+1)]
                    Rik = np.linalg.norm(RikVec)
                    
                    RjkVec = pos[2*k:2*(k+1)] - pos[2*j:2*(j+1)]
                    Rjk = np.linalg.norm(RjkVec)
                    
                    f2 += (1 + lamb * np.dot(RijVec, RikVec) / (Rij * Rik))**xi * np.exp(- eta * (Rij * Rij + Rik * Rik + Rjk * Rjk) / self.Rc**2) * self.cutOffFunction(Rij) * self.cutOffFunction(Rik) * self.cutOffFunction(Rjk)
            f2 *= 2**(1 - xi)
            featureVector[Nf_r + p] = f2
        return featureVector

    def cutOffFunction(self, r):
        if r <= self.Rc:
            return 0.5 * (1 + np.cos(np.pi * r / self.Rc))
        return 0
        
        
