import numpy as np


class bob_features():
    def __init__(self, X=None):
        """
        --input--
        X:
        Contains data. Each row represents a structure given by cartesian coordinates
        in the form [x1, y1, ... , xN, yN]
        """
        self.X = X
    
    def get_featureMat(self, X=None):
        if X is not None:
            self.X = X
        Ndata = self.X.shape[0]
        Natoms = int(self.X.shape[1]/2)
        Nfeatures = int(Natoms*(Natoms-1)/2)
        G = np.zeros((Ndata, Nfeatures))
        I = np.zeros((Ndata, Nfeatures, 2))
        for n in range(Ndata):
            G[n, :], I[n, :, :] = self.get_singleFeature(self.X[n])
        self.G = G
        self.I = I
        return self.G, self.I
    
    def get_singleFeature(self, x):
        """
        --input--
        x: atomic positions for a single structure in the form [x1, y1, ... , xN, yN]
        """
        Natoms = int(np.size(x, 0)/2)
        x = x.reshape((Natoms, 2))

        # Calculate inverse bond lengths
        g = np.array([1/np.linalg.norm(x[j]-x[i]) for i in range(Natoms) for j in range(i+1, Natoms)])
        
        # Make list of atom indices corresponding to the elements of the feature g
        atomIndices = np.array([(i, j) for i in range(Natoms) for j in range(i+1, Natoms)]).astype(int)
        
        # Get indices that sort g in decending order
        sorting_indices = np.argsort(-g)


        # Sort features and atomic indices
        g_ordered = g[sorting_indices]
        atomIndices_ordered = atomIndices[sorting_indices]
        return g_ordered, atomIndices_ordered


        # return g, atomIndices

    def get_featureGradient(self, pos, g, atomIndices):
        Nr = len(pos)
        pos = pos.reshape((int(Nr/2), 2))
        Nfeatures = np.size(g, 0)
        
        atomIndices = np.array(atomIndices).astype(int)
        # Calculate gradient of bob-feature
        gDeriv = np.zeros((Nfeatures, Nr))
        for i in range(Nfeatures):
            a0 = atomIndices[i, 0]
            a1 = atomIndices[i, 1]
            inv_r = g[i]
            gDeriv[i, 2*a0:2*a0+2] += inv_r**3*np.array([pos[a1, 0] - pos[a0, 0], pos[a1, 1] - pos[a0, 1]])
            gDeriv[i, 2*a1:2*a1+2] += -inv_r**3*np.array([pos[a1, 0] - pos[a0, 0], pos[a1, 1] - pos[a0, 1]])
        return gDeriv
