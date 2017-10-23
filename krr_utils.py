import numpy as np
import datetime

def getConstrainedStructureDataset(Ndata,Natoms):
    """
    Returns a Ndata-sized dataset with Natoms structures with spacial constrains

    boxsize = 1.5 * np.sqrt(Natoms)
    rmin = 0.9
    rmax = 1.5
    """
    
    return np.array([makeConstrainedStructure(Natoms) for i in range(Ndata)])
    
def getEnergyAndForces(potential,X,*args):
    """
    """
    Natoms,params = args
    N = X.shape[0]
    E = np.zeros(N)
    F = np.zeros((N, 2*Natoms))
    for i,x in enumerate(X):
        E[i], F[i] = potential(x, *params)

    return E,F

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




def saveModel(filename,KRR=None,HO=None,comment=None):
    """

    """
    if KRR==None:
        KRR = 'No KRR model was saved'
    if HO==None:
        HO = 'No HO object was saved'
    if comment==None:
        comment = 'No comment was saved'
    if comment=='date':
        comment = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    np.save(filename,np.array([KRR,HO,comment]))


def loadModel(filename):
    """

    """

    KRR,HO,comment = np.load(filename)[:]
    return KRR,HO,comment
    

