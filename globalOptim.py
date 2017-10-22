import numpy as np
from scipy.optimize import minimize
from doubleLJ import doubleLJ
import matplotlib.pyplot as plt

class globalOptim():
    """
    --Input--
    Efun:
    Function that returns energy of a structure given
    atomic positions in the form [x0, y0, x1, y1, ...]
    
    gradfun:
    Function that returns energy of a structure given
    atomic positions in the form [x0, y0, x1, y1, ...]

    MLmodel:
    Model that given training data can predict energy and gradient of a structure.
    Hint: must include a fit, predict_energy and predict_force methods.
    
    Natoms:
    Number of atoms in structure.
    
    Niter:
    Number of monte-carlo steps.

    boxsize:
    Side length of the square in which the atoms are confined.

    dmax:
    Max translation of each coordinate when perturbing the current structure to
    form a new candidate.
    
    sigma:
    Variable controling how likly it is to accept a worse structure.
    Hint: Should be on the order of the energy difference between local minima.

    Nstag:
    Max number of iterations without accepting new structure before
    the search is terminated.
    """
    def __init__(self, Efun, gradfun, MLmodel=None, Natoms=6, Niter=50, boxsize=None, dmax=0.1, sigma=1, Nstag=5,
                 maxIterLocal=10, fracPerturb=0.4, radiusRange = [0.9, 1.5]):
        self.Efun = Efun
        self.gradfun = gradfun
        self.MLmodel = MLmodel
        self.Natoms = Natoms
        if boxsize is not None:
            self.boxsize = boxsize
        else:
            self.boxsize = 1.5 * np.sqrt(self.Natoms)
        self.bounds = [(0, boxsize)] * Natoms * 2
        self.dmax = dmax
        self.Niter = Niter
        self.sigma = sigma
        self.Nstag = Nstag
        self.maxIterLocal = maxIterLocal
        self.Nperturb = max(2, int(self.Natoms*fracPerturb))
        self.rmin = radiusRange[0]
        self.rmax = radiusRange[1]

        # Initialize arrays to store structures for model training
        self.Xsaved = np.zeros((4000, 2*Natoms))
        self.Esaved = np.zeros(4000)
        # initialize index to keep track of the ammount of data saved
        self.ksaved = 0
        
        ### Statistics ###
        # Function evaluations
        self.Nfeval = 0
        # Predicted energy of structure relaxed with ML model
        self.ErelML = []
        # Energy of structure relaxed with true potential
        self.ErelTrue = []
        # True energy of resulting from relaxation with ML model
        self.ErelMLTrue = []
        # Energy of structure relaxed with ML model followed by relaxation with true potential
        self.E2rel = []
        # Predicted energy of unrelaxed structure
        self.EunrelML = []
        # Energy of unrelaxed structure
        self.Eunrel = []
        # All force components of unrelaxed structure (ML)
        self.FunrelML = []
        # All force components of unrelaxed structure (True)
        self.FunrelTrue = []
        # The number of training data used
        self.ktrain = []

        self.testCounter = 0
        self.Ntest_array = np.logspace(1, 3, 30)
        
    def runOptimizer(self):
        self.makeInitialStructure()
        self.Ebest = self.E
        self.Xbest = self.X
        k = 0
        for i in range(self.Niter):
            if  False:  # self.ksaved > self.Ntest_array[self.testCounter]:
                print('ksaved=', self.ksaved)
                self.testCounter += 1
                self.trainModel()
                self.ktrain.append(self.ksaved)
                Enew_unrelaxed, Xnew_unrelaxed = self.makeNewCandidate()
                Enew, Xnew = self.relax(Xnew_unrelaxed)
                ErelML, XrelML = self.relax(Xnew_unrelaxed, ML=True)
                ErelML_relax, XrelML_relax = self.relax(XrelML)
                #self.plotStructures(Xnew, XrelML, Xnew_unrelaxed)
                ErelMLTrue = self.Efun(XrelML)
                # Data for relaxed energies
                self.ErelML.append(ErelML)
                self.ErelMLTrue.append(ErelMLTrue)
                self.ErelTrue.append(Enew)
                self.E2rel.append(ErelML_relax)
                # Data for unrelaxed energies
                self.Eunrel.append(Enew_unrelaxed)
                self.EunrelML.append(self.MLmodel.predict_energy(pos=Xnew_unrelaxed))
                # Data for unrelaxed forces
                FnewML = self.MLmodel.predict_force(pos=Xnew_unrelaxed)
                FnewTrue = -self.gradfun(Xnew_unrelaxed)
                self.FunrelML.append(FnewML)
                self.FunrelTrue.append(FnewTrue)
            else:
                Enew_unrelaxed, Xnew_unrelaxed = self.makeNewCandidate()
                Enew, Xnew = self.relax(Xnew_unrelaxed)

            print(self.ksaved)    

            dE = Enew - self.E
            if dE <= 0:  # Accept better structure
                self.E = Enew
                self.X = Xnew
                k = 0
                if Enew < self.Ebest:  # Update the best structure
                    self.Ebest = Enew
                    self.Xbest = Xnew
            else:
                p = np.random.rand()
                if p < np.exp(-dE/self.sigma):  # Accept worse structure
                    self.E = Enew
                    self.X = Xnew
                    k = 0
                else:  # Decline structure
                    k += 1
            
            if k >= self.Nstag:  # The search has converged or stagnated.
                print('The convergence/stagnation criteria was reached')
                break
            # print('E=', self.E)

            if self.ksaved > 1500:  # self.testCounter > 29:
                break
        
    def makeInitialStructure(self):
        """
        Generates an initial structure by placing an atom one at a time inside
        a square of size self.boxsize and cheking if it satisfies the constraints
        based on the previously placed atoms.
        
        Constraints:
        1) Distance to all atoms must be > self.rmin
        2) Distance to nearest neighbor must be < self.rmax
        """
        def validPosition(X, xnew):
            Natoms = int(len(X)/2) # Current number of atoms
            if Natoms == 0:
                return True
            connected = False
            for i in range(Natoms):
                r = np.linalg.norm(xnew - X[2*i:2*i+2])
                if r < self.rmin:
                    return False
                if r < self.rmax:
                    connected = True
            return connected
        
        Xinit = np.zeros(2*self.Natoms)
        for i in range(self.Natoms):
            while True:
                xnew = np.random.rand(2) * self.boxsize
                if validPosition(Xinit[:2*i], xnew):
                    Xinit[2*i:2*i+2] = xnew
                    break
        
        self.E, self.X = self.relax(Xinit)
        
    def makeNewCandidate(self):
        """
        Makes a new candidate by perturbing a subset of the atoms in the
        current structure.

        The perturbed atoms are added to the fixed atoms one at a time.

        Pertubations are applied to an atom until it satisfies the
        constraints with respect to the fixed atoms and the previously
        placed perturbed atoms.

        Constraints:
        1) Distance to all atoms must be > self.rmin
        2) Distance to nearest neighbor must be < self.rmax
        """

        # Function to determine if perturbation of a single atom is valid
        # with respect to a set of static atoms.
        def validPerturbation(X, index, perturbation, index_static):
            connected = False
            xnew = X[2*index:2*index+2] + perturbation
            for i in index_static:
                if i == index:
                    continue
                r = np.linalg.norm(xnew - X[2*i:2*i+2])
                if r < self.rmin:
                    return False
                if r < self.rmax:
                    connected = True
            return connected

        InitialStructureOkay = np.array([validPerturbation(self.X, i, np.array([0,0]), np.arange(self.Natoms))
                                         for i in range(self.Natoms)])
        if not np.all(InitialStructureOkay):
            print('Einit:', self.Efun(self.X))
            assert np.all(InitialStructureOkay)

        # Pick atoms to perturb
        i_permuted = np.random.permutation(self.Natoms)
        i_perturb = i_permuted[:self.Nperturb]
        i_static = i_permuted[self.Nperturb:]
        Xperturb = self.X.copy()
        for i in i_perturb:
            # Make valid perturbation on this atom
            while True:
                perturbation = 2*self.dmax * (np.random.rand(2) - 0.5)
            
                # check if perturbation rersult in acceptable structure
                if validPerturbation(Xperturb, i, perturbation, i_static):
                    Xperturb[2*i:2*i+2] += perturbation
                    # Add accepted perturbet atom to static set
                    i_static = np.append(i_static, i)
                    break

        # Save structure for training
        Eperturb = self.Efun(Xperturb)
        self.Xsaved[self.ksaved] = Xperturb
        self.Esaved[self.ksaved] = Eperturb
        self.ksaved += 1
        
        return Eperturb, Xperturb

    def trainModel(self):
        self.MLmodel.fit(self.Esaved[:self.ksaved], positionMat=self.Xsaved[:self.ksaved])
        #GSkwargs = {'reg': np.logspace(-6, -3, 5), 'sigma': np.logspace(-1, 1, 5)}
        #MAE, params = self.MLmodel.gridSearch(self.Esaved[:self.ksaved], positionMat=self.Xsaved[:self.ksaved], **GSkwargs)
        #print('sigma:', params['sigma'], 'reg:', params['reg'])
        
    def relax(self, X=None, ML=False):
        ## determine which model to use for potential ##
        if ML:
            # Use ML potential and forces
            def Efun(pos):
                return self.MLmodel.predict_energy(pos=pos) + self.artificialPotential(pos)
            def gradfun(pos):
                return -self.MLmodel.predict_force(pos=pos)
            # Set up minimizer
            options = {'gtol': 1e-5}
            def localMinimizer(X):
                res = minimize(Efun, X, method="L-BFGS-B", jac=gradfun,
                               bounds=self.bounds, options=options)
                return res
        else:
            # Use double Lennard-Johnes
            # Set up Minimizer
            options = {'maxiter': self.maxIterLocal}  # , 'gtol': 1e-3}
            def localMinimizer(X):
                res = minimize(self.Efun, X, method="L-BFGS-B", jac=self.gradfun, tol=1e-5,
                               bounds=self.bounds, options=options)
                return res

        ## Run Local minimization ##
        if ML is False:
            # Minimize using double-LJ potential + save trajectories
            X0 = X.copy()
            for i in range(100):
                res = localMinimizer(X0)
                # Save training data
                if res.fun < 0:
                    self.Xsaved[self.ksaved] = res.x
                    self.Esaved[self.ksaved] = res.fun
                    self.ksaved += 1
                # print('Number of iterations:', res.nit, 'fev:', res.nfev)
                if res.success: # Converged
                    break
                X0 = res.x
            return self.Esaved[self.ksaved-1], self.Xsaved[self.ksaved-1]
        else:
            # Minimize using ML potential
            res = localMinimizer(X)
            print('Number of ML iterations:', res.nit, 'fev:', res.nfev)
            print('Convergence status:', res.status)
            print(res.message)
            return res.fun, res.x

    def artificialPotential(self, x):
        N = x.shape[0]
        Natoms = int(N/2)
        x = np.reshape(x, (Natoms, 2))
        E = 0
        for i in range(Natoms):
            for j in range(i+1, Natoms):
                r = np.sqrt(np.dot(x[i] - x[j], x[i] - x[j]))
                if r < self.rmin:
                    E += 1e4 * (self.rmin - r)
        return E


    def plotStructures(self, X1=None, X2=None, X3=None):
        xbox = np.array([0, self.boxsize, self.boxsize, 0, 0])
        ybox = np.array([0, 0, self.boxsize, self.boxsize, 0])

        plt.gca().cla()
        x1 = X1[0::2]
        y1 = X1[1::2]
        x2 = X2[0::2]
        y2 = X2[1::2]
        x3 = X3[0::2]
        y3 = X3[1::2]
        plt.plot(xbox, ybox, color='k')
        plt.scatter(x1, y1, s=22, color='r')
        plt.scatter(x2, y2, s=22, color='b')
        plt.scatter(x3, y3, s=22, color='g', marker='x')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.pause(4)
        
