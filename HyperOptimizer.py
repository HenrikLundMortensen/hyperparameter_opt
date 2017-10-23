import numpy as np
import IPython
import time
from scipy.optimize import minimize


class HyperOptimizer():
    """
    Class used for optimizing hyperparameters. Only for sigma in KRR right now. 
    """

    def __init__(self,krr,sig=1,G=None,E=None,runs=100,k=5,hlr=1,tol = 1e-5):
        """
        Initialize. 

        Args:
        krr: Kernel Ridge Regression model class
        G: Feature matrix of the examples
        E: Energies of the examples
        runs: Number of runs (steps) in the optimizer
        k: For k-fold cross validation
        hlr: Hyper learning rate (stepsize in sigma optimization)
        tol: tolerance for abs(gradient) in optimizer
        
        """
        self.krr = krr
        self.G = G
        self.Ntrain = G.shape[0]
        self.sig_start = sig
        self.runs = runs
        self.k = k
        self.E = E
        self.hlr = hlr
        self.tol = tol

        # Create split indices
        self._split()


    def _split(self):
        """
        Create indicies for training and validation. Splits N_samples permuted indicies into k parts. 
        Also creates the batch_id matrix, such that the diagonal of batch_id is False, and all other is
        True.
        """
        
        N_samples = self.G.shape[0]
        self.ids = np.random.permutation(N_samples)
        self.ids_split = np.array(np.split(self.ids[0:int(N_samples/self.k)*self.k],self.k))
        self.batch_id = np.ones((self.k,self.k),dtype=bool) - np.identity(self.k,dtype=bool)


    def _get_train_and_val(self,ki):
        """
        Extracts and returns the training and validation set based on the ki'th entry of the batch_id matrix.

        Args: 
        ki: The i'th fold in the k-fold cross validation.
        """
        
        # Pick out the training parts and concatenate
        ids_train = np.concatenate(self.ids_split[self.batch_id[ki]])

        # The remaining part is validation 
        ids_val = self.ids_split[ki]

        Gtrain = self.G[ids_train]
        Etrain = self.E[ids_train]
        Gval = self.G[ids_val]    
        Eval = self.E[ids_val]

        return Gtrain,Gval,Etrain,Eval,ids_train



    def gridSearch(self,sigma_guess_arr,verbose=False):
        """
        Performs gridsearch over sigma values in sigma_guess_arr
        """
        Nsig = len(sigma_guess_arr)
        err = np.zeros(Nsig)
        
        for i in range(Nsig):
            err[i],_= self._sigmaObjective(sigma_guess_arr[i],getGrad=False)
            if verbose:
                print('Grid Search:\tError = %4.6f\t sigma = %4.6f' %(err[i],sigma_guess_arr[i]))
        return sigma_guess_arr[np.argmin(err)]
        
    def optimize(self):
        """
        Performs the optimization with cross validation. The optimal sigma is set in
        self.krr.comparator.sigma.
        """

        error = np.zeros(self.k)
        
        # Initial values for sigma
        self.krr.comparator.sigma = self.sig_start

        for i in range(self.runs):
            grad = 0

            # Cross validation
            for ki in range(self.k):
                Gtrain,Gval,Etrain,Eval,_ = self._get_train_and_val(ki)
                self.krr.fit(Etrain,featureMat=Gtrain)

                # Calculate error
                Epred = self.krr.predict(Gtrain,Gval)
                error[ki] = (Eval-Epred).T@(Eval-Epred)
                
                # Gradients from each cross validation fold are added
                grad += self.krr.getSigmaGradient(Gtrain,Gval,Etrain,Eval)
            
            print('grad = %4.12f, Error_val = %4.12f' %(grad,np.mean(error)))

            # Update sigma
            self.krr.comparator.sigma = self.krr.comparator.sigma - self.hlr*grad/self.k
            if abs(grad) < self.tol:
                break

        self.optimal_sig = self.krr.comparator.sigma
        self.krr.fit(self.E,featureMat=self.G)


    def optimizeSigmaVec(self,method='constant_hlr',verbose=False):
        """
        """
        # Initial values for sigma

        if method=='constant_hlr':
            minerr = 0
            for i in range(self.runs):
                error,grad = self._sigmaVecObjective(self.krr.sigmaVec)

                if minerr != 0:
                    if minerr > error:
                        minerr = error
                        minsigmaVec = self.krr.sigmaVec
                else:
                    minerr = error
                    minsigmaVec = self.krr.sigmaVec
                
                self.krr.sigmaVec = self.krr.sigmaVec - self.hlr*grad

                if verbose:
                    print('Gradient Descent:\tError = %4.6f' %(error))
        else:
            options={'maxiter': self.runs}
            res = minimize(fun=self._sigmaVecObjective,
                           x0=self.krr.sigmaVec,
                           method=method,
                           jac=True,
                           options=options)
            self.krr.sigmaVec = res.x
            

        self.krr.sigmaVec = minsigmaVec
        self.optimal_sigVec = self.krr.sigmaVec
        self.krr.fitSV(self.E,featureMat=self.G,sigmaVec = self.krr.sigmaVec)



    def initializeSigmaVec(self,m,q):
        """
        Calculates an initial guess for the sigmaVec based on the feature space density
        """

        # Get distance matrix
        distmat = np.sqrt(self.krr.comparator.getDistMat(self.G,self.G))

        # Sort and remove the first column after sorting (which are always zeros)
        distmat = np.sort(distmat,axis=1)[:,1:]

        # Do exponential scaling of the distance and sum the m nearest points
        # sigmaVec = np.sum(np.exp(-q*np.power(distmat[:,0:m],2)),axis=1)
        # sigmaVec = np.mean(np.sqrt(distmat[:,0:m]),axis=1)
        sigmaVec = np.mean(distmat[:,0:m],axis=1)
        return sigmaVec
        
        
        

    def _sigmaObjective(self,sigma,getGrad=True):
        """
        """

        grad = 0
        error = np.zeros(self.k)

        self.krr.comparator.sigma = sigma
        # Cross validation
        for ki in range(self.k):
            Gtrain,Gval,Etrain,Eval,ids_for_train = self._get_train_and_val(ki)
            self.krr.fit(Etrain,featureMat=Gtrain)

            # Calculate error
            Epred = self.krr.predict(Gtrain,Gval)
            error[ki] = (Eval-Epred).T@(Eval-Epred)

            if getGrad:
                # Gradients from each cross validation fold are added
                grad += self.krr.getSigmaGradient(Gtrain,Gval,Etrain,Eval)

        error = np.mean(error)

        return error,self.Ntrain*grad
        

    def _sigmaVecObjective(self,sigmaVec,getGrad=True):
        """
        Performs k-fold cross validation and returns validation error and hyper parameter gradient.
        For use with scipy.optimize.minimize
        """

        grad = np.zeros(len(self.ids))
        error = np.zeros(self.k)
        tic = time.time()
        # Cross validation
        for ki in range(self.k):
            Gtrain,Gval,Etrain,Eval,ids_for_train = self._get_train_and_val(ki)
            self.krr.fitSV(Etrain,featureMat=Gtrain,sigmaVec=sigmaVec[ids_for_train])

            # Calculate error
            Epred = self.krr.predictSigmaVec(Gtrain,Gval,sigmaVec=sigmaVec[ids_for_train])
            error[ki] = (Eval-Epred).T@(Eval-Epred)

            if getGrad:
                # Gradients from each cross validation fold are added
                grad[ids_for_train] += self.krr.getSigmaVecGradient(Gtrain,Gval,Etrain,Eval,sigmaVec[ids_for_train])            
                
            
        total_time = time.time() - tic
        error = np.mean(error)

        # print('Error = %4.12f, grad_time = %4.12f, total_time = %4.12f' %(error,grad_time,total_time))
        # print('Error = %4.12f' %(error))        
        return error,grad

        



    # def optimizeSigmaVec(self):
    #     """
    #     """
    #     # Initial values for sigma
    #     self.krr.comparator.sigma = self.sig_start
    #     error_copy = 0
    #     for i in range(self.runs):
    #         grad = np.zeros(len(self.ids))

    #         error = np.zeros(self.k)
    #         # Cross validation
            
    #         for ki in range(self.k):
    #             Gtrain,Gval,Etrain,Eval,ids_for_train = self._get_train_and_val(ki)
    #             self.krr.fitSV(Etrain,featureMat=Gtrain,sigmaVec=self.krr.sigmaVec[ids_for_train])

    #             # Calculate error
    #             Epred = self.krr.predictSigmaVec(Gtrain,Gval,sigmaVec=self.krr.sigmaVec[ids_for_train])
    #             error[ki] = (Eval-Epred).T@(Eval-Epred)

    #             # Gradients from each cross validation fold are added
    #             grad[ids_for_train] += self.krr.getSigmaVecGradient(Gtrain,Gval,Etrain,Eval,self.krr.sigmaVec[ids_for_train])

    #         error = np.mean(error)
    #         print('Error_val = %4.12f' %(error))
    #         # Update sigma
    #         self.krr.sigmaVec = self.krr.sigmaVec - self.hlr*grad/self.k
    #         if abs(error_copy-error) < self.tol:
    #             break
    #         error_copy = error

    #     self.optimal_sigVec = self.krr.sigmaVec
    #     self.krr.fitSV(self.E,featureMat=self.G,sigmaVec = self.krr.sigmaVec)
    


