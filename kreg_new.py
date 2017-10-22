from __future__ import print_function
import numpy as np
import sys
from time import time
from kernelregression_new import KernelRegression
from fingerprint_kernel4 import FingerprintsComparator
from ase.calculators.calculator import Calculator, all_changes

class Kreg(Calculator):
    """ Kerenel rigide regression calculator.
    """
    implemented_properties = ['energy']
    default_parameters = {}
    nolabel = True

    def __init__(self, data, feature_matrix=None, similarity_matrix=None, sigma=None, L=1e-5, comp=None,
                 rcut=None, max_similarity=1e-6, bias='avg', **kwargs):
        """ Arguments:

        data: A list of ASE atoms objects.

        feature_matrix: A list of features in the same order as the atoms objects If this is not 
        provided one will be calculated.

        similarity_matrix: A matrix containing all similarities. If this is not provided one will be calculated.

        sigma: Width of the kernel used for kernel ridge regression. If not set one will be calculated.

        L: This is the regularization term. Normally you can just use the default value.

        comp: Kreg has a default comperator. If you wish to use another set it here. The comperator must have af 
        method named get_features_atoms() to retrive atomic features.

        rcut: Cut off radius is desired for the default feature. If not set, an optimal cut off radius wil be calculated.

        max_similarity: The threshold used to determin if two structures are identical.

        Hint. It can be very time consuming to calculate the feature and similarity matrix. If you use the same system 
        more than once consider saving the feature and the similarity for later use.
        """

        Calculator.__init__(self, **kwargs)
        self.data_values = np.array([a.get_potential_energy() for a in data])
        args = np.argsort(self.data_values)
        self.data = [data[i] for i in args]
        self.data_values[args]
        self.n_data = len(self.data)
        if comp is None:
            a = self.data[0]
            cell = a.get_cell()
            pbc = list(a.get_pbc())
            if rcut is None:
                if sum(pbc):
                    rcut = pbc[0]*max(cell[0])**2+pbc[1]*max(cell[1])**2+pbc[2]*max(cell[2])**2
                    pos = a.get_positions()
                    add = (1-pbc[0])*(max(pos.T[0])-min(pos.T[0]))**2+(1-pbc[1])*(max(pos.T[1])-min(pos.T[1]))**2+\
                        (1-pbc[2])*(max(pos.T[2])-min(pos.T[2]))**2
                    rcut = np.sqrt(rcut+2.25*add)/2.
                else:
                    rcut = np.sqrt(cell[0][0]**2+cell[1][1]**2+cell[2][2])
            self.comp = FingerprintsComparator(a, n_top=None, cell=cell, dE=1000.,
                                               cos_dist_max=max_similarity, rcut=rcut, binwidth=0.05, pbc=pbc,
                                               maxdims=[cell[0][0],cell[1][1],cell[2][2]], sigma=0.5, nsigma=4)
        else:
            self.comp = comp
        self.feature_matrix = feature_matrix
        if self.feature_matrix is None:
            self.feature_matrix = self.get_feature_matrix()
        self.similarity_matrix = similarity_matrix
        if self.similarity_matrix is None:
            self.similarity_matrix = self.get_similarity_matrix()
        self.kreg = KernelRegression(self.data_values, 5, self.feature_matrix, self.similarity_matrix, self.comp)
        self.L = L
        self.bias = bias
        if sigma is None:
            self.train()
        else:
            self.sigma = sigma
            self.kreg.sigma= sigma
        self.max_similarity = max_similarity
        
    def get_feature_matrix(self):
        """ This method returns the feature matrix. If the feature marix have not yet been calculated, it will be calculated.
        """
        if self.feature_matrix is not None:
            return self.feature_matrix
        
        print('Calculating Feature Matrix...')
        feature_matrix = []
        t1 = time()
        for i in range(self.n_data):
            feature_matrix.append(self.comp.get_features(self.data[i]))
            iteration = i + 1
            fraction_done = iteration/float(self.n_data)
            t2 = time() - t1
            time_left = (t2/iteration)*(self.n_data-iteration)
            m, s = divmod(time_left, 60)
            h, m = divmod(m, 60)
            d, h = divmod(h, 24)
            sys.stdout.flush()
            print('[{:7.2%}] Time left: [{:02.0f}:{:02.0f}:{:02.0f}:{:02.0f}] '.format(fraction_done,d,h,m,s),end='\r')
        print()

        return feature_matrix

    def get_similarity_matrix(self):
        """ This method returns the feature matrix. If the feature marix have not yet been calculated, it will be calculated.
        """
        if self.similarity_matrix is not None:
            return self.similarity_matrix

        print('Calculating Similarity Matrix...')
        similarity_matrix = np.zeros((self.n_data,self.n_data))
        t1 = time()
        for i in range(self.n_data-1):
            similarity_matrix[i][i+1:] = np.apply_along_axis(self.comp.get_similarity,
                                                             0,
                                                             [self.feature_matrix[i+1:]],
                                                             self.feature_matrix[i])
            iteration = (i+1)*self.n_data - ((i+1)**2-(i+1))/2. - (i+1)
            fraction_done = iteration/((self.n_data**2-self.n_data)/2.)
            t2 = time() - t1
            time_left = (t2/iteration)*((self.n_data**2-self.n_data)/2.-iteration)
            m, s = divmod(time_left, 60)
            h, m = divmod(m, 60)
            d, h = divmod(h, 24)
            sys.stdout.flush()
            print('[{:7.2%}] Time left: [{:02.0f}:{:02.0f}:{:02.0f}:{:02.0f}] '.format(fraction_done,d,h,m,s),end='\r')
        print()
        similarity_matrix = similarity_matrix + similarity_matrix.T

        return similarity_matrix

    def relax(self, atoms):
        """ This method relaxes a structure 'atoms' to the nearest known structure in the known data
        """
        feature = self.comp.get_features(atoms)
        for i in range(self.n_data):
            similarities = np.apply_along_axis(self.comp.get_similarity,1,np.asarray([self.feature_matrix]).T,feature)
        ix = np.argmin(similarities)
        return self.data[ix]

    def train(self, mode='grid',grid=None):
        """ This method trains the kernel and gets sigma for the kernel
        Two modes available 'grid' and 'k-fold'
        """
        if mode == 'k-fold':
            MAE, sigma, _, _ = self.kreg.cross_validation(sigma0=0.5, L=self.L, bias=self.bias)
        if mode == 'grid':
            if grid is None:
                grid = np.power(10,np.arange(-2,2,0.25))
            MAEs = np.zeros(len(grid))
            for i, sigma in enumerate(grid):
                MAEs[i], _, _, _ = self.kreg.cross_validation(sigma, opt_sigma=False, bias=self.bias)
                #print(MAEs[i])
            index = np.argmin(MAEs)
            self.kreg.sigma = grid[index]
            self.kreg.set_kernel_matrix(grid[index],range(self.n_data))
        self.sigma = grid[index]
        print('Sigma set to: {}'.format(grid[index]))

    def calculate(self, atoms=None,
                  properties=['energy'],
                  system_changes=all_changes,
                  local=False,
                  select_atoms=[] ):
        """ This method returns the energy of the structure and a list of the relative energies for the atoms
        if local_energies is set to True.
        """
        Calculator.calculate(self, atoms, properties, system_changes)  # properties and system_changes are only used here

        if local:  # calculate local eneregies
            if select_atoms == []:  # if select atoms is not set, local energy is calculated for all atoms
                select_atoms = range(len(atoms[:]))
            n_structs = len(select_atoms) + 1  # number of atoms in the structure + 1

            # calculate features for the new structure and versions of the structure with an atom missing
            new_features = self.comp.get_features_atoms(atoms)
            new_features = [new_features[i] for i in range(len(new_features)) if (i == 0) or (i-1 in select_atoms)]

            pred_values, _ = self.kreg.predict_values(new_features=new_features, L=self.L, bias=self.bias)
            energy_struct = pred_values[0]
            energy_atoms = energy_struct-np.array(pred_values[1:])
            self.results['energy'] = (energy_struct,energy_atoms)

        else:  # calculate eneregy for the input structure only (this is faster)
            feature = self.comp.get_features(atoms)  # calculate feature

            pred_values, _ = self.kreg.predict_values(new_features=[feature], L=self.L, bias=self.bias)
            energy_struct = pred_values[0]
            self.results['energy'] = energy_struct

        return self.results['energy']

    def pred_forces(self, ):

    def add_data(self,atoms):
        """This method takes an atoms object with a set energy as an input and adds it to the dataset
        """
        feature = self.comp.get_features(atoms)
        new_similarity = np.apply_along_axis(self.comp.get_similarity,
                                             0,
                                             [self.feature_matrix],
                                             feature)
        if np.any(new_similarity < self.max_similarity):
            print('This structure already exists in the dataset. The structure was discarded.')
            return
        value = atoms.get_potential_energy()
        idx = np.searchsorted(self.data_values, value)
        self.data = np.insert(self.data, idx, atoms)
        self.data_values = np.insert(self.data_values, idx, value)
        self.n_data += 1
        self.kreg.add_data(feature,value)
        self.feature_matrix = self.kreg.feature_matrix
        self.similarity_matrix = self.kreg.similarity_matrix
