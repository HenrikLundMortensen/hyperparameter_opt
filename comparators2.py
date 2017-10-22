from copy import copy
from scipy.special import erf
from ase.ga.atoms_attach import enable_features_methods
import numpy as np
from itertools import product


def cosine_dist(f1,f2):
    norm1 = np.linalg.norm(f1)
    norm2 = np.linalg.norm(f2)
    distance = np.sum(np.array(f1)*np.array(f2))/(norm1*norm2)
    
    cos_dist = 0.5*(1-distance)
        
    return cos_dist

def norm2_dist(f1,f2):
    distance = np.linalg.norm(np.array(f1)-np.array(f2))
    
    return distance

def norm1_dist(f1,f2):
    distance = np.sum(abs(np.array(f1)-np.array(f2)))

    return distance

class BagOfBonds(object):

    def __init__(self, a_opt=None, n_top=None, pair_cor_cum_diff=0.02,
                 pair_cor_max=0.7, dE=0.5, mic=False, excluded_types=[]):
        self.pair_cor_cum_diff = pair_cor_cum_diff
        self.pair_cor_max = pair_cor_max
        self.dE = dE
        self.n_top = n_top or 0
        self.numbers = a_opt[-self.n_top:].numbers
        self.mic = mic
        self.excluded_types = excluded_types

    def get_features(self,a):
        """ Utility method used to calculate interatomic distances
            returned as a dict sorted after atomic type
            (e.g. (6,6), (6,42), (42,42)). 
        """
        atoms = a[-self.n_top:]

        unique_types = sorted(list(set(self.numbers)))
        unique_types = [u for u in unique_types if u not in self.excluded_types]
        pair_cor = {}
        for idx, u1 in enumerate(unique_types):
            i_u1 = [i for i in range(len(atoms)) if atoms[i].number == u1]
            for u2 in unique_types[idx:]:
                i_u2 = [i for i in range(len(atoms)) if atoms[i].number == u2]
                d = []
                if u1 == u2:
                    for i, n1 in enumerate(i_u1):
                        for n2 in i_u2[i+1:]:
                            d.append(float(atoms.get_distance(n1,n2,self.mic)))
                else:
                    for i, n1 in enumerate(i_u1):
                        for n2 in i_u2:
                            d.append(float(atoms.get_distance(n1,n2,self.mic)))

                d.sort()
                if len(d) == 0:
                    continue
                pair_cor[(u1,u2)] = d

        enable_features_methods(a)
        a.set_features(pair_cor)
        return pair_cor

    def get_features_atoms(self,a):
        """ Utility method used to calculate interatomic distances
            returned a set of dicts sorted after atomic type
            (e.g. (6,6), (6,42), (42,42)). The first entry in the set 
            is feature of the structure
        """
        atoms = a[-self.n_top:]
        results = []
        d = np.zeros([len(atoms),len(atoms)])
        type_chart = [atoms[i].number for i in range(len(atoms))]
        unique_types = sorted(list(set(self.numbers)))
        unique_types = [u for u in unique_types if u not in self.excluded_types]
        n = 0
        type_dict = {}
        for idx,u1 in enumerate(unique_types):
            type_dict[u1] = np.count_nonzero(type_chart == u1)
            for u2 in unique_types[idx:]:
                type_dict[(u1,u2)] = n
                type_dict[(u2,u1)] = n
                n += 1
        mask = np.zeros([d.shape[0],d.shape[1],n], dtype=bool)
        for i in range(len(atoms)):
            for j in range(i+1,len(atoms)):
                d[i,j] = atoms.get_distance(i,j,self.mic)
                bond_type = type_dict[(type_chart[i],type_chart[j])]
                mask[i,j,bond_type] = True
        results.append({})
        for idx,u1 in enumerate(unique_types):
            for u2 in unique_types[idx:]:
                results[0][(u1,u2)] = sorted(d[mask[:,:,type_dict[(u1,u2)]]])
        for i in range(len(atoms)):
            temp_mask = copy(mask)
            temp_mask[i,:,:] = False
            temp_mask[:,i,:] = False
            results.append({})
            for idx,u1 in enumerate(unique_types):
                for u2 in unique_types[idx:]:
                    without_atom = sorted(d[temp_mask[:,:,type_dict[(u1,u2)]]])
                    if type_chart[i] == u1 and type_chart[i] == u2:
                        missing_bonds = type_dict[u1]-1
                    elif type_chart[i] == u1:
                        missing_bonds = type_dict[u2]
                    elif type_chart[i] == u2:
                        missing_bonds = type_dict[u1]
                    else:
                        missing_bonds = 0
                    if len(without_atom):
                        with_atom = np.append(without_atom,[without_atom[-1]]*missing_bonds)
                        results[i+1][(u1,u2)] = list(with_atom)
                    else:
                        with_atom = np.array([0]*missing_bonds)
                        results[i+1][(u1,u2)] = list(with_atom)
        return results

    def get_similarity(self,f1,f2):
        """ Method for calculating the similarity between two objects with features
        f1 and f2, respectively.
        """
        if isinstance(f1,np.ndarray):
            f1 = f1[0]

        d1 = np.asarray(sum(zip(*sorted(f1.items()))[1],[]))
        d2 = np.asarray(sum(zip(*sorted(f2.items()))[1],[]))

        d_norm = (np.sum(d1)+np.sum(d2))/2.
            
        df = np.abs(d1-d2)

        cum_diff = np.sum(df)

        s = cum_diff / d_norm 
        return s

    def looks_like(self, a1, a2, check_fragments=False):
        from ase.ga.utilities import get_fragments

        # Energy criterium
        dE = abs(a1.get_potential_energy() - a2.get_potential_energy())
        if dE >= self.dE:
            return False

        structures = [a1,a2]

        # Check for fragments of equal composition
        compare_fragments = False
        if check_fragments:
            fragment_structures = []

            fragments = []
            for a in structures:
                enable_features_methods(a)
                if a.get_fragments() == None:
                    a_fragments = get_fragments(a)
                    a.set_fragments(a_fragments)
                    fragments.append(a_fragments)
                else:
                    fragments.append(a.get_fragments())

            a1_fragments = fragments[0]
            a2_fragments = fragments[1]

            if len(a1_fragments) == len(a2_fragments) and len(a1_fragments) > 1:
                compare_fragments = True
                for idx in range(len(a1_fragments)):
                    fragment1 = a1_fragments[idx]
                    fragment2 = a2_fragments[idx]
                    if not sorted(a1[fragment1].numbers) == \
                            sorted(a2[fragment2].numbers):
                        compare_fragments = False
                        break
                
                    fragment_structures.append(a1[fragment1])
                    fragment_structures.append(a2[fragment2])

                if compare_fragments:
                    structures = fragment_structures

        # Structure criterium
        ds = []
        for a in structures:
            enable_features_methods(a)
            if a.get_features() == None or compare_fragments:
                d = self.get_features(a)
#                if len(d) == 0:
#                    continue
                ds.append(d)
            else:
                ds.append(a.get_features())

        a1_ds = ds[0::2]
        a2_ds = ds[1::2]

        max_d12 = []
        for idx in range(len(a1_ds)):
            for key in a1_ds[idx].keys():
                d1 = a1_ds[idx][key]
                d2 = a2_ds[idx][key]
                max_d12.append(max(np.abs(np.array(d1)-np.array(d2))))
            if self.get_similarity(a1_ds[idx],a2_ds[idx]) > self.pair_cor_cum_diff:
                return False
            
        if max(max_d12) > self.pair_cor_max:
            return False

#        print self.get_similarity(a1_ds[idx],a2_ds[idx]), max(max_d12)
        return True

class SeaOfBonds(object):

    def __init__(self, a_opt, n_top=None, rcut=20., binwidth=0.5, 
                 excluded_types=[], sigma='auto', nsigma=4, 
                 pbc=[True,True,False]):
        self.n_top = n_top or 0
        self.numbers = a_opt[-self.n_top:].numbers
        self.cell = a_opt.get_cell()
        self.rcut = rcut
        self.binwidth = binwidth
        self.excluded_types = excluded_types
        if sigma == 'auto':
            self.sigma = 0.5*binwidth
        else:
            self.sigma = sigma
        self.nsigma = nsigma
        self.pbc = pbc

    def get_features(self, a):
        atoms = a[-self.n_top:]

        unique_types = sorted(list(set(self.numbers)))
        unique_types = [u for u in unique_types if u not in self.excluded_types]

        # Include neighboring unit cells within rcut
        cell_vec_norms = np.apply_along_axis(np.linalg.norm,0,self.cell)
        cell_neighbors = []
        for i in range(3):
            ncell_max = int(np.floor(self.rcut/cell_vec_norms[i]))
            if self.pbc[i]:
                cell_neighbors.append(range(-ncell_max,ncell_max+1))
            else:
                cell_neighbors.append([0])

        # Binning parameters
        m = int(np.ceil(self.nsigma*self.sigma/self.binwidth))
        nbins = int(np.ceil(self.rcut*1./self.binwidth))
        smearing_norm = erf(self.binwidth*(2*m+1)*1./(np.sqrt(2)*self.sigma)) # Correction to nsigma cutoff
        
        # Get interatomic distances
        pos = atoms.get_positions()

        pair_cor = {}    
        for idx, u1 in enumerate(unique_types):
            i_u1 = [i for i,atom in enumerate(atoms) if atom.number == u1]
            for u2 in unique_types[idx:]:
                i_u2 = [i for i,atom in enumerate(atoms) if atom.number == u2]
                rdf = np.zeros(nbins)
                for n1 in i_u1:
                    pi = np.array(pos[n1])
                    for disp_vec in product(*cell_neighbors):
                        displacement = np.dot(self.cell.T,np.array(disp_vec).T)
                        if u1 == u2 and disp_vec == (0,0,0): # Avoid self-counting in unit cell
                            pj = [p for n2,p in enumerate(pos) if n2 in i_u2 and n2 == n1]
                        else:
                            pj = [p for n2,p in enumerate(pos) if n2 in i_u2]
                        if len(pj) == 0:
                            continue

                        displaced_pos = pj + displacement
                        ds = np.apply_along_axis(np.linalg.norm,1,
                                                 displaced_pos-pi) # Interatomic distances
                            
                        for dij in ds:
                            rbin = int(np.floor(dij/self.binwidth)) # Bin for dij
                            for i in range(-m,m+1): # Bins corresponding to (+/-)nsigma*sigma from rbin
                                newbin = rbin + i
                                if newbin < 0 or newbin >= nbins:
                                    continue

                                c = 1./(np.sqrt(2)*self.sigma)
                                value = 0.5*erf(c*((newbin+1)*self.binwidth - dij)) - 0.5*erf(c*(newbin*self.binwidth - dij))

                                value /= smearing_norm
                                rdf[newbin] += value

                pair_cor[(u1,u2)] = rdf
                
        return pair_cor

    def get_similarity(self,f1,f2):
        """ Method for calculating the similarity between two objects with features
        f1 and f2, respectively.
        """

        # Calculate similarity.
        s = 0.
        for key in sorted(f1.keys()):
            d1 = f1[key]
            d2 = f2[key]

            while len(d1) < len(d2):
                d1.append(0.)
            while len(d2) < len(d1):
                d2.append(0.)

            d_norm = np.sum(np.mean([d1,d2],axis=0))

            ntype1 = float(sum([i == key[0] for i in self.numbers]))
            ntype2 = float(sum([i == key[1] for i in self.numbers]))
            ntype = np.mean([ntype1,ntype2])

            df = np.abs(np.array(d1)-np.array(d2))
            cum_diff = np.sum(df)

            s += cum_diff / d_norm# * ntype / float(len(self.numbers))
    
        return s

    def cosine_dist(self,f1,f2,numbers):
        keys = sorted(f1)
        unique_numbers = sorted(list(set(numbers)))
        typedic = {}
        for u in unique_numbers:
            typedic[(u)] = sum([u == i for i in numbers])
        
        w = {}
        wtot = 0
        for key in keys:
            while len(f1[key]) < len(f2[key]):
                f1[key].append(0.)
            while len(f2[key]) < len(f1[key]):
                f2[key].append(0.)

            weight = typedic[key[0]]*typedic[key[1]]
            wtot += weight
            w[key] = weight
        for key in keys:
            w[key] *= 1./wtot

        norm1 = 0
        norm2 = 0
        for key in keys:
            norm1 += (np.linalg.norm(f1[key])**2)*w[key]
            norm2 += (np.linalg.norm(f2[key])**2)*w[key]
        norm1 = np.sqrt(norm1)
        norm2 = np.sqrt(norm2)

        distance = 0.
        for key in keys:
            distance += np.sum(np.array(f1[key])*np.array(f2[key]))*w[key]/(norm1*norm2)

        distance = 0.5*(1-distance)

        return distance

#        norm1 = 0.
#        norm2 = 0.
#        for key in sorted(f1.keys()):
#            c1 = f1[key]
#            c2 = f2[key]

#            while len(c1) < len(c2):
#                c1.append(0.)
#            while len(c2) < len(c1):
#                c2.append(0.)

#            norm1 += np.linalg.norm(c1)
#            norm2 += np.linalg.norm(c2)

#        for key in sorted(f1.keys()):
#            distance += np.sum(np.array(c1)*np.array(c2))/(norm1*norm2)
            
#        cos_dist = 0.5*(1-distance)

#        return cos_dist
