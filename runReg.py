import numpy as np
from ase import Atoms
from ase.io import read
from ase.visualize import view
from kernelregression_new import KernelRegression
from fingerprint_kernel4 import FingerprintsComparator
from kreg_new import Kreg

atoms_list = read('data_SnO.traj@0:5')

print(atoms_list[0].get_positions())
print(atoms_list[0].get_scaled_positions())
print(atoms_list[0].get_atomic_numbers())

E0 = np.array([atoms.get_potential_energy() for atoms in atoms_list])

cell = atoms_list[0].get_cell()
calc = Kreg(atoms_list, rcut=3)

"""
for atoms in atoms_list:
    atoms.set_calculator(calc)
Ereg = np.array([atoms.get_potential_energy() for atoms in atoms_list])
N = np.size(E0, 0)
MSE = 1/N*np.dot(Ereg-E0, Ereg-E0)
print(MSE)
"""



