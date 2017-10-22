import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt(fname='LC_bob_N7_search2.txt', delimiter='\t')
N = data[:,0]
MAE_energy = data[:,1]
MAE_force = data[:, 2:]

plt.figure(1)
plt.title('Energy learning curve (structures from search, with CV)')
plt.loglog(N, MAE_energy)
plt.xlabel('# training data')
plt.ylabel('FVU')

plt.figure(2)
plt.title('Force learning curve (structures from search, with CV)')
plt.loglog(N, MAE_force)
plt.xlabel('# training data')
plt.ylabel('FVU')

plt.show()
