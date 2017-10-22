import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import statsmodels.api as sm
from scipy import stats
from scipy.spatial import distance_matrix, KDTree

def measure(n):
    "Measurement model, return two coupled measurements."
    m1 = np.random.normal(size=n)
    m2 = np.random.normal(scale=0.5, size=n)
    return m1+m2, m1-m2


m12= np.random.rand(2000,2)
m1, m2 = measure(500)

xmin = m1.min()
xmax = m1.max()
ymin = m2.min()
ymax = m2.max()

X, Y = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
positions = np.vstack([X.ravel(), Y.ravel()])
# values = np.vstack(m12.T)
values = np.vstack([m1,m2])
kernel = ndi.gaussian_filter(values,0.2)

dm = distance_matrix(m1.reshape((len(m1),-1)),m2.reshape((len(m2),-1)))
dmsort = np.sort(dm,axis=1)

bw = np.mean(np.power(np.mean(dmsort,axis=1),1))
# bw=[bw,bw]
dens = sm.nonparametric.KDEMultivariate(data=[m1,m2], var_type='cc', bw=[bw,bw])

kdt = KDTree([m1,m2])




Z = dens.pdf(positions).reshape(200,200)
Z = Z/np.max(Z)
# kernel = stats.gaussian_kde(values)
# Z = np.reshape(kernel(positions).T, X.shape)









fig, ax = plt.subplots()
ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,
          extent=[xmin, xmax, ymin, ymax])
# ax.plot(m12.T[0], m12.T[1], 'r.', markersize=4)
ax.plot(m1, m2, 'k.', markersize=2)
# ax.set_xlim([0, 1])
# ax.set_ylim([0, 1])
ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])

plt.savefig('density.png')
