import numpy as np
import matplotlib.pyplot as plt
import doubleLJ as dLJ


def gaussKernel(d, sigma):
    return np.exp(-1/(2*sigma**2)*d**2)


if __name__ == "__main__":
    fig, main = plt.subplots()

    eps, r0, sigma = 1.8, 1.7, np.sqrt(0.02)
    x = np.array([0, 1.1])
    y = np.array([0, 1.1])

    pos = np.c_[x, y]
    print(pos)
    pos = pos.reshape(4)
    print(pos)
    D, I = dLJ.bobFeatures(pos)
    ED, F = dLJ.doubleLJ(np.array([0, 0, 0, 1/D]), eps, r0, sigma)
    main.scatter(D, ED, color='k')

    # Plot kernel
    sig = 0.02
    d_array = np.linspace(-0.07, 0.07, 100)
    print(d_array)
    K = np.array([gaussKernel(d, sig) for d in d_array])
    main.plot(d_array+D, K+ED, color='k')
    
    left, bottom, width, height = [0.62, 0.62, 0.2, 0.2]
    inset = fig.add_axes([left, bottom, width, height])
    posis = pos.reshape((2, 2))
    inset.scatter(posis[:,0], posis[:, 1], color='k')

    
    color_array = ["r", "b", "y", "g"]
    dx = np.array([-0.2, -0.1, 0.1, 0.2])
    for i in range(4):
        pos_temp = np.array([0, 0, 1.1+dx[i], 1.1])
        inset.scatter(pos_temp[2], pos_temp[3], color=color_array[i])
        D, I = dLJ.bobFeatures(pos_temp)
        ED, F = dLJ.doubleLJ(np.array([0, 0, 0, 1/D]), eps, r0, sigma)
        main.scatter(D, ED, color=color_array[i])
    
    inset.set_xlabel('x')
    inset.set_ylabel('y')
    main.set_xlabel('Descriptor = 1/abs(r)')
    main.set_ylabel('E')









    # plot double-LJ potential in descriptor space 1/r
    r = np.linspace(0.8, 2.5, 100)
    x1 = np.array([0, 0])
    x2 = np.c_[r, np.zeros(100)]

    E = np.zeros(100)
    Fx = np.zeros(100)
    for i in range(100):
        X = np.array([x1, x2[i, :]]).reshape(4)
        E[i], F = dLJ.doubleLJ(X, eps, r0, sigma)
        Fx[i] = F[0]
    main.plot(1/r, E)
    main.set_xlim([0.4, 1.2])
    main.set_ylim([-3, 2])
    plt.show()
    
    #plt.scatter(x, y)
    #plt.show()
