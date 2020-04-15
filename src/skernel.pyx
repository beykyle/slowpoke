__author__ = "Kyle Beyer"
__email__ = "beykyle@umich.edu"

import numpy as np


def solver_eq_leth(sig_s , alpha , sig_t, sig_s_reduced, u, phi1 , du):
    """
    Numerical slowing down solver for arbirary number of nuclides and arbitrary lethargy grid
    @param sig_s          macroscopic scatter xs: list of np arrays, 1 for each nuclide. Size of each array = num groups
    @param alpha          max fractional energy loss from elastic scatter: list of floats, 1 for each nuclide
    @param sig_t          macroscopic total xs: single np array of total macro xs summed over all nuclides
    @param sig_s_reduced  macroscopic total xs: single np array scatter macro xs / (1-alpha) summed over all nuclides
    @ param u             lethargy grid: np array, size = size of cross section arrays + 1
    @ phi1                flux in first lethargy group
    """
    # set boundary condition
    num_groups = len(u) - 1 
    p = np.zeros(num_groups)
    p[0] = phi1

    # put sig_s in numpy array
    num_nuclides = len(sig_s)
    sig_s = np.vstack(sig_s)
    
    # calculate lowest group that can scatter into current group for each
    max_group_dist = np.array([int(round(np.log(1/a) / du)) for a in alpha ])

    # precompute exponential factors
    efn = np.exp(-1*u[1:]) - np.exp(-1*u[:1])
    efp = np.exp(u[1:]) - np.exp(u[:1])
    
    # sweep groups
    numerator = 0.0
    for i in range(1,num_groups):
        numerator = 0.0
        # sweep nuclides
        for j in range(0,num_nuclides):
            # get lowest group that can scatter into group i
            n = i - max_group_dist[j] if max_group_dist[j] < i else 0
            # sweep over groups that can scatter into group i
            for k in range(n,i):
                numerator = numerator +  efn[i] * efp[k] * p[k] * sig_s[j][k]

        # calculate denominator
        # calculate flux in currrent group
        p[i] = numerator / ( (du) * (sig_t[i]) - (sig_s_reduced[i]) * (du - 1 + np.exp(-du)) )

    return p
