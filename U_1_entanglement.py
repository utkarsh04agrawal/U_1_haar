import time
import numpy as np
from itertools import permutations


def state_indices_for_a_given_particle_number(A, n, L): #A is subsystem, n is number of particles in A
    indices_list = np.arange(0,2**L,1).reshape((2,)*L)
    L_A = len(A)
    charge_config = [0]*L_A
    charge_config[0:n] = [1]*n
    charge_config = (set(permutations(charge_config)))
    A = list(sorted(A))
    indices_list = np.moveaxis(indices_list, A, range(0,L_A,1))
    n_charge_indices = [indices_list[i] for i in charge_config]
    return np.array(n_charge_indices)


def number_entropy(state: np.ndarray, A, n=1):
    L_A = len(A)
    L = int(np.log2(len(state)))
    different_charges_in_A = np.arange(0,L_A+1,1)
    charge_dist = []
    for charge in different_charges_in_A:
        indices = state_indices_for_a_given_particle_number(A, charge, L).flatten()
        p_charge = np.sum(np.abs(state[indices])**2)
        charge_dist.append(p_charge)

    charge_dist = np.array(charge_dist)
    if n == 1:
        charge_dist[charge_dist == 0] = 1
        return np.sum(-charge_dist*np.log2(charge_dist))
    else:
        return -np.log2(np.sum(charge_dist**n))/(n-1)