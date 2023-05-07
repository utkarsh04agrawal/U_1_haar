import numpy as np
import time
import pickle
import os
import sys
import matplotlib.pyplot as pl
import importlib
import unitary_sampler
import entanglement
import U_1_entanglement

from qiskit import *
from qiskit import extensions
from qiskit import QuantumCircuit
import qiskit
from qiskit.quantum_info import random_unitary, Statevector, DensityMatrix, entropy, partial_trace


from qiskit_evolution_utils import evenlayer, oddlayer, measurement_layer, outcome_history, scramble



## main method for making and running the circuit
def get_circuit(L,T,p,BC='PBC',seed=1,seed_loc = 1):
    q = qiskit.QuantumRegister(L, 'q')
    circ = QuantumCircuit(q)

    L_max = 60
    T_max = 300
    assert T<T_max
    unitary_gates = unitary_sampler.get_U_1_unitary_circuit_haar(seed_unitary=seed,number_of_gates=int(L_max*T_max)) # 
    for i in range(0,L,2):
        circ.x([q[i]])

    p_locations = []
    total_N_m = 0

    rng_loc = np.random.default_rng(seed=seed_loc)

    for t in range(T):
        # even layer
        start = time.time()
        if t%2 == 0:
            circ = evenlayer(circ,unitary_gates[L_max*t: L_max*t + L//2],L,q)
        else:
            circ = oddlayer(circ,unitary_gates[L_max*t: L_max*t + L//2],L,q,BC=BC)
            
        measured_locations = list(np.where(rng_loc.uniform(0,1,L)<p)[0])
        p_locations.append(list(measured_locations))
        N_m = len(measured_locations)
        total_N_m += N_m

        circ = measurement_layer(circ,t,measured_locations)
        circ.save_statevector(str(t))
        print(t,time.time()-start)

    return circ, p_locations, total_N_m


def run_circuit(L,T,p,seed_U,seed_loc,BC='PBC'):

    backend = Aer.get_backend('statevector_simulator')

    circ, p_locations, total_N_m  = get_circuit(L,T,p,BC=BC,seed=seed_U,seed_loc=seed_loc) # total_N_m is total # of measurements performed

    circ = qiskit.transpile(circ,backend=backend)

    job = qiskit.execute(circ,backend=backend)
    results = job.result()
    if total_N_m == 0:
        measurement_array = np.zeros((L,T))
    else:
        measurement_array = outcome_history(results, L, T, p_locations)
    
    return job, measurement_array, p_locations



def get_entropy(job,L):
    state_list = []

    entropy_dic = {}
    num_entropy_dic = {}

    states = job.result().data()

    T = len(state)-1 # -1 because circ stores the final time state automatically, so it will be double counted

    for t in range(T):
        for l in np.arange(1,int(L/2)+1,1): # looping over different sub-system sizes
            B = list(np.arange(0,l,1))
            state = np.asarray(states[str(t)])
            # state_list.append(state)
            if l not in entropy_dic:
                entropy_dic[l] = []
                num_entropy_dic[l] = []

            entropy_dic[l].append(entanglement.renyi_entropy(state,B,2))
            num_entropy_dic[l].append(U_1_entanglement.number_entropy(state,B,2))

            ## Using Qiskit's native function for entropy but they are slower than the custom method I have
#             rho_a = partial_trace(state,list(B))
#             entropy_list[l].append(entropy(rho_a))

    return entropy_dic, num_entropy_dic


