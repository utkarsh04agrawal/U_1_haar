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

import qiskit
from qiskit import QuantumCircuit, Aer
from qiskit.quantum_info import random_unitary, Statevector, DensityMatrix, entropy, partial_trace

from qiskit_evolution_utils import weak_measurement_layer, scramble
from qiskit_evolution_utils import evenlayer, oddlayer, outcome_history


def get_weak_circuit_ancilla(L,T,theta,t_scr,scram_U_type='haar',evolution_U_type='haar',BC='PBC',seed_unitary=1,):
    q = qiskit.QuantumRegister(L, 'q')
    ancilla = qiskit.QuantumRegister(1,'a')
    measurement_ancilla = qiskit.QuantumRegister(1,'ma')
    circ = qiskit.QuantumCircuit(measurement_ancilla,q,ancilla)

    L_max = 60
    T_max = 300
    assert T<T_max
    
    for i in range(0,L,2):
        circ.x([q[i]])

    circ.h(ancilla)
    circ.cnot(ancilla[0],q[0])
    circ.cnot(ancilla[0],q[1])

    circ = scramble(circ,qubits=q,t_scr=t_scr,seed_scram=1000*seed_unitary,which_U=scram_U_type)

    unitary_gates = unitary_sampler.get_U_1_unitary_circuit(seed_unitary=seed_unitary,number_of_gates=int(L_max*T_max),which_U=evolution_U_type) # 
    p_locations = []
    total_N_m = 0

    circ.save_statevector(str(0))

    for t in range(T):
        # even layer
        start = time.time()
        if t%2 == 0:
            circ = evenlayer(circ,unitary_gates[L_max*t: L_max*t + L//2],L,q)
        else:
            circ = oddlayer(circ,unitary_gates[L_max*t: L_max*t + L//2],L,q,BC=BC)
            
        p_locations.append(list(range(L)))
        N_m = len(p_locations[-1])
        total_N_m += N_m
        circ = weak_measurement_layer(circ,ancilla=measurement_ancilla,p_qbit=q,L=L,t=t,theta=theta,m_locations=p_locations[-1])

        circ.save_statevector(str(t+1))
        # print(t,time.time()-start)

    return circ, p_locations, total_N_m



def simulate_weak_circuit_ancilla(circ,m_locs,L,T,shots=1):
    backend = Aer.get_backend('aer_simulator')

    
    job = qiskit.execute(circ,backend=backend,shots=shots)
    results = job.result()
    # if total_N_m == 0:
    #     measurement_array = np.zeros((L,T))
    # else:
    measurement_array = outcome_history(results, L, T, m_locs)
    
    return job, measurement_array, m_locs




# def get_entropy(L,T,results=None):
#     state_list = []

#     entropy_dic = {}
#     num_entropy_dic = {}

#     if results is None:


#     states = job.result().data()

#     T = len(state)-1 # -1 because circ stores the final time state automatically, so it will be double counted

#     for t in range(T):
#         for l in np.arange(1,int(L/2)+1,1): # looping over different sub-system sizes
#             B = list(np.arange(0,l,1))
#             state = np.asarray(states[str(t)])
#             # state_list.append(state)
#             if l not in entropy_dic:
#                 entropy_dic[l] = []
#                 num_entropy_dic[l] = []

#             entropy_dic[l].append(entanglement.renyi_entropy(state,B,2))
#             num_entropy_dic[l].append(U_1_entanglement.number_entropy(state,B,2))

#             ## Using Qiskit's native function for entropy but they are slower than the custom method I have
# #             rho_a = partial_trace(state,list(B))
# #             entropy_list[l].append(entropy(rho_a))

#     return entropy_dic, num_entropy_dic