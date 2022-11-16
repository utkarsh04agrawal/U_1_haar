import numpy as np
import time
import pickle
import os
import sys
import matplotlib.pyplot as pl
import importlib
import haar_sampler
import entanglement
import U_1_entanglement

from qiskit import *
from qiskit import extensions
from qiskit import QuantumCircuit
import qiskit
from qiskit.quantum_info import random_unitary, Statevector, DensityMatrix, entropy, partial_trace



def random_U_1_gate(rng: np.random.default_rng):

    U_11_11 = np.exp(-1j*2*np.pi*rng.uniform())
    U_00_00 = np.exp(-1j*2*np.pi*rng.uniform())
    U_01 = haar_sampler.haar_qr(2,rng)
    U = np.zeros((4,4),dtype=complex)
    U[0,0] = U_11_11
    U[3,3] = U_00_00
    U[1:3,1:3] = U_01
    return U


def toy_example():
    q = qiskit.QuantumRegister(2, 'q')
    c = qiskit.ClassicalRegister(1,'c')
    circ = QuantumCircuit(q,c)
    backend = Aer.get_backend('statevector_simulator')
    job = qiskit.execute(circ,backend=backend)
    for i in range(0,2,2):
        circ.x(q[i])
        U = extensions.UnitaryGate(random_U_1_gate(),label=r'$U$')
        circ.append(U,[q[i],q[i+1]])
        print(U)
        circ.save_statevector(str(1))
        U = extensions.UnitaryGate(random_U_1_gate(),label=r'$U$')
        circ.append(U,[q[i],q[i+1]])
        circ.save_statevector(str(2))
        
    circ.draw('mpl')

    # job = qiskit.execute(circ,backend=backend)
    # results = job.result()
    # results.data()


## this function take measurement locations and outcomes to return an array of size (L,2*T) with outcomes marked as +1,-1 if measured, otherwise 0
def outcome_history(circuit_results,L,T,p_locations):
    outcome_str = list(circuit_results.get_counts().keys())[0]
    outcome_str = list(reversed(outcome_str))
    outcome_list = []
    for s in outcome_str[2:]:
        if not outcome_list:
            outcome_list.append([])
        if s != ' ':
            outcome_list[-1].append(2*int(s)-1)
        else:
            outcome_list.append([])
#     print(outcome_str,'\n',outcome_list)
#     print(p_locations, results.get_counts(circ))
    
    measurement_array = np.zeros((2*T,L))

    for t in range(2*T):
        loc = p_locations[t]
        measurement_array[t,loc] = outcome_list[t]
    
    return measurement_array



def evenlayer(circ,L,q,rng):
    for i in range(0,L-1,2):          
        U = extensions.UnitaryGate(random_U_1_gate(rng))
        circ.append(U,[q[i],q[i+1]])
    return circ

def oddlayer(circ,L,q,rng,BC='PBC'):
    for i in range(1,L-1,2):
        U = extensions.UnitaryGate(random_U_1_gate(rng))
        circ.append(U,[q[i],q[i+1]])
        if BC == 'PBC' and L%2 == 0:
            U = extensions.UnitaryGate(random_U_1_gate(rng))
            circ.append(U,[q[0],q[-1]])
    return circ


def measurement_layer(circ,m_locations):
    creg = qiskit.ClassicalRegister(len(m_locations))
    circ.add_register(creg)
    circ.measure(m_locations,creg)
    return circ

## main method for making and running the circuit
def get_circuit(L,T,p,rng,BC='PBC'):
    q = qiskit.QuantumRegister(L, 'q')
    c = qiskit.ClassicalRegister(1,'c')
    circ = QuantumCircuit(q,c)

    for i in range(0,L,2):
        circ.x([q[i]])

    p_locations = []
    total_N_m = 0

    for t in range(T):
        # even layer
        start = time.time()
        circ = evenlayer(circ,L,q,rng)
            
        measured_locations = list(np.where(np.random.uniform(0,1,L)<p)[0])
        p_locations.append(list(measured_locations))
        N_m = len(measured_locations)
        total_N_m += N_m

        circ = measurement_layer(circ,measured_locations)
        circ.save_statevector(str(2*t))
        print(t,time.time()-start)

        # odd layer
        start = time.time()
        circ = oddlayer(circ,L,q,rng,BC)

        measured_locations = list(np.where(np.random.uniform(0,1,L)<p)[0])
        p_locations.append(list(measured_locations))
        N_m = len(measured_locations)
        total_N_m += N_m

        circ = measurement_layer(circ,measured_locations)
        circ.save_statevector(str(2*t+1))
        print(t+0.5,time.time()-start)

    return circ, p_locations, total_N_m


def run_circuit(L,T,p,seed,BC='PBC'):
    rng = np.random.default_rng(seed=seed) #random number generator with seed=seed

    backend = Aer.get_backend('statevector_simulator')

    circ, p_locations, total_N_m  = get_circuit(L,T,p,rng,BC) # total_N_m is total # of measurements performed

    circ = qiskit.transpile(circ,backend=backend)

    job = qiskit.execute(circ,backend=backend)
    results = job.result()
    if total_N_m == 0:
        measurement_array = np.zeros((L,T))
    else:
        measurement_array = outcome_history(results, L, T, p_locations)
    
    return job, measurement_array, p_locations


a,b,c = run_circuit(10,10,0.1,1)


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


# method to collect data # INCOMPLETE
def collect_data(L,T,p_list):
    # p_list = [0.05]
    seed_list = np.arange(1,201,1)
    for p in p_list:
        start = time.time()
        filename_ent = 'U_1/data/entropy_L='+str(L)+'_T='+str(T)+'_p='+str(p)
        filename_num_ent = 'U_1/data/number_entropy_L='+str(L)+'_T='+str(T)+'_p='+str(p)
        filename_outcome = 'U_1/data/outcomes_L='+str(L)+'_T='+str(T)+'_p='+str(p)
        filename_state = 'U_1/data/state_L='+str(L)+'_T='+str(T)+'_p='+str(p)
        
        ent_dic = {}
        out_dic = {}
        state_dic = {}
        num_ent_dic = {}
#         if os.path.isfile(filename_ent):
#             with open(filename_ent,'rb') as f:
#                 ent_dic = pickle.load(f)
#             with open(filename_num_ent,'rb') as f:
#                 num_ent_dic = pickle.load(f)
#             with open(filename_outcome,'rb') as f:
#                 out_dic = pickle.load(f)
#             with open(filename_state,'rb') as f:
#                 state_dic = pickle.load(f)
        for seed in seed_list:
            if seed in ent_dic:
                continue
            ent_list, num_ent_list, outcomes, states = get_ent_list(L,T,p,seed)
            ent_dic[seed] = ent_list
            num_ent_dic[seed] = num_ent_list
            out_dic[seed] = outcomes
            state_dic[seed] = states
        with open(filename_ent,'wb') as f:
            pickle.dump(ent_dic,f)
        with open(filename_num_ent,'wb') as f:
            pickle.dump(num_ent_dic,f)
        with open(filename_outcome, 'wb') as f:
            pickle.dump(out_dic,f)
        with open(filename_state, 'wb') as f:
            pickle.dump(state_dic,f)
        print('Done! L=',L,' p=',p,'time:',time.time()-start)
