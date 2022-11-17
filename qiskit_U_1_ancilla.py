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

from qiskit_U_1 import random_U_1_gate,outcome_history, evenlayer, oddlayer, measurement_layer
from qiskit import *
from qiskit import extensions
from qiskit import QuantumCircuit
import qiskit


def scrambled_initial_state_purification(L, Q, BC='PBC'):
    assert Q<=L//2
    T_scram = 4*L
    filedir = 'data/scrambled_states_purification/'
    if not os.path.isdir(filedir):
        os.makedirs(filedir)
    filename = filedir + 'L='+str(L)+'_T='+str(T_scram)+'_Q='+str(Q)
    if os.path.isfile(filename):
        with open(filename,'rb') as f:
            state = pickle.load(f)
        return state
    q = qiskit.QuantumRegister(L+1, 'q')
    c = qiskit.ClassicalRegister(1,'c')
    circ = QuantumCircuit(q,c)
    for i in range(1,Q+1,1):
        circ.x(q[2*i-1])
    circ.h(0)
    circ.cnot(0,1)
    circ.cnot(0,2) # delete this line to study sharpening
    
    rng = np.random.default_rng(seed=99999)


    for _ in range(T_scram):
        for i in range(1,L,2):
            U = extensions.UnitaryGate(random_U_1_gate(rng),label=r'$U$')
            circ.append(U,[q[i],q[i+1]])
        for i in range(2,L,2):
            U = extensions.UnitaryGate(random_U_1_gate(rng),label=r'$U$')
            circ.append(U,[q[i],q[i+1]])
        U = extensions.UnitaryGate(random_U_1_gate(rng),label=r'$U$')
        if BC=='PBC' and L%2==0:
            circ.append(U,[q[1],q[-1]])
            
            
    backend = Aer.get_backend('statevector_simulator')
    job = qiskit.execute(circ,backend=backend)
    state = np.asarray(job.result().data()['statevector'])
    with open(filename,'wb') as f:
        pickle.dump(state,f)
    return state


def scrambled_initial_state_sharpening(L, Q, BC='PBC'):
    assert Q<=L//2
    T_scram = 4*L
    filedir = 'data/scrambled_states_sharpening/'
    if not os.path.isdir(filedir):
        os.makedirs(filedir)
    filename = filedir + 'L='+str(L)+'_T='+str(T_scram)+'_Q='+str(Q)
    if os.path.isfile(filename):
        with open(filename,'rb') as f:
            state = pickle.load(f)
        return state
    q = qiskit.QuantumRegister(L+1, 'q')
    c = qiskit.ClassicalRegister(1,'c')
    circ = QuantumCircuit(q,c)
    for i in range(1,Q+1,1):
        circ.x(q[2*i-1])
    circ.h(0)
    circ.cnot(0,1)
    # circ.cnot(0,2) # delete this line to study sharpening
    
    rng = np.random.default_rng(seed=99999)

    for _ in range(T_scram):
        for i in range(1,L,2):
            U = extensions.UnitaryGate(random_U_1_gate(rng),label=r'$U$')
            circ.append(U,[q[i],q[i+1]])
        for i in range(2,L,2):
            U = extensions.UnitaryGate(random_U_1_gate(rng),label=r'$U$')
            circ.append(U,[q[i],q[i+1]])
        U = extensions.UnitaryGate(random_U_1_gate(rng),label=r'$U$')
        if BC=='PBC' and L%2==0:
            circ.append(U,[q[1],q[-1]])
            
            
    backend = Aer.get_backend('statevector_simulator')
    job = qiskit.execute(circ,backend=backend)
    state = np.asarray(job.result().data()['statevector'])
    with open(filename,'wb') as f:
        pickle.dump(state,f)
    return state


## main method for making and running the purification circuit
def get_circuit(L,T,Q,p,rng,BC='PBC',sharpening=False):
    assert Q<=L//2
    q = qiskit.QuantumRegister(L+1, 'q') # qubit #0 is ancilla
    c = qiskit.ClassicalRegister(1,'c')
    circ = QuantumCircuit(q,c)

    if sharpening:
        scrambled_state = scrambled_initial_state_sharpening(L=L,Q=Q,BC=BC)
    else:
        scrambled_state = scrambled_initial_state_purification(L=L,Q=Q,BC=BC)
    circ.initialize(scrambled_state)
    # backend = Aer.get_backend('statevector_simulator')
    # circ = qiskit.transpile(circ,backend=backend)

    p_locations = []
    total_N_m = 0

    for t in range(T):
        # even layer
        # start = time.time()
        circ = evenlayer(circ,L,q[1:],rng)  # q[1:] are system qubits, q[0] is the ancilla
            
        measured_locations = np.where(np.random.uniform(0,1,L)<p)[0]+1 # +1 because we don't want to measure ancilla
        p_locations.append(list(measured_locations))
        N_m = len(measured_locations)
        total_N_m += N_m

        circ = measurement_layer(circ,measured_locations)
        circ.save_statevector(str(2*t))
        # print(t,time.time()-start)

        # odd layer
        # start = time.time()
        circ = oddlayer(circ,L,q,rng,BC)

        measured_locations = np.where(np.random.uniform(0,1,L)<p)[0] + 1
        p_locations.append(list(measured_locations))
        N_m = len(measured_locations)
        total_N_m += N_m

        circ = measurement_layer(circ,measured_locations)
        circ.save_statevector(str(2*t+1))
        # print(t+0.5,time.time()-start)

    return circ, p_locations, total_N_m


def run_circuit(L,T,Q,p,seed,BC='PBC'):
    rng = np.random.default_rng(seed=seed) #random number generator with seed=seed

    backend = Aer.get_backend('aer_simulator')

    circ, p_locations, total_N_m  = get_circuit(L,T,Q=Q,p=p,rng=rng,BC=BC) # total_N_m is total # of measurements performed
    circ.save_statevector('f')
    circ = qiskit.transpile(circ,backend=backend)
    job = qiskit.execute(circ,backend=backend)
    results = job.result()

    if total_N_m == 0:
        measurement_array = np.zeros((2*T,L+1))
    else:
        measurement_array = outcome_history(results, L+1, T, p_locations)
    
    return job, measurement_array, p_locations

