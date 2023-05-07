from qiskit import *
from qiskit import extensions
from qiskit import QuantumCircuit
import qiskit
from qiskit.quantum_info import random_unitary, Statevector, DensityMatrix, entropy, partial_trace
import entanglement
import unitary_sampler
import numpy as np
L_max = 60
T_max = 300

def random_U_1_gate(rng: np.random.default_rng):

    U_11_11 = np.exp(-1j*2*np.pi*rng.uniform())
    U_00_00 = np.exp(-1j*2*np.pi*rng.uniform())
    U_01 = unitary_sampler.haar_qr(2,rng)
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
    outcomes = list(circuit_results.get_counts().items())
    measurement_data = []
    for outcome_str, count in outcomes:
        outcome_str = list(reversed(outcome_str))
        outcome_list = []
        for s in outcome_str[:]:
            if not outcome_list:
                outcome_list.append([])
            if s != ' ':
                outcome_list[-1].append(2*int(s)-1)
            else:
                outcome_list.append([])
#     print(outcome_str,'\n',outcome_list)
#     print(p_locations, results.get_counts(circ))
    
        measurement_array = np.zeros((T,L))

        for t in range(T):
            loc = p_locations[t]
            measurement_array[t,loc] = outcome_list[t]
        
        measurement_data.append((measurement_array,count))
    
    return measurement_data


def scramble(circ,qubits,t_scr,seed_scram,which_U='haar'):
    L = len(qubits)
    assert t_scr < T_max

    # assert which_U in ['haar','matchgate'], print('which_U can only be either \'haar\' or \'matchgate\'')
    # if which_U == 'haar':
    #     U_label = 'haar'
    # if which_U == 'matchgate':
    #     U_label = 'matchgate'

    # file = 'data/scrambled_states/'+U_label+'_seed='+str(seed_scram)
    # if os.path.isfile(file):
    #     with open(file,'rb') as f:
    #         pickle.load() 

    unitary_gates = unitary_sampler.get_U_1_unitary_circuit(seed_unitary=seed_scram,number_of_gates=int(L_max*T_max),which_U=which_U)

    for t in range(t_scr):
        # even layer
        if t%2 == 0:
            circ = evenlayer(circ,unitary_gates[L_max*t: L_max*t + L//2],L,qubits)
        else:
            circ = oddlayer(circ,unitary_gates[L_max*t: L_max*t + L//2],L,qubits,BC='PBC')

    return circ


def evenlayer(circ,U_list,L,q):
    for i in range(0,L-1,2):          
        U = U_list[i//2]
        circ.append(extensions.UnitaryGate(U),[q[i],q[i+1]])
    return circ

def oddlayer(circ,U_list,L,q,BC='PBC'):
    for i in range(1,L-1,2):
        U = U_list[(i+1)//2]
        circ.append(extensions.UnitaryGate(U),[q[i],q[i+1]])
    if BC == 'PBC' and L%2 == 0:
        U = U_list[0]
        circ.append(extensions.UnitaryGate(U),[q[0],q[-1]])
    return circ


def measurement_layer(circ,t,m_locations):
    creg = qiskit.ClassicalRegister(len(m_locations),'c_'+str(t))
    circ.add_register(creg)
    circ.measure(m_locations,creg)
    return circ


def weak_measurement_circuit(circ, q, q_a, theta):
    """

    Args:
        circ (_type_): qiskit circuit
        q (_type_): physical qubit
        q_a (_type_): ancilla qubit
        theta (_type_): measurement strength

    Returns:
        circ: qiskit circuit with measurement circuit added
    """

    circ.rx(theta,q_a)

    ##### Doing exp{i*theta/2 Z_q*X_qa} ########
    circ.h(q_a) # Rotating X_qa to Z_qa

    # apply exp{i*theta/2 Z_q*Z_qa}
    circ.cnot(q,q_a)
    circ.rz(-theta,q_a)
    circ.cnot(q,q_a)

    circ.h(q_a)

    return circ


def weak_measurement_layer(circ,ancilla,p_qbit,theta,L:int,t:int,m_locations=None):
    """To implement exp{-i*theta/2 [1-Z_q]X_qa} = exp{-i*theta X_qa/2} exp{i*theta/2 Z_q*X_qa} on physical qubits. This performs weak measurement.

    Args:
        circ (_type_): quantum circuit
        ancilla (_type_): ancilla qubit to be used for measurement
        p_qbit (_type_): list of physical qubits
        theta (_float): measurement strength. theta = 0: no measurement. theta=pi/2: projective measurement
        L (_int): system size
        t (int): time step
        m_locations (list): measurement_locations. Default: measure all locations

    Returns:
        _type_: _description_
    """
    if m_locations is None:
        m_locations = list(range(L))
    creg = qiskit.ClassicalRegister(len(m_locations),'c_'+str(t))
    circ.add_register(creg)

    ## Implementing weak measurement
    for i in m_locations:
        circ = weak_measurement_circuit(circ,p_qbit[i],ancilla,theta=theta)
        circ.measure(ancilla,creg[i])
    return circ


def job_entropy(job,interval,renyi_index):
    states = job.result().data()
    T = len(states)
    entropy_data = []
    # start = time.time()
    for t in range(T-1):
        
        B = list(interval)
        state = np.asarray(states[str(t)])
        # print(t,state,state.shape)
        entropy_data.append(entanglement.renyi_entropy(state,B,renyi_index=renyi_index))
    # print("end_time=", time.time() - start)
    return entropy_data


def job_correlation(job,L,physical_indices=None,num_ancilla_qubits=2):
    if physical_indices is None:
        physical_indices = list(range(L))

    states = job.result().data()
    T = len(states)
    C = np.zeros((T,L,L))


    entropy_dic = []
    for t in range(T-1):
        state = np.asarray(states[str(t)])
        state = state.reshape((2,)*(L+num_ancilla_qubits))
        for i in range(L):
            for j in range(i,L,1):
                if i==j:
                    C[t,i,i] = 1
                    continue
                x = physical_indices[i]
                y = physical_indices[j]
                state = np.swapaxes(state,x,0)
                state = np.swapaxes(state,y,1)
                C[t,i,j] = -np.sum(2*(np.abs(state[0,1,:])**2 + np.abs(state[1,0,:])**2))
                C[t,j,i] = C[t,i,j]
                state = np.swapaxes(state,x,0)
                state = np.swapaxes(state,y,1)
    return C
