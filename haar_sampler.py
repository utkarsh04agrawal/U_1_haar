import numpy as np
import pickle
import os


def haar_qr(N, rng: np.random.default_rng):
    """" Generate Haar unitary of size NxN using QR decomposition """
    A, B = rng.normal(0,1,size=(N,N)), rng.normal(0,1,size=(N,N))
    Z = A + 1j * B
    Q, R = np.linalg.qr(Z)
    Lambda = np.diag(R.diagonal()/np.abs(R.diagonal()))
    return np.dot(Q,Lambda)


def get_unitary_circuit(seed_unitary, N, number_of_gates):
    filename = 'unitary_gates_data/N='+str(N)+'_seed='+str(seed_unitary)
    if os.path.isfile(filename):
        with open(filename,'rb') as f:
            unitary_gates, unitary_rng = pickle.load(f)

    else:
        unitary_gates = []
        unitary_rng = np.random.default_rng(seed=seed_unitary)

    number_of_gates_available = len(unitary_gates)
    while number_of_gates_available < number_of_gates:
        unitary_gates.append(haar_qr(4, unitary_rng))
        number_of_gates_available += 1
    with open(filename, 'wb') as f:
        pickle.dump([unitary_gates, unitary_rng],f)
    return unitary_gates



def U_1_sym_gate_sampler(rng: np.random.default_rng):
    U = np.zeros((4,4), dtype=complex)
    phase1 = rng.uniform(0,1)
    phase2 = rng.uniform(0, 1)
    U[0, 0] = np.exp(-1j * 2 * np.pi * phase1)
    U[3, 3] = np.exp(-1j * 2 * np.pi * phase2)
    U_10 = haar_qr(2,rng)
    U[np.array([1,2]).reshape((2,-1)),np.array([1,2])] = U_10
    return U


def get_U_1_unitary_circuit(seed_unitary, number_of_gates):
    print()
    filename = 'unitary_gates_data/U(1)/N=' + str(4) + '_seed=' + str(seed_unitary)
    if os.path.isfile(filename):
        with open(filename, 'rb') as f:
            unitary_gates, unitary_rng = pickle.load(f)

    else:
        unitary_gates = []
        unitary_rng = np.random.default_rng(seed=seed_unitary)

    number_of_gates_available = len(unitary_gates)
    while number_of_gates_available < number_of_gates:
        unitary_gates.append(U_1_sym_gate_sampler(unitary_rng))
        number_of_gates_available += 1
    with open(filename, 'wb') as f:
        pickle.dump([unitary_gates, unitary_rng], f)
    return unitary_gates