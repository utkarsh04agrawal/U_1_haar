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




def scramble(state,L,t_scr,seed_scram,which_U='haar'):
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
            state = evenlayer(state,unitary_gates[L_max*t: L_max*t + L//2],L)
        else:
            state = oddlayer(state,unitary_gates[L_max*t: L_max*t + L//2],L,BC='PBC')

    return state



def evenlayer(state,U_list,L):
    L_A = len(state.shape) - L
    state = state.reshape((4,)*(L//2) + (2,)*L_A)
    for i in range(0,L-1,2):          
        U = U_list[i//2]
        state = np.tensordot(U,state,axes=(-1,i//2))
        state = np.moveaxis(state,0,i//2)
    state = state.reshape((2,)*(L+L_A))
    return state

def oddlayer(state,U_list,L,BC='PBC'):
    L_A = len(state.shape) - L
    state = np.moveaxis(state,L-1,0)
    state = state.reshape((4,)*(L//2) + (2,)*L_A)
    for i in range(1,L-1,2):          
        U = U_list[(i+1)//2]
        state = np.tensordot(U,state,axes=(-1,(i+1)//2))
        state = np.moveaxis(state,0,(i+1)//2)
    if BC == 'PBC' and L%2 == 0:
        U = U_list[0]
        state = np.tensordot(U,state,axes=(-1,0))
    state = state.reshape((2,)*(L+L_A))
    state = np.moveaxis(state,0,L-1)
    return state


def measurement_layer(state,m_locations,rng_outcome: np.random.default_rng):
    for m in m_locations:
        state = np.swapaxes(state,m,0)
        p_0 = np.sum(np.abs(state[0,:])**2)
        
        if rng_outcome.uniform(0,1) < p_0:
            outcome = 0
        else:
            outcome = 1
        
        if outcome == 0:
            state[1,:]=0
        elif outcome == 1:
            state[0,:] = 0
        S = np.sum(np.abs(state.flatten())**2)
        state = state/S**0.5
        state = np.swapaxes(state,0,m)
    
    return state



def weak_measurement_layer(state,theta,L:int,rng_outcome: np.random.default_rng, m_locations=None):
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
    
    ## Implementing weak measurement
    for m in m_locations:
        state = np.swapaxes(state,m,0)
        p_0 = np.sum(np.abs(state[0,:])**2)
        p_1 = 1-p_0
        p_0a = p_0 + p_1 * np.cos(theta) 
        if rng_outcome.uniform(0,1) < p_0a:
            outcome = 0
        else:
            outcome = 1
        
        if outcome == 0:
            state[1,:] = state[1,:]*np.cos(theta)
        elif outcome == 1:
            state[0,:] = 0
        S = np.sum(np.abs(state.flatten())**2)
        state = state/S**0.5
        state = np.swapaxes(state,0,m)
    return state


def state_entropy(state,interval,renyi_index):
    # start = time.time()
        
    B = list(interval)
    entropy_data = entanglement.renyi_entropy(state,B,renyi_index=renyi_index)
    # print("end_time=", time.time() - start)
    return entropy_data


def state_correlation(state,L):
    
    C = np.zeros((L,L))


    entropy_dic = []
  
    for i in range(L):
        for j in range(i,L,1):
            if i==j:
                C[i,i] = 1
                continue
            x = i
            y = j
            state = np.swapaxes(state,x,0)
            state = np.swapaxes(state,y,1)
            C[i,j] = -np.sum(2*(np.abs(state[0,1,:])**2 + np.abs(state[1,0,:])**2))
            C[j,i] = C[i,j]
            state = np.swapaxes(state,x,0)
            state = np.swapaxes(state,y,1)
    return C
