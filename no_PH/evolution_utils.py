import time
from Circuits.circuit_evolution import evenlayer, oddlayer, weak_measurement_layer, measurement_layer
import Circuits.entanglement as entanglement
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


def get_weak_data(state,L,L_A,T,t_scr,theta,evolution_type,scrambling_type,seed_scram,seed_unitary,seed_outcomes,BC='PBC'):
    assert scrambling_type in ['haar','matchgate'], print('scrambling_type can only be either \'haar\' or \'matchgate\'')
    assert evolution_type in ['haar','matchgate'], print('evolution_type can only be either \'haar\' or \'matchgate\'')

    assert T<T_max

    state = scramble(state,L,t_scr,seed_scram=seed_scram,which_U=scrambling_type)

    unitary_gates = unitary_sampler.get_U_1_unitary_circuit(seed_unitary=seed_unitary,number_of_gates=int(L_max*T_max),which_U=evolution_type)
    cache = {}

    outcome_rng = np.random.default_rng(seed=seed_outcomes)
    total_N_m = 0

    entropy_data = []
    correlation_data = []
    for t in range(T):
        if t%2 == 0:
            state = evenlayer(state,unitary_gates[L_max*t: L_max*t + L//2],L=L)
        else:
            state = oddlayer(state,unitary_gates[L_max*t: L_max*t + L//2],L=L,BC=BC)
        
        m_locations = list(range(L)) # locations of weak measurement
        total_N_m += L
        state = weak_measurement_layer(state,theta,L,rng_outcome=outcome_rng,m_locations=m_locations)

        entropy = get_entropy(L,L_A,state)
        entropy_data.append(entropy)
        correlation_data.append(state_correlation(state,L))
    
    cache['total_N_m'] = total_N_m
    cache['m_locations'] = m_locations
    
    return state, np.array(entropy_data), correlation_data, cache


def get_proj_data(state,L,L_A,T,t_scr,p,evolution_type,scrambling_type,seed_scram,seed_unitary,seed_outcomes,BC='PBC'):
    assert scrambling_type in ['haar','matchgate'], print('scrambling_type can only be either \'haar\' or \'matchgate\'')
    assert evolution_type in ['haar','matchgate'], print('evolution_type can only be either \'haar\' or \'matchgate\'')

    assert T<T_max

    state = scramble(state,L,t_scr,seed_scram=seed_scram,which_U=scrambling_type)

    unitary_gates = unitary_sampler.get_U_1_unitary_circuit(seed_unitary=seed_unitary,number_of_gates=int(L_max*T_max),which_U=evolution_type)
    
    cache = {}

    outcome_rng = np.random.default_rng(seed=seed_outcomes)
    total_N_m = 0

    entropy_data = []
    correlation_data = []
    for t in range(T):
        if t%2 == 0:
            state = evenlayer(state,unitary_gates[L_max*t: L_max*t + L//2],L=L)
        else:
            state = oddlayer(state,unitary_gates[L_max*t: L_max*t + L//2],L=L,BC=BC)
        
        m_locations = np.where(outcome_rng.uniform(0,1,L)<p)[0] # locations of measurement
        total_N_m += L
        state = measurement_layer(state,rng_outcome=outcome_rng,m_locations=m_locations)

        entropy = get_entropy(L,L_A,state)
        entropy_data.append(np.array(entropy))
        correlation_data.append(state_correlation(state,L))
    
    cache['total_N_m'] = total_N_m
    cache['m_locations'] = m_locations
    
    return state, np.array(entropy_data), correlation_data, cache


def state_entropy(state,interval,renyi_index):
    # start = time.time()
        
    B = list(interval)
    entropy_data = entanglement.renyi_entropy(state,B,renyi_index=renyi_index)
    # print("end_time=", time.time() - start)
    return entropy_data


def get_entropy(L,L_A,state):
    entropy_list = []
    for A in range(0,L//2+1,1):
        interval = list(range(L-A,L+L_A)) # Interval to calculate entropy of
        # A = 0 correponds to entropy of ancilla 
        entropy_list.append(state_entropy(state,interval=interval,renyi_index=1))
    return entropy_list

def state_correlation(state,L):
    
    C = np.zeros((L,L))
  
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
