import numpy as np

def configuration_to_index(confi):
    if len(np.shape(confi)) != 1:
        L = len(confi[0])
    else:
        L = len(confi)
    indices = 2**(np.arange(L-1,-1,-1))
    confi = np.array(confi)
    index = np.sum((confi)*indices,axis=len(confi.shape)-1)
    return index


def index_to_confi(index, L, system=None):
    if system is None:
        system = np.arange(0,L,1)
    system = np.array(system)
    len_sys = len(system)
    if type(index)!=int:
        index = np.array(index)
        len_index = len(index)
        temp =  index.reshape((len_index,1)) / (2 ** (L - 1 - system.reshape((1,len_sys))))
        return temp.astype(int) % 2

    else:
        return ((index / (2 ** (L - 1 - system))).astype(int)) % 2


def reduced_density_matrix(vector, sub_system):
    #sub_system is a list of indices belonging to sub-system A
    L = int(np.log2(len(vector)))
    if len(vector.shape) > 1:
        L = len(vector.shape)
        vector = vector.reshape(2**L)
    sub_system = np.array(sub_system)
    A = int(len(sub_system))
    
    # psi matrix is writing psi = psi_matrix_ij |i>_A |j>_B
    psi_matrix = np.zeros((2**A,2**(L-A)),dtype=complex)
    
    system_indices = list(range(L))
    complement = np.array([i for i in system_indices if i not in sub_system])

    temp = np.arange(0, 2**L, 1)
    A_config = index_to_confi(temp, L, sub_system)
    B_config = index_to_confi(temp, L, complement)
    A_index = configuration_to_index(A_config)
    B_index = configuration_to_index(B_config)
    psi_matrix[A_index, B_index] = vector[temp]
    # for i in range(2**L):
    #     A_config = ((i/(2**(L-1-sub_system))).astype(int))%2
    #     B_config = ((i/(2**(L-1-complement))).astype(int))%2
    #     A_index = configuration_to_index(A_config)
    #     B_index = configuration_to_index(B_config)
    #     psi_matrix[A_index,B_index] = vector[i]
    
    u,schmidt_values,v = np.linalg.svd(psi_matrix,compute_uv=True,full_matrices=False)
    return u, (schmidt_values), v


def renyi_entropy(vector, sub_system, renyi_index):
    _,schmidt_values,_ = reduced_density_matrix(vector, sub_system)
    if np.round(np.sum(schmidt_values**2),6)!=1:
        print('ah, Schimdt values not normalized',sub_system, np.sum(schmidt_values**2))
    if renyi_index == 1:
        schmidt_values[schmidt_values==0]=1
        entropy = np.sum(-schmidt_values**2*np.log2(schmidt_values**2))
    else:
        entropy = np.log2(np.sum(schmidt_values**(2*renyi_index)))/(1-renyi_index)
    
    return entropy
