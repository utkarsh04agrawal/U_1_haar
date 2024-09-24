import sys
sys.path.append('/Users/utkarshagrawal/Documents/Postdoc/U_1_haar')
import numpy as np
from evolution_utils import scramble
from collect_utils import run_weak, save_data
from analysis_utils import save_S_vs_t


# Don't change these
L_max = 60
T_max = 300

PH = True

# Fixed parameters
L_A = 1
L_list = [8,10,12,14,16][:4]
theta_list = np.linspace(0.1,0.8,8)[:2]
initial_state = False
scram_depth = 1
depth = 4
scrambling_type = 'matchgate'
evolution_type = 'matchgate'
root_direc = 'data_weak_measurement/purification/'
seed_scrambling = 1000
shots = 300
BC = 'PBC'
to_save = True
collect = False

if collect:
    for L in L_list:
        initial_state = np.zeros((2,)*(L+L_A))
        index_state_0 = (1,0)*(L//2) + (0,)*L_A 
        index_state_1 = (1,0)*(L//2 - 1) + (0,1) + (1,)*L_A
        initial_state[index_state_0] = 1/2**0.5
        initial_state[index_state_1] = 1/2**0.5 
        T = int(depth*L)
        t_scr = int(scram_depth*L)
        initial_state = scramble(initial_state,L,t_scr,seed_scrambling,which_U=scrambling_type,PH=PH)

        for theta in theta_list:
            theta = round(theta,3)
        
            data = run_weak(theta=theta*np.pi/2,L=L,L_A=L_A,shots=shots,T=T,t_scr=0,
                            scrambling_type=scrambling_type,evolution_type=evolution_type,
                            initial_state=initial_state,BC=BC,PH=PH,
                            seed_scrambling=None,seed_outcome=None,seed_unitary=1
                            )
            
            if to_save:
                save_data(data,L,T,t_scr,
                        scrambling_type,evolution_type,root_direc,theta,BC,shots)


for theta in theta_list[:1]:
    theta = round(theta,3)
    save_S_vs_t(theta,L_list,BC=BC,evolution_type=evolution_type,scrambling_type=scrambling_type,root_direc=root_direc,A=0,depth=depth,scr_depth=scram_depth)

    #half_system_S_vs_t(theta,L_list,BC=BC,evolution_type=evolution_type,scrambling_type=scrambling_type,root_direc=root_direc,depth=depth,scr_depth=scram_depth)