import numpy as np
import os, shutil
import matplotlib.pyplot as pl
import pickle
import time

from evolution_utils import get_weak_data
from analysis_utils import get_filename, half_system_S_vs_t

# Don't change these
L_max = 60
T_max = 300


# Fixed parameters
L_A = 0
L_list = [8,10,12,14]
theta_list = np.linspace(0,0.8,9)
initial_state = False
scram_depth = 0
depth = 4
scrambling_type = 'matchgate'
evolution_type = 'matchgate'
root_direc = 'data_weak_measurement/without_ancilla/'
seed_unitary = 1
seed_scrambling = 1000*seed_unitary
shots = 5
BC = 'PBC'


def run():
    for theta in theta_list:
        theta = np.round(theta,3)
        for L in L_list[:]:
            start = time.time()
            entropy_data = []
            correlation_data = []
            for _ in range(shots):
                
                #define parameters
                if initial_state is False:
                    state = np.zeros((2,)*(L+L_A))
                    index_state_0 = (1,0)*(L//2) + (0,)*L_A 
                    index_state_1 = (1,0)*(L//2 - 1) + (0,1) + (1,)*L_A
                    state[index_state_0] = 1/2**0.5
                    state[index_state_1] = 1/2**0.5 
                t_scr = int(scram_depth*L)
                T = int(4*L)

                # get data
                seed_outcome = np.random.randint(0,1000000000,1)
                state,entropy,correlation,_ = get_weak_data(state,L,L_A,T,t_scr,theta*np.pi/2,scrambling_type=scrambling_type,evolution_type=evolution_type,seed_unitary=seed_unitary,seed_scram=seed_scrambling,seed_outcomes=seed_outcome,BC=BC)
                entropy_data.append(entropy)
                correlation_data.append(correlation)
                
            data = {'entropy':entropy_data,
                    'correlation':correlation_data,
                    'seed_outcome': seed_outcome,
                    'seed_unitary':seed_unitary,
                    'seed_scrambling':seed_scrambling}
                            
            #save data
            file_dir = get_filename(t_scr=t_scr,scram_U_type=scrambling_type,evolution_U_type=evolution_type,root_direc=root_direc)
            file_dir = file_dir + '/L='+str(L)
            file_dir = file_dir+'/T='+str(T)+'_tscr='+str(t_scr)+'_theta='+str(theta)+'_BC='+BC
            # if os.path.isdir(file_dir):
            #     shutil.rmtree(file_dir)
            if not os.path.isdir(file_dir):
                os.makedirs(file_dir)
            filename = file_dir+'/shots='+str(shots)+'_'+str(np.random.randint(0,1000000000,1))
            
            with open(filename,'wb') as f:
                pickle.dump(data,f)
            
            print("L={}, theta={}, time={}".format(L,theta,time.time()-start))


# run()


for theta in theta_list:
    theta = round(theta,3)
    #save_S_vs_t(theta,L_list,BC=BC,evolution_type=evolution_type,scrambling_type=scrambling_type,root_direc=root_direc,A=0,depth=depth,scr_depth=scram_depth)

    half_system_S_vs_t(theta,L_list,BC=BC,evolution_type=evolution_type,scrambling_type=scrambling_type,root_direc=root_direc,depth=depth,scr_depth=scram_depth)