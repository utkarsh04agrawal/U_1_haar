from evolution_utils import state_entropy
import os
import pickle
import numpy as np
import matplotlib.pyplot as pl

def get_filename(t_scr,evolution_U_type,scram_U_type,root_direc):
    file_dir = root_direc

    if t_scr==0:
        file_dir = file_dir + 'without_scrambling' 
    else:
        if scram_U_type == 'haar':
            file_dir = file_dir + 'haar_scrambling' 
        elif scram_U_type == 'matchgate':
            file_dir = file_dir + 'matchgate_scrambling' 
    if evolution_U_type == 'haar':
        file_dir = file_dir + '/haar_evolution'
    if evolution_U_type == 'matchgate':
        file_dir = file_dir + '/matchgate_evolution'
    
    if not os.path.isdir(file_dir):
        os.makedirs(file_dir)

    return file_dir

def get_entropy(L,L_A,state):
    entropy_list = []
    for A in range(0,L//2+1,1):
        interval = list(range(L-A,L+L_A)) # Interval to calculate entropy of
        # A = 0 correponds to entropy of ancilla 
        entropy_list.append(state_entropy(state,interval=interval,renyi_index=1))
    return entropy_list
    
    
def read_entropy(L,T,t_scr,theta,BC,scrambling_type,evolution_type,root_direc):
    file_dir = get_filename(t_scr=t_scr,scram_U_type=scrambling_type,evolution_U_type=evolution_type,root_direc=root_direc)
    file_dir = file_dir + '/L='+str(L)
    file_dir = file_dir+'/T='+str(T)+'_tscr='+str(t_scr)+'_theta='+str(theta)+'_BC='+BC
    entropy_list = []
    for filename in os.listdir(file_dir):
        with open(file_dir + '/' + filename,'rb') as f:
            data = pickle.load(f)
        entropy_list.extend(data['entropy'])
    entropy = np.array(entropy_list)
    return entropy


def get_slope(xdata,ydata):
    zz = np.polyfit(xdata,ydata,1,full=True)
    p = np.poly1d(zz[0])
    return zz[0], p

def save_S_vs_t(theta,L_list,BC,evolution_type,scrambling_type,root_direc,A=0,depth=4,scr_depth=1):
    pl.figure(1)
    slope = []
    for L in L_list:
        T = int(depth*L)
        t_scr = int(scr_depth*L)
        entropy = read_entropy(L,T,t_scr,theta,BC=BC,scrambling_type=scrambling_type,evolution_type=evolution_type,root_direc=root_direc)
        ydata = np.average(entropy[:,:,A],axis=0)
        err = np.std(entropy[:,:,A],axis=0)/(entropy.shape[0]-1)**0.5
        xdata = np.arange(0,T,1)
        zz,p = get_slope(xdata[2*L:3*L],np.log(ydata[2*L:3*L]))
        slope.append(-zz[0])
        # print(entropy[-1])
        pl.errorbar(xdata[:]/L,ydata[:],yerr=err[:],ls='-',marker='o',label='L={}'.format(L))
        pl.plot(xdata[:]/L,np.exp(p(np.array(xdata))),'k',ls=':')

    pl.yscale('log')
    pl.ylabel(r'$S_A(t)$',fontsize=16)
    pl.xlabel(r'$t/L$',fontsize=16)
    pl.title(r'$p={}$'.format(theta),fontsize=16)
    pl.legend()
    save_file = get_filename(t_scr,evolution_type,scrambling_type,root_direc) + '/figures/S_vs_t_A='+str(A)+'BC='+str(BC)+'/'
    if not os.path.isdir(save_file):
        os.makedirs(save_file)
    pl.savefig(save_file+'theta={}.pdf'.format(theta))
    pl.close(1)

    pl.figure(2)
    pl.plot(L_list[:],[1/i for i in slope],'-o')
    pl.ylabel(r'$\tau$',fontsize=16)
    pl.xlabel(r'$L$',fontsize=16)
    pl.yscale('log')
    pl.savefig(save_file+'decay_rate_theta={}.pdf'.format(theta))
    pl.close(2)


def half_system_S_vs_t(theta,L_list,BC,evolution_type,scrambling_type,root_direc,depth=4,scr_depth=1):
    pl.figure(1)
    slope = []
    for L in L_list:
        T = int(depth*L)
        t_scr = int(scr_depth*L)
        entropy = read_entropy(L,T,t_scr,theta,BC=BC,scrambling_type=scrambling_type,evolution_type=evolution_type,root_direc=root_direc)
        ydata = np.average(entropy[:,:,L//2],axis=0)
        err = np.std(entropy[:,:,L//2],axis=0)/(entropy.shape[0]-1)**0.5
        xdata = np.arange(0,T,1)
        zz,p = get_slope(xdata[2*L:3*L],np.log(ydata[2*L:3*L]))
        slope.append(zz[0])
        # print(entropy[-1])
        pl.errorbar(xdata[:]/L,ydata[:],yerr=err[:],ls='-',marker='o',label='L={}'.format(L))
        pl.plot(xdata[:]/L,np.exp(p(np.array(xdata))),'k',ls=':')

    pl.yscale('log')
    pl.ylabel(r'$S_A(t)$',fontsize=16)
    pl.xlabel(r'$t/L$',fontsize=16)
    pl.title(r'$p={}$'.format(theta),fontsize=16)
    pl.legend()
    save_file = get_filename(t_scr,evolution_type,scrambling_type,root_direc) + '/figures/half_system_S_vs_t_BC='+str(BC)+'/'
    if not os.path.isdir(save_file):
        os.makedirs(save_file)
    pl.savefig(save_file+'theta={}.pdf'.format(theta))
    pl.close(1)

    pl.figure(2)
    pl.plot(L_list[:],[i for i in slope],'-o')
    pl.ylabel(r'$\tau$',fontsize=16)
    pl.xlabel(r'$L$',fontsize=16)
    pl.yscale('log')
    pl.savefig(save_file+'decay_rate_theta={}.pdf'.format(theta))
    pl.close(2)