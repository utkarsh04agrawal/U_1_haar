import os
import pickle
from datetime import datetime

print(os.getcwd())

def merge(measurement_type,evolution_type,scrambling_type,setup):
    root_direc = 'data_'+measurement_type+'_measurement/'+setup+'/'

    file_dir = root_direc+scrambling_type+'_scrambling/'+evolution_type+'_evolution/'

    print(file_dir)
    print(os.listdir(root_direc))
           

    for L_file in os.listdir(file_dir):
        L = int(L_file[2:])
        print(L)
        for sub_dir in os.listdir(file_dir + L_file):
            final_dir = file_dir + L_file + '/' + sub_dir
            
            individual_dir = final_dir+'/indiviudal_files/'
            if not os.path.isdir(individual_dir):
                os.makedirs(individual_dir)
            
            merged_dir = final_dir + '/merged_files/'
            if not os.path.isdir(merged_dir):
                os.makedirs(merged_dir)

            merged_data = {'entropy':[],
                'correlation':[]}
            for data_file in os.listdir(final_dir):
                if 'shots=' not in data_file:
                    continue
                with open(final_dir+'/'+data_file,'rb') as f:
                    data = pickle.load(f)
                merged_data['entropy'].extend(data['entropy'])
                merged_data['correlation'].extend(data['correlation'])
                os.rename(final_dir+'/'+data_file, individual_dir+data_file)
            
            N = len(merged_data['entropy'])
            if N>0:
                with open(merged_dir+'/data_'+datetime.today().strftime('%Y-%m-%d')+'_samples='+str(N),'wb') as f:
                    pickle.dump(merged_data,f)
            

