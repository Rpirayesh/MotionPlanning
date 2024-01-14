import numpy as np
import pickle
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib.patches as mpatches
from N_Values import *
from Joint2 import *
import pandas as pd
# from sklearn.feature_selection import mutual_info_regression

def nested_defaultdict():
    return defaultdict(nested_defaultdict)

def Dot_Product_Normalized(List1, List2):
      
    # Convert lists to NumPy arrays
    array1 = np.array(List1)
    array2 = np.array(List2)

    # Compute dot product
    dot_product = np.dot(array1, array2)

    # Compute norms
    norm1 = np.linalg.norm(array1)
    norm2 = np.linalg.norm(array2)

    # Normalize dot product
    normalized_dot_product = dot_product / (norm1 * norm2)
    return normalized_dot_product

Config1_desired = ['Fixed_B_S0', 'Fixed_K_S0', 'Fixed_S_S0', 'Fixed_W_S0']
# Config1_desired = ['Fixed_S_S0']
# Config1_desired_robots = ['Baxter', 'Wam', 'Sawyer', 'Kuka']
# Config2_desired = ['S0', 'S1', 'S1T1']
# Config2_desired = 'S1'
# Config2_desired = 'S1T1'
# Desired_Layers = ['0','1','2','3']

Ditance_list = []


Metric_dict = {
    'Kuka': {'BB': [], 'R_BB': [], 'ST_BB': [], 'BB-Config': [],'BMT': [],'ST_BMT': [] ,'BMT-Config': [], 'R_BMT': [], 'BT': [], 'ST_BT': [],'BT_Config': [], 'R_BT': [], 'N_E':[], 'Mape-T':[], 'Sim_Rank': []},
    'Wam': {'BB': [], 'R_BB': [], 'ST_BB': [], 'BB-Config': [],'BMT': [],'ST_BMT': [] ,'BMT-Config': [], 'R_BMT': [], 'BT': [], 'ST_BT': [],'BT_Config': [], 'R_BT': [], 'N_E':[], 'Mape-T':[], 'Sim_Rank': []},
    'Sawyer': {'BB': [], 'R_BB': [], 'ST_BB': [], 'BB-Config': [],'BMT': [],'ST_BMT': [] ,'BMT-Config': [], 'R_BMT': [], 'BT': [], 'ST_BT': [],'BT_Config': [], 'R_BT': [], 'N_E':[], 'Mape-T':[], 'Sim_Rank': []},
    'Baxter': {'BB': [], 'R_BB': [], 'ST_BB': [], 'BB-Config': [],'BMT': [],'ST_BMT': [] ,'BMT-Config': [], 'R_BMT': [], 'BT': [], 'ST_BT': [],'BT_Config': [], 'R_BT': [], 'N_E':[], 'Mape-T':[], 'Sim_Rank': []},
    'All': {'Sim_Rank': []}
}
 


Cleaned_List = ['Down_B_S0', 'Down_K_S0', 'Down_S_S0', 'Down_W_S0', 'Fixed_B_S0', 'Fixed_K_S0', 'Fixed_S_S0', 'Fixed_W_S0', 'Up_B_S0', 'Up_K_S0', 'Up_S_S0', 'Up_W_S0', 'Down_BK_S1', 'Down_BS_S1', 'Down_BW_S1', 'Down_KS_S1', 'Down_KW_S1', 'Down_SW_S1', 'Down_BKS_S1', 'Down_BKW_S1', 'Down_BSW_S1', 'Down_KSW_S1', 'Fixed_BK_S1', 'Fixed_BS_S1', 'Fixed_BW_S1', 'Fixed_KS_S1', 'Fixed_KW_S1', 'Fixed_SW_S1', 'Fixed_BKS_S1', 'Fixed_BKW_S1', 'Fixed_BSW_S1', 'Fixed_KSW_S1', 'Up_BK_S1', 'Up_BS_S1', 'Up_BW_S1', 'Up_KS_S1', 'Up_KW_S1', 'Up_SW_S1', 'Up_BKS_S1', 'Up_BKW_S1', 'Up_BSW_S1', 'Up_KSW_S1', 'Down_KStoB_S1T1', 'Down_KWtoB_S1T1', 'Down_SWtoB_S1T1', 'Down_KSWtoB_S1T1', 'Down_BStoK_S1T1', 'Down_BWtoK_S1T1', 'Down_SWtoK_S1T1', 'Down_BSWtoK_S1T1', 'Down_BKtoS_S1T1', 'Down_BWtoS_S1T1', 'Down_KWtoS_S1T1', 'Down_BKWtoS_S1T1', 'Down_BKtoW_S1T1', 'Down_BStoW_S1T1', 'Down_KStoW_S1T1', 'Down_BKStoW_S1T1', 'Fixed_KStoB_S1T1', 'Fixed_KWtoB_S1T1', 'Fixed_SWtoB_S1T1', 'Fixed_KSWtoB_S1T1', 'Fixed_BStoK_S1T1', 'Fixed_BWtoK_S1T1', 'Fixed_SWtoK_S1T1', 'Fixed_BSWtoK_S1T1', 'Fixed_BKtoS_S1T1', 'Fixed_BWtoS_S1T1', 'Fixed_KWtoS_S1T1', 'Fixed_BKWtoS_S1T1', 'Fixed_BKtoW_S1T1', 'Fixed_BStoW_S1T1', 'Fixed_KStoW_S1T1', 'Fixed_BKStoW_S1T1', 'Up_KStoB_S1T1', 'Up_KWtoB_S1T1', 'Up_SWtoB_S1T1', 'Up_KSWtoB_S1T1', 'Up_BStoK_S1T1', 'Up_BWtoK_S1T1', 'Up_SWtoK_S1T1', 'Up_BSWtoK_S1T1', 'Up_BKtoS_S1T1', 'Up_BWtoS_S1T1', 'Up_KWtoS_S1T1', 'Up_BKWtoS_S1T1', 'Up_BKtoW_S1T1', 'Up_BStoW_S1T1', 'Up_KStoW_S1T1', 'Up_BKStoW_S1T1']
Configs_List = []
for Config1_desired_Index in Cleaned_List:
    if 'Fixed' in Config1_desired_Index:
        Configs_List.append(Config1_desired_Index)


# Make the the baselines dictionary
Index0=[]
All_R_BT=[]
All_R_BMT = []
# Keys_All = list(PA_dict['All'].keys())
        
# Indexing the parallel
N_Pc = 5
N = len(Configs_List) 
Computer = 'baymax'
ind1 = 1 if Computer == 'baymax' else (2 if Computer == 'glados' else (3 if Computer == 'kitt' else(4 if Computer =='baymax'  else 5)))
# ind1 = 1
Begin =  int((ind1-1)*N/N_Pc)
End = int((ind1)*N/N_Pc)
dict_Table_comp = {}
dict_Table_comp['Config-Sour'] = []
dict_Table_comp['Config-Targ'] = []
dict_Table_comp['Layer1_Sim'] = []
dict_Table_comp['Layer2_Sim'] = []
dict_Table_comp['Layer3_Sim'] = []
# dict_Table_comp['Layer_Sim_Ave'] = []
dict_Table_comp['Epoch-Sour'] = []
dict_Table_comp['Epoch-Targ'] = []
dict_Table_comp['Epoch-N'] = []
dict_Table_comp['Error_Sour'] = []
dict_Table_comp['Error_Targ'] = []
dict_Table_comp['Error_N'] = []
# dict_Table_comp['EXP'] = []

 
ii = 0
for Index0 in Config1_desired:
    key_dict = 'Kuka' if 'K' in Index0 else ('Wam' if 'W' in Index0 else ('Baxter' if 'B' in Index0 else('Sawyer' if 'S' in Index0 else None)))
    print("keys=", key_dict)
    # make the Baseline to MT and T
    Index1 = []
    
    for Index1 in Configs_List:#[Begin:End]:
        Index1_Index = Index1.split('_')
        # Go through the layers
        for Layer in range(3):
            key_Layer = 'Layer1_Sim' if Layer==0 else ('Layer2_Sim' if Layer==1 else 'Layer3_Sim')  
            if 'S0' == Index1_Index[2] and key_dict[0] == Index1_Index[1]: 
                Data_in_A, Mape_A, Epochs_A, Best_Config_A = PathData(Index1)
                A = process_network(Data_in_A)
                Index2 = list(A.keys())[-1]  # Since we only have one robnot for the baseline
                A_Layer_Keys_list = list(A[Index2].keys())


                
                    # if Config1_desired_robots in Index2:
                for Index3 in Configs_List:
                    Index3_Index = Index3.split('_')
                    if  'S1' == Index3_Index[2]:# and key_dict[0] not in Index3_Index[1]:
                        # B = process_network(PA_dict['All'][Index3][0])
                        Data_in_B, Mape_B, Epochs_B, Best_Config = PathData(Index3)
                        B = process_network(Data_in_B)

                        for Index4 in B.keys():
                            B_Layer_Keys_list = list(B[Index4].keys())
                            # distance_procrustes = float(str(compute_normalized_mutual_information_large((A[Index2][A_Layer_Keys_list[Layer]]), (B[Index4][B_Layer_Keys_list[Layer]])))[0:9])
                            distance_procrustes = compute_normalized_mutual_information((A[Index2][A_Layer_Keys_list[Layer]]), (B[Index4][B_Layer_Keys_list[Layer]]))
                            print('Sim=',distance_procrustes)
                            # distance_procrustes = normalized_mutual_information(Normal(A[Index2][A_Layer_Keys_list[Layer]]), Normal(B[Index4][B_Layer_Keys_list[Layer]]))

                            dict_Table_comp[key_Layer].append(distance_procrustes) 

                            Metric_dict[key_dict]['BMT'].append(distance_procrustes)
                            config =  Index3 + '_' + Index4 + '_' + 'L' +  str(Layer) 
                            Metric_dict[key_dict]['BMT-Config'].append(config)

                                        # for the table comparison

                                        
                            if Layer == 2:
                                dict_Table_comp['Config-Sour'].append(Best_Config_A)
                                dict_Table_comp['Epoch-Sour'].append(Epochs_A)
                                dict_Table_comp['Error_Sour'].append(Mape_A)
                                ii+= 1 
                                # Sum_Layers = dict_Table_comp['Layer1_Sim'][ii] + dict_Table_comp['Layer2_Sim'][ii] + distance_procrustes
                                # dict_Table_comp['Layer_Sim_Ave'].append(Sum_Layers)                                
                                Targ = Best_Config + '_' + Index4
                                dict_Table_comp['Config-Targ'].append(Targ)
                                dict_Table_comp['Epoch-Targ'].append(Epochs_B)
                                dict_Table_comp['Error_Targ'].append(Mape_B)
                                Epoch_Diff_T = (int(Epochs_A)-int(Epochs_B))/int(Epochs_A)
                                dict_Table_comp['Epoch-N'].append(Epoch_Diff_T)
                                Acc_N = (Mape_A-Mape_B)/Mape_A
                                dict_Table_comp['Error_N'].append(Acc_N) 


                        for Index5 in Configs_List:
                            Index5_Index = Index5.split('_')
                            # string = Index3_Index[1]
                            # index_to = string.find('to')
                            # Index1 in string[index_to + len(substring_to):]
                            Ind = Index5_Index[1]
                            if  'S1T1' == Index5_Index[2] and key_dict[0] == Ind[-1] and Ind[0:-3] == Index3_Index[1]:
                                Data_in_B, Mape_B, Epochs_B, Best_Config = PathData(Index5)
                                B = process_network(Data_in_B)
                                # going throught the robots
                                for Index6 in B.keys():
                                    # Index7 = 0
                                    B_Layer_Keys_list = list(B[Index6].keys())
                                    distance_procrustes = compute_normalized_mutual_information((A[Index2][A_Layer_Keys_list[Layer]]), (B[Index6][B_Layer_Keys_list[Layer]]))
                                    print('Sim=',distance_procrustes)
                                    dict_Table_comp[key_Layer].append(distance_procrustes) 
                                    # distance_procrustes = PA_dict[Index1][Index2][Index5][Index6][Layer]['procrustes']
                                    Metric_dict[key_dict]['BT'].append(distance_procrustes)
                                    config = Index5 + '_' + Index6 + '_' +  'L' + str(Layer) 
                                    Metric_dict[key_dict]['BT_Config'].append(config)
                                    Epoch_Diff = (int(Epochs_A)-int(Epochs_B))/int(Epochs_A)

                                    # Targ = Best_Config + '_' + Index6 
                                    # dict_Table_comp['Config-Targ'].append(Targ)                                
                                    Metric_dict[key_dict]['N_E'].append(Epoch_Diff) 
                                    Metric_dict[key_dict]['Mape-T'].append(Mape_B)  

                                        # for the table comparison
                                    if Layer == 2:
                                        dict_Table_comp['Config-Sour'].append(Best_Config_A)
                                        dict_Table_comp['Epoch-Sour'].append(Epochs_A)
                                        dict_Table_comp['Error_Sour'].append(Mape_A)
                                        ii+= 1 
                                        # Sum_Layers = dict_Table_comp['Layer1_Sim'][ii] + dict_Table_comp['Layer2_Sim'][ii] + distance_procrustes
                                        # dict_Table_comp['Layer_Sim_Ave'].append(Sum_Layers)
                                        Targ = Best_Config + '_' + Index6
                                        dict_Table_comp['Config-Targ'].append(Targ)
                                        dict_Table_comp['Epoch-Targ'].append(Epochs_B)
                                        dict_Table_comp['Error_Targ'].append(Mape_B)
                                        Epoch_Diff_T = (int(Epochs_A)-int(Epochs_B))/int(Epochs_A)
                                        dict_Table_comp['Epoch-N'].append(Epoch_Diff_T)
                                        Acc_N = (Mape_A-Mape_B)/Mape_A
                                        dict_Table_comp['Error_N'].append(Acc_N) 


                        
                    ## Baseline to baseline
                    if  'S0' == Index3_Index[2]:# and key_dict[0] not in Index3_Index[1]:
                        Data_in_B, Mape_B, Epochs_B, Best_Config = PathData(Index3)
                        B = process_network(Data_in_B)
                        Index8 = list(B.keys())[-1]  # Since we only have one robnot for the baseline
                        B_Layer_Keys_list = list(B[Index8].keys())
                        # Index9_BMT = 0
                        distance_procrustes = compute_normalized_mutual_information((A[Index2][A_Layer_Keys_list[Layer]]), (B[Index8][B_Layer_Keys_list[Layer]]))
                        print('Sim=',distance_procrustes)
                        # key_Layer = 'Layer1_Sim' if Layer==0 else ('Layer2_Sim' if Layer==1 else 'Layer3_Sim')  
                        dict_Table_comp[key_Layer].append(distance_procrustes )  
                        # distance_procrustes = PA_dict[Index1][Index2][Index3][Index8][Layer]['procrustes']
                        Metric_dict[key_dict]['BB'].append(distance_procrustes)
                        config =  Index3 + '_' + 'L' + str(Layer) 
                        Metric_dict[key_dict]['BB-Config'].append(config)   
                                                                # for the table comparison
                        if Layer == 2:
                            dict_Table_comp['Config-Sour'].append(Best_Config_A)
                            dict_Table_comp['Epoch-Sour'].append(Epochs_A)
                            dict_Table_comp['Error_Sour'].append(Mape_A)

                            ii+= 1 
                            # Sum_Layers = dict_Table_comp['Layer1_Sim'][ii] + dict_Table_comp['Layer2_Sim'][ii] + distance_procrustes
                            # dict_Table_comp['Layer_Sim_Ave'].append(Sum_Layers)
                            Targ = Best_Config
                            dict_Table_comp['Config-Targ'].append(Targ)
                            dict_Table_comp['Epoch-Targ'].append(Epochs_B)
                            Epoch_Diff_T = (int(Epochs_A)-int(Epochs_B))/int(Epochs_A)
                            dict_Table_comp['Epoch-N'].append(Epoch_Diff_T)
                            dict_Table_comp['Error_Targ'].append(Mape_B)
                            Acc_N = (Mape_A-Mape_B)/(Mape_A)
                            dict_Table_comp['Error_N'].append(Acc_N)              

    sorted_indices = np.argsort(Metric_dict[key_dict]['BMT'])
    ranking = np.argsort(sorted_indices)
    Metric_dict[key_dict]['R_BMT'] = ranking
    Metric_dict[key_dict]['ST_BMT'] = sum(Metric_dict[key_dict]['BMT'])


    sorted_indices = np.argsort(Metric_dict[key_dict]['BT'])
    ranking = np.argsort(sorted_indices)
    Metric_dict[key_dict]['R_BT'] = ranking
    Metric_dict[key_dict]['ST_BT'] = sum(Metric_dict[key_dict]['BT'])

    sorted_indices = np.argsort(Metric_dict[key_dict]['BB'])
    ranking = np.argsort(sorted_indices)
    Metric_dict[key_dict]['R_BB'] = ranking
    Metric_dict[key_dict]['ST_BB'] = sum(Metric_dict[key_dict]['BB'])
   
print('Metric_dict=',Metric_dict)

with open('dict_Table_comp.pkl', 'wb') as file:
    pickle.dump(dict_Table_comp, file)

## Plotting
import numpy as np


def Extr(config):
    z = config.split('_')
    return z[1]
print("Table=", dict_Table_comp)

# Save it as a excel
# Convert the dictionary to a DataFrame
df = pd.DataFrame(dict_Table_comp)

# Save the DataFrame to an Excel file
# df.to_excel("Table_results_MI_Large_200.xlsx", index=False)
# df.to_excel("Table_results_MI_Large_3.xlsx", index=False) # for the Normalized, hist
# df.to_excel("Table_results_MI_Large_R_sample_ratio=0.8_NoNormal.xlsx", index=False) # for the Normalized
df.to_excel("Table_results_MI_NoHist-NoNormal.xlsx", index=False)