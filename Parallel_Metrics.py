import numpy as np
import pickle
import glob
import matplotlib.pyplot as plt
from scoring import *
from collections import defaultdict
import torch

def nested_defaultdict():
    return defaultdict(nested_defaultdict)

config_dict = nested_defaultdict()



def _get_network_type(weight_dict):
    if any(x.startswith('s_layer_') for x in weight_dict):
        if any(x.startswith('t_layer_') for x in weight_dict):
            network = 'transfer'
        else:
            network = 'multitask'
    else:
        network = 'baseline'
    return network


def _get_layer_key(layer):
    ordinal = 0 if layer.startswith('s_layer') else 2 if 'output' in layer else 1
    try:
        num = int(layer.split('_')[-1])
    except ValueError:
        num = 0
    return ordinal, num


def _get_layers(weight_dict, network, robot_name):
    if network == 'multitask':
        keys = (k for k in weight_dict if (k.startswith('s_layer_') or k.startswith(robot_name)))
    else:
        keys = weight_dict
    return list(sorted((x.split('/')[0] for x in keys), key=_get_layer_key))


def _relu(x):
    x[x < 0] = 0
    return x


def _linear(x):
    return x


def calc_layer_outputs(weight_dict, bias_dict, inp, robot_name, network=None):
    if network is None:
        network = _get_network_type(weight_dict)
    layers = _get_layers(weight_dict, network, robot_name)
    print(layers)
    wkeys = [l + '/weights' for l in layers]
    bkeys = [l + '/bias' for l in layers]
    activation = [_linear if 'output' in l else _relu for l in layers]
    x = inp
    results = dict()
    for l, w, b, sigma in zip(layers, wkeys, bkeys, activation):
        w = weight_dict[w]
        b = bias_dict[b]
        x = sigma(np.matmul(x, w) + b)
        results[l] = x
    return results


def process_network(data):
    weight_dict = data[0]
    network = _get_network_type(weight_dict)
    bias_dict = data[1]
    robots = list(filter(lambda x: x != 'input', data[2]))

    results = dict()
    for robot in robots:
        if network == 'multitask':
            inp = data[2]['input'][robot]
        else:
            inp = data[2][robot]['input']
        results[robot] = calc_layer_outputs(weight_dict, bias_dict, inp, robot, network)
    return results



# The Metric
def procrustes(A, B):
    """
    Computes Procrustes distance between representations A and B
    """
    A_sq_frob = torch.sum(A ** 2)
    B_sq_frob = torch.sum(B ** 2)
    nuc = torch.linalg.norm(A @ B.T, ord="nuc")  # O(p * p * n)
    return A_sq_frob + B_sq_frob - 2 * nuc

def Normal(A):
    # Subtract the mean value from each column
    mean_column = torch.mean(A, axis=0)
    A_centered = A - mean_column

    # Compute the Frobenius norm
    frobenius_norm = torch.linalg.norm(A_centered, 'fro')

    # Divide by the Frobenius norm to obtain the normalized representation A*
    A_normalized = A_centered / frobenius_norm
    return A_normalized

# Make sure you have a GPU available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    print("No GPU available, using CPU instead.")

# implement it on the dictionary

with open('F_dict.pkl', 'rb') as f:
     PA_dict = pickle.load(f)


# A_all = process_network(PA_dict['All']['Fixed_BSW_S1'][0])
# B_all = process_network(PA_dict['All']['Down_W_S0'][0])
# C_all = process_network(PA_dict['All']['Down_BSWtoK_S1T1'][0]) 

# plot 

# Keys_A = list(A_all['All'].keys())
# Keys_B = list(B_all['All'].keys())
Keys_All = list(PA_dict['All'].keys())

d_list=[]
config_dict = nested_defaultdict()
for Index1 in range(len(Keys_All)):
    for Index2 in range(len(Keys_All)):
        if 'Fixed' in Keys_All[Index1] and 'Fixed' in Keys_All[Index2]:# and 'S0' in Keys_All[Index1] and 'S0' in Keys_All[Index2]:
            print("config1=", Keys_All[Index1])
            print("config2=", Keys_All[Index2])
            # for keys1 in PA_dict['All'][Keys_All[Index1]]:
            #     for keys2 in PA_dict['All'][Keys_All[Index2]]:

                        # Keys_A_Output = list(A_all.keys())
                        # Keys_B_Output = list(B_all.keys())
                        # for j in 

            A = process_network(PA_dict['All'][Keys_All[Index1]][0])
            B = process_network(PA_dict['All'][Keys_All[Index2]][0])
            for keys1 in A.keys():
                for keys2 in B.keys():
                    for Layer in range(4):
                        A_Layer_Keys_list = list(A[keys1].keys())
                        B_Layer_Keys_list = list(B[keys2].keys())

                        
                        A_torch = torch.tensor(A[keys1][A_Layer_Keys_list[Layer]], device=device, dtype=torch.float)
                        B_torch = torch.tensor(B[keys2][B_Layer_Keys_list[Layer]], device=device, dtype=torch.float)

                        # A_torch_N = torch.tensor(A[keys1][A_Layer_Keys_list[Layer]], device=device, dtype=torch.float)
                        # B_torch_N = torch.tensor(B[keys2][B_Layer_Keys_list[Layer]], device=device, dtype=torch.float)

                        

                        ditance = procrustes(Normal(A_torch), Normal(B_torch))
                        d_list.append(ditance)
                        # [Keys_All[Index1] A.keys() # A_Layer_Keys_list 
                        distance_numpy = ditance.cpu().numpy()
                        config = Keys_All[Index1] + keys1 + ' ' +  'to' + ' ' + Keys_All[Index2] + keys2
                        config_dict[Keys_All[Index1]][keys1][Keys_All[Index2]][keys2][Layer] = distance_numpy   

                        plt.scatter(Layer+1, distance_numpy, label=config)#, marker=markers)
                    
print("distance=", d_list)
# save the dict
with open('config_dict.pkl', 'wb') as file:
    pickle.dump(config_dict, file)
# Plot Histogram

#plt.hist(d_list, bins=5, edgecolor='black')

# Plot scatter plot

# Add labels and a title
plt.xlabel(' Layer ')
plt.ylabel('Distance Values')
plt.legend(bbox_to_anchor=(1.1, 1.16), loc='upper right', fontsize=5)
# plt.title('Histogram')
# Save the plot to a file
plt.savefig('Distance_All_Layers1ls.png')
# Display the histogram
plt.show()

# plt.scatter(list(range(1,5)), d_list)
# plt.xlabel('Layers')
# plt.ylabel('procrustes distance')
# plt.title('Comparing the distance for each layer for Down_BKStoW_S1T1 and Down_W_S0')

# plt.show()

# B=
