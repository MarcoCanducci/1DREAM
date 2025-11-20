import pandas as pd
import numpy as np


import pickle
import sys
sys.path.append('../../../SGTM/Python')


from AGTMModule import *

#Loading the data

NoisyD = pd.read_csv('../Synthetic_Manifold_2.csv', header=None).to_numpy()

FG, Subsets = pickle.load(open("../Output/Crawling_output.pkl", 'rb'))


#User parameters
IntDim = 1
radius = 3.5
epsilon = 2
mem = 2


net , logL , GMDist , NoisyMan = Standardized_AGTM_InitTrain(FG, NoisyD, IntDim,radius,epsilon,mem)

#For the output, mostly net, GMDist, and NoisyMan are relevant.
pickle.dump((net, GMDist, NoisyMan), open("../Output/SGTM_output.pkl", 'wb'))

#Exporting the centers of the gaussians after training. 
#These centers labelled here V_ are stored in the gmmnet property of net. 

for iter in range(len(net)):
    node_pos = net[iter].gmmnet.V_
    output_node_pos_name = "../Output/SGTM_positions_node_" + str(iter+1) + ".csv"
    np.savetxt(output_node_pos_name, node_pos, delimiter=',', fmt = '%5.8f')
    
    
    
    

    
    
    
