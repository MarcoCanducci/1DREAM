import pandas as pd
import numpy as np


import pickle
import sys
sys.path.append('../../../Crawling/Python')



from CrawlingModule import *

#Loading the data
NoisyD = pd.read_csv('../Synthetic_Manifold_2.csv', header=None).to_numpy()
spine_data =   pd.read_csv('../Output/Selected_data_after_DimIndex_original.csv', header=None).to_numpy()
#spine_data =   pd.read_csv('../Output/Selected_data_after_DimIndex_smoothed.csv', header=None).to_numpy()

#User parameters
radius = 3.5
ldim = 1
betha = 0.4


#Running Crawling function
FG, NoisyMan, _ = MultiM(spine_data,NoisyD,radius,ldim,betha)

print("Number of spines = ", len(NoisyMan))

#Exporting Data in binary format
pickle.dump((FG, NoisyMan), open("../Output/Crawling_output.pkl", 'wb'))


for i,G in enumerate(FG):
	#To retrieve positions of all nodes of a graph G
	pos = AllNodeNames(G, 3)
	np.savetxt("../Output/NodePos"+str(i+1)+".csv" , pos, delimiter=',', fmt = '%5.8f')
	
	#To retrieve the subset of points surrounding the graph G
	Subset = NoisyMan[i]
	np.savetxt("../Output/Subset"+str(i+1)+".csv" , Subset, delimiter=',', fmt = '%5.8f')	
	
	
	
	    
	    




