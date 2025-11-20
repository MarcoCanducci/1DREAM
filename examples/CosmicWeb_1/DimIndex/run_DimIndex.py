import pandas as pd
import numpy as np


import sys
sys.path.append('../../../DimIndex/Python')

from DimIndexModule import *

#User parameters
radius = 1.5
cutoff = 5
Simplex = "Barycentric"
smooth = 'l2'

# The value of the Smoothed parameter can be changed inside of the Dimindex Module.
# The default value is 2.0

#Loading the data
LAAT_data_org =   pd.read_csv('../Output/LAAT_output_selected_data.csv', header=None).to_numpy()
MBMS_data_org =   pd.read_csv('../Output/MBMS_output.csv', header=None).to_numpy()

#Discard any sparse neighborhoods of size 1.5 that have less than 5 points
LAAT_data, MBMS_data, Labels, _, _ = Filtering(LAAT_data_org, MBMS_data_org, radius, cutoff)



#Running Dim index function
Struct , indexes = Dim_Index(MBMS_data,radius,Simplex,smooth)


#Selecting 1 dimensional data with the original index
idx_1dimension = indexes[:,0]
select_data_original = MBMS_data[idx_1dimension == 0]

#Selecting 1 dimensional data with the smoothed index
idx_1dimension = indexes[:,1]
select_data_smoothed = MBMS_data[idx_1dimension == 0]


print("Number of 1-dimensional points using original method = ", len(select_data_original))
print("Number of 1-dimensional points using smoothed method = ", len(select_data_smoothed))


#Exporting the Dim index values and the selected Data in the Output folder of this example
np.savetxt("../Output/Labels.csv", Labels, delimiter=',', fmt = '%i')
np.savetxt("../Output/dimindexes.csv", indexes, delimiter=',', fmt = '%d')
np.savetxt("../Output/Selected_data_after_DimIndex_original.csv", select_data_original, delimiter=',', fmt = '%5.8f')
np.savetxt("../Output/Selected_data_after_DimIndex_smoothed.csv", select_data_smoothed, delimiter=',', fmt = '%5.8f')


