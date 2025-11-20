##########################################################################################
##########################################################################################
######################################  LIBRARIES  #######################################
##########################################################################################
##########################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

##########################################################################################
##########################################################################################
####################################  FIGURE PARAMTERS  ##################################
##########################################################################################
##########################################################################################

tick_size = 20
legend_size = 20
label_size = 20
title_size = 30
other_size = 10

plt.rc('font', size=other_size)          # controls default text sizes
plt.rc('axes', titlesize=title_size)     # fontsize of the plot title
plt.rc('axes', labelsize=label_size)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=tick_size)    # fontsize of the tick labels
plt.rc('ytick', labelsize=tick_size)    # fontsize of the tick labels
plt.rc('legend', fontsize=legend_size)    # legend fontsize
plt.rc('figure', titlesize=other_size)  # fontsize of the figure title

##########################################################################################
##########################################################################################
####################################  MANIFOLD PARAMTERS  ################################
##########################################################################################
##########################################################################################

example_name = "Synthetic_Manifold_1"

MBMS_data_org =   pd.read_csv('../Output/MBMS_output.csv', header=None).to_numpy()
dimindexes =   pd.read_csv('../Output/dimindexes.csv', header=None).to_numpy()
Labels = np.loadtxt('../Output/Labels.csv', delimiter=',', dtype=int)

MBMS_data = MBMS_data_org[np.where(Labels==1)]

#Original Index
#Filtering the data
data_dimidx_original_1 = MBMS_data[dimindexes[:,0]==0]
data_dimidx_original_2 = MBMS_data[dimindexes[:,0]==1]
data_dimidx_original_3 = MBMS_data[dimindexes[:,0]==2]
idx1_size_original = len(data_dimidx_original_1)
idx2_size_original = len(data_dimidx_original_2)
idx3_size_original = len(data_dimidx_original_3)

#Smoothed index
#Filtering the data
data_dimidx_smoothed_1 = MBMS_data[dimindexes[:,1]==0]
data_dimidx_smoothed_2 = MBMS_data[dimindexes[:,1]==1]
data_dimidx_smoothed_3 = MBMS_data[dimindexes[:,1]==2]
idx1_size_smoothed = len(data_dimidx_smoothed_1)
idx2_size_smoothed = len(data_dimidx_smoothed_2)
idx3_size_smoothed = len(data_dimidx_smoothed_3)

##########################################################################################
##########################################################################################
#################################  PLOTTING DATA  ########################################
##########################################################################################
##########################################################################################

print("\nPLOTTING DimIndex OUTPUT OF THE EXAMPLE " + example_name.upper() )

boundaries_max = np.max(MBMS_data)
boundaries_min = np.min(MBMS_data)

fig = plt.gcf()
fig.set_size_inches(16, 10)

#Original index full dimensions
ax = fig.add_subplot(241, projection='3d')
sc = ax.scatter(data_dimidx_original_1[:,0],data_dimidx_original_1[:,1],data_dimidx_original_1[:,2], s=0.5,color = "red")
sc = ax.scatter(data_dimidx_original_2[:,0],data_dimidx_original_2[:,1],data_dimidx_original_2[:,2], s=0.5,color = "blue")
sc = ax.scatter(data_dimidx_original_3[:,0],data_dimidx_original_3[:,1],data_dimidx_original_3[:,2], s=0.5,color = "magenta")
ax.view_init(44, 125)
ax.set_title("Original Index")
ax.set_xlabel('X-axis'  , labelpad=10)
ax.set_ylabel('Y-axis'  , labelpad=10)
ax.set_zlabel('Z-axis'  , labelpad=10)
ax.set_xlim(boundaries_min,boundaries_max)
ax.set_ylim(boundaries_min,boundaries_max)
ax.set_zlim(boundaries_min,boundaries_max)
ax.legend(["1","2","3"],title="Dimension",markerscale=15,ncol=3,loc=  (0.0,0.78), columnspacing=0 , borderpad=0.05,handletextpad=0,title_fontsize=20)

#Original index 1 dimension structures
ax = fig.add_subplot(242, projection='3d')
sc = ax.scatter(data_dimidx_original_1[:,0],data_dimidx_original_1[:,1],data_dimidx_original_1[:,2], s=0.5,color = "red")
ax.view_init(44, 125)
ax.set_title("particles = " + str(idx1_size_original) )
ax.set_xlabel('X-axis'  , labelpad=10)
ax.set_ylabel('Y-axis'  , labelpad=10)
ax.set_zlabel('Z-axis'  , labelpad=10)
ax.set_xlim(boundaries_min,boundaries_max)
ax.set_ylim(boundaries_min,boundaries_max)
ax.set_zlim(boundaries_min,boundaries_max)

#Original index 2 dimension structures
ax = fig.add_subplot(243, projection='3d')
sc = ax.scatter(data_dimidx_original_2[:,0],data_dimidx_original_2[:,1],data_dimidx_original_2[:,2], s=0.5,color = "blue")
ax.view_init(44, 125)
ax.set_title("particles = " + str(idx2_size_original) )
ax.set_xlabel('X-axis'  , labelpad=10)
ax.set_ylabel('Y-axis'  , labelpad=10)
ax.set_zlabel('Z-axis'  , labelpad=10)
ax.set_xlim(boundaries_min,boundaries_max)
ax.set_ylim(boundaries_min,boundaries_max)
ax.set_zlim(boundaries_min,boundaries_max)

#Original index 3 dimension structures
ax = fig.add_subplot(244, projection='3d')
sc = ax.scatter(data_dimidx_original_3[:,0],data_dimidx_original_3[:,1],data_dimidx_original_3[:,2], s=0.5,color = "magenta")
ax.view_init(44, 125)
ax.set_title("particles = " + str(idx3_size_original) )
ax.set_xlabel('X-axis'  , labelpad=10)
ax.set_ylabel('Y-axis'  , labelpad=10)
ax.set_zlabel('Z-axis'  , labelpad=10)
ax.set_xlim(boundaries_min,boundaries_max)
ax.set_ylim(boundaries_min,boundaries_max)
ax.set_zlim(boundaries_min,boundaries_max)

#Smoothed index full dimensions
ax = fig.add_subplot(245, projection='3d')
sc = ax.scatter(data_dimidx_smoothed_1[:,0],data_dimidx_smoothed_1[:,1],data_dimidx_smoothed_1[:,2], s=0.5,color = "red")
sc = ax.scatter(data_dimidx_smoothed_2[:,0],data_dimidx_smoothed_2[:,1],data_dimidx_smoothed_2[:,2], s=0.5,color = "blue")
sc = ax.scatter(data_dimidx_smoothed_3[:,0],data_dimidx_smoothed_3[:,1],data_dimidx_smoothed_3[:,2], s=0.5,color = "magenta")
ax.view_init(44, 125)
ax.set_title("Smoothed Index")
ax.set_xlabel('X-axis'  , labelpad=10)
ax.set_ylabel('Y-axis'  , labelpad=10)
ax.set_zlabel('Z-axis'  , labelpad=10)
ax.set_xlim(boundaries_min,boundaries_max)
ax.set_ylim(boundaries_min,boundaries_max)
ax.set_zlim(boundaries_min,boundaries_max)

#Smoothed index 1 dimension structures
ax = fig.add_subplot(246, projection='3d')
sc = ax.scatter(data_dimidx_smoothed_1[:,0],data_dimidx_smoothed_1[:,1],data_dimidx_smoothed_1[:,2], s=0.5,color = "red")
ax.view_init(44, 125)
ax.set_title("particles = " + str(idx1_size_smoothed) )
ax.set_xlabel('X-axis'  , labelpad=10)
ax.set_ylabel('Y-axis'  , labelpad=10)
ax.set_zlabel('Z-axis'  , labelpad=10)
ax.set_xlim(boundaries_min,boundaries_max)
ax.set_ylim(boundaries_min,boundaries_max)
ax.set_zlim(boundaries_min,boundaries_max)

#Smoothed index 2 dimension structures
ax = fig.add_subplot(247, projection='3d')
sc = ax.scatter(data_dimidx_smoothed_2[:,0],data_dimidx_smoothed_2[:,1],data_dimidx_smoothed_2[:,2], s=0.5,color = "blue")
ax.view_init(44, 125)
ax.set_title("particles = " + str(idx2_size_smoothed) )
ax.set_xlabel('X-axis'  , labelpad=10)
ax.set_ylabel('Y-axis'  , labelpad=10)
ax.set_zlabel('Z-axis'  , labelpad=10)
ax.set_xlim(boundaries_min,boundaries_max)
ax.set_ylim(boundaries_min,boundaries_max)
ax.set_zlim(boundaries_min,boundaries_max)

#Smoothed index 3 dimension structures
ax = fig.add_subplot(248, projection='3d')
sc = ax.scatter(data_dimidx_smoothed_3[:,0],data_dimidx_smoothed_3[:,1],data_dimidx_smoothed_3[:,2], s=0.5,color = "magenta")
ax.view_init(44, 125)
ax.set_title("particles = " + str(idx3_size_smoothed) )
ax.set_xlabel('X-axis'  , labelpad=10)
ax.set_ylabel('Y-axis'  , labelpad=10)
ax.set_zlabel('Z-axis'  , labelpad=10)
ax.set_xlim(boundaries_min,boundaries_max)
ax.set_ylim(boundaries_min,boundaries_max)
ax.set_zlim(boundaries_min,boundaries_max)

plt.tight_layout(pad=5.0, w_pad=5.0, h_pad=8.0)

##########################################################################################
##########################################################################################
######################################  EXPORTING  #######################################
##########################################################################################
##########################################################################################

print("\nEXPORTING THE PLOT")

plt.savefig('../Images/4.-DimIndex_output.png')

print("\nSCRIPT FINISHED\n")

##########################################################################################

#plt.show()