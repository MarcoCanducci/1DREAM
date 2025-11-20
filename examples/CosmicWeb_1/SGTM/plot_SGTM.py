##########################################################################################
##########################################################################################
######################################  LIBRARIES  #######################################
##########################################################################################
##########################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import colorsys 
import pickle

import sys
sys.path.append('../../../SGTM/Python')
from AGTMModule import *

from warnings import simplefilter   #Skip sklearn warnings 
simplefilter("ignore", category=ConvergenceWarning)

##########################################################################################
##########################################################################################
####################################  FIGURE PARAMTERS  ##################################
##########################################################################################
##########################################################################################

tick_size = 10
legend_size = 10
label_size = 10
title_size = 30
sub_title_size = 20
other_size = 10

plt.rc('font', size=other_size)          # controls default text sizes
plt.rc('axes', titlesize=sub_title_size)     # fontsize of the plot title
plt.rc('axes', labelsize=label_size)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=tick_size)    # fontsize of the tick labels
plt.rc('ytick', labelsize=tick_size)    # fontsize of the tick labels
plt.rc('legend', fontsize=legend_size)    # legend fontsize
plt.rc('figure', titlesize=title_size)  # fontsize of the figure title

##########################################################################################
##########################################################################################
####################################  MANIFOLD PARAMTERS  ################################
##########################################################################################
##########################################################################################

example_name = "CosmicWeb_1"

data = pd.read_csv('../' + example_name + '.csv', header=None).to_numpy()

FG, Subsets = pickle.load(open("../Output/Crawling_output.pkl", 'rb'))
net, GMDist, NoisyMan = pickle.load(open("../Output/SGTM_output.pkl", 'rb'))

#Importing the positions of the centers of the gaussian trained by SGTM.
nodes_pos = [] 
for iter in range(len(net)):
	node_pos = pd.read_csv("../Output/SGTM_positions_node_" + str(iter + 1) + ".csv", header=None).to_numpy()
	nodes_pos.append(node_pos)

number_of_subsets = len(nodes_pos)

##########################################################################################
##########################################################################################
#################################  PLOTTING DATA  ########################################
##########################################################################################
##########################################################################################

print("\nPLOTTING DimIndex OUTPUT OF THE EXAMPLE " + example_name.upper() )

#getting colors
def getDistinctColors(n): 
    return [colorsys.hsv_to_rgb(1.0 / (n + 1)  * value, 1.0, 1.0) for value in range(0, n)]

subplot_columns = np.ceil(np.sqrt(number_of_subsets+1)).astype(int)
colors_pallette = np.array(getDistinctColors(number_of_subsets+1))
subplot_rows = np.ceil((number_of_subsets+1)/subplot_columns).astype(int)

#plot boundaries
boundaries_min_x = np.min(data[:,0]) 
boundaries_min_y = np.min(data[:,1]) 
boundaries_min_z = np.min(data[:,2]) 
boundaries_max_x = np.max(data[:,0]) 
boundaries_max_y = np.max(data[:,1]) 
boundaries_max_z = np.max(data[:,2]) 

# Plotting GMM filamnets
#Plot
fig = plt.gcf()
fig.set_size_inches(16, 10)

st = fig.suptitle("Likelihood to belong to the different filaments")

ax = fig.add_subplot(subplot_rows,subplot_columns,1, projection='3d')
ax.scatter(data[:,0], data[:,1], data[:,2], s=0.001,c = [colors_pallette[0]], zorder = 1, label = "Data")
for iter in range(number_of_subsets):
    counter = 5
    for tup in FG[iter].edges():
        t1 = nodes_pos[iter][tup[0]]
        t2 = nodes_pos[iter][tup[1]]
        x = np.array((t1[0], t2[0]))
        y = np.array((t1[1], t2[1]))
        z = np.array((t1[2], t2[2]))
        if counter == 5:
            ax.plot(x, y, z, c =colors_pallette[iter+1],linewidth=1,marker='D', zorder = counter , label='Fil ' + str(iter+1))
        else:
            ax.plot(x, y, z, c =colors_pallette[iter+1],linewidth=1,marker='D', zorder = counter)
        counter = counter + 5

ax.view_init(44, 125)
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title('SGTM filaments')
lgnd = plt.legend(borderpad=0.05,handletextpad=0.5)
lgnd.legend_handles[0]._sizes = [40]

#Likelihood of every filament
for filament_idx in range(number_of_subsets):
     # Getting the log-likelihoods of all particles in the dataset 
# from the model of the arm with index filament_idx (as an example). 
    GM = GMDist[filament_idx].fit(NoisyMan[filament_idx])
    S = GM.score_samples(data)
    #Color points by the computed likelihoods 
    ax = fig.add_subplot(subplot_rows,subplot_columns,2 + filament_idx, projection='3d')
    sc = ax.scatter(data[:,0], data[:,1], data[:,2], s=0.001, c=np.exp(0.001*S))

    ax.view_init(44, 125)
    ax.set_xlabel('X-axis'  , labelpad=10)
    ax.set_ylabel('Y-axis'  , labelpad=10)
    ax.set_zlabel('Z-axis'  , labelpad=10)
    ax.set_xlim(boundaries_min_x,boundaries_max_x)
    ax.set_ylim(boundaries_min_y,boundaries_max_y)
    ax.set_zlim(boundaries_min_z,boundaries_max_z)
    ax.set_title("Filament " + str(filament_idx+1))
    plt.colorbar(sc,pad=0.09)

plt.tight_layout(pad=subplot_columns, w_pad=0 , h_pad=subplot_rows)

##########################################################################################
##########################################################################################
######################################  EXPORTING  #######################################
##########################################################################################
##########################################################################################

print("\nEXPORTING THE PLOT")

plt.savefig('../Images/6.-SGTM_output.png')

print("\nSCRIPT FINISHED\n")

##########################################################################################

#plt.show()
