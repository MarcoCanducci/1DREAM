##########################################################################################
##########################################################################################
######################################  LIBRARIES  #######################################
##########################################################################################
##########################################################################################

import numpy as np
import matplotlib.pyplot as plt
import pickle
import colorsys 
import sys
sys.path.append('../../../Crawling/Python')

from CrawlingModule import *

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

example_name = "Synthetic_Manifold_2"

FG, Subsets = pickle.load(open("../Output/Crawling_output.pkl", 'rb'))

number_of_subsets = len(FG)
print("Number of spines = ", number_of_subsets)

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
colors_pallette = np.array(getDistinctColors(number_of_subsets))
subplot_rows = np.ceil((number_of_subsets+1)/subplot_columns).astype(int)

#plot boundaries
boundaries_min_x = np.min([np.min(np.transpose(Subsets[j])[0]) for j in range(number_of_subsets)]) 
boundaries_min_y = np.min([np.min(np.transpose(Subsets[j])[1]) for j in range(number_of_subsets)]) 
boundaries_min_z = np.min([np.min(np.transpose(Subsets[j])[2]) for j in range(number_of_subsets)]) 
boundaries_max_x = np.max([np.max(np.transpose(Subsets[j])[0]) for j in range(number_of_subsets)]) 
boundaries_max_y = np.max([np.max(np.transpose(Subsets[j])[1]) for j in range(number_of_subsets)]) 
boundaries_max_z = np.max([np.max(np.transpose(Subsets[j])[2]) for j in range(number_of_subsets)]) 

#Plot
fig = plt.gcf()
fig.set_size_inches(16, 10)

#First plot containing all the filaments
ax = fig.add_subplot(subplot_rows,subplot_columns,1, projection='3d')
counter = 5
for filament_idx in range(number_of_subsets):

    sc = ax.scatter(np.transpose(Subsets[filament_idx])[0],np.transpose(Subsets[filament_idx])[1],np.transpose(Subsets[filament_idx])[2], s=0.1, c = [colors_pallette[filament_idx]],zorder=1)
    pos = AllNodeNames(FG[filament_idx], 3)
    xi = pos[:, 0]
    yi = pos[:, 1]
    zi = pos[:, 2]
    sc = ax.scatter(xi, yi, zi)

    for tup in FG[filament_idx].edges():
        t1 = np.frombuffer(FG[filament_idx].nodes[tup[0]]['Name'])
        t2 = np.frombuffer(FG[filament_idx].nodes[tup[1]]['Name'])
        x = np.array((t1[0], t2[0]))
        y = np.array((t1[1], t2[1]))
        z = np.array((t1[2], t2[2]))
        ax.plot(x, y, z, c='black',linewidth=1,zorder=counter,marker='D')
        counter = counter + 5

ax.view_init(44, 125)
ax.set_title(str(number_of_subsets) + " filaments found\n with Crawling")
ax.set_xlabel('X-axis'  , labelpad=10)
ax.set_ylabel('Y-axis'  , labelpad=10)
ax.set_zlabel('Z-axis'  , labelpad=10)
ax.set_xlim(boundaries_min_x,boundaries_max_x)
ax.set_ylim(boundaries_min_y,boundaries_max_y)
ax.set_zlim(boundaries_min_z,boundaries_max_z)

#Filaments
for filament_idx in range(number_of_subsets):
    ax = fig.add_subplot(subplot_rows,subplot_columns,2 + filament_idx, projection='3d')
    counter = 5
    sc = ax.scatter(np.transpose(Subsets[filament_idx])[0],np.transpose(Subsets[filament_idx])[1],np.transpose(Subsets[filament_idx])[2], s=0.1,c = [colors_pallette[filament_idx]],zorder=1)
    pos = AllNodeNames(FG[filament_idx], 3)
    xi = pos[:, 0]
    yi = pos[:, 1]
    zi = pos[:, 2]
    sc = ax.scatter(xi, yi, zi)
    
    for tup in FG[filament_idx].edges():
        t1 = np.frombuffer(FG[filament_idx].nodes[tup[0]]['Name'])
        t2 = np.frombuffer(FG[filament_idx].nodes[tup[1]]['Name'])
        x = np.array((t1[0], t2[0]))
        y = np.array((t1[1], t2[1]))
        z = np.array((t1[2], t2[2]))
        ax.plot(x, y, z, c='black',linewidth=1,zorder=counter,marker='D')
        counter = counter + 5

    ax.view_init(44, 125)
    ax.set_xlabel('X-axis'  , labelpad=10)
    ax.set_ylabel('Y-axis'  , labelpad=10)
    ax.set_zlabel('Z-axis'  , labelpad=10)
    ax.set_xlim(boundaries_min_x,boundaries_max_x)
    ax.set_ylim(boundaries_min_y,boundaries_max_y)
    ax.set_zlim(boundaries_min_z,boundaries_max_z)
    ax.set_title("Filament " + str(filament_idx+1))

plt.tight_layout(pad=subplot_columns, w_pad=0 , h_pad=subplot_rows)

##########################################################################################
##########################################################################################
######################################  EXPORTING  #######################################
##########################################################################################
##########################################################################################

print("\nEXPORTING THE PLOT")

plt.savefig('../Images/5.-Crawling_output.png')

print("\nSCRIPT FINISHED\n")

##########################################################################################

#plt.show()