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

example_name = "CosmicWeb_1"

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
colors_pallette = np.array(getDistinctColors(9))
subplot_rows = np.ceil((number_of_subsets+1)/subplot_columns).astype(int)


CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

tableau20 = ['#4E79A7','#A0CBE8','#F28E2B','#FFBE7D',
             '#59A14F','#8CD17D','#B6992D','#F1CE63',
             '#499894','#86BCB6','#E15759','#FF9D9A',
             '#79706E','#BAB0AC','#D37295','#FABFD2',
             '#B07AA1','#D4A6C8','#9D7660','#D7B5A6']

#plot boundaries
boundaries_min_x = np.min([np.min(np.transpose(Subsets[j])[0]) for j in range(number_of_subsets)]) 
boundaries_min_y = np.min([np.min(np.transpose(Subsets[j])[1]) for j in range(number_of_subsets)]) 
boundaries_min_z = np.min([np.min(np.transpose(Subsets[j])[2]) for j in range(number_of_subsets)]) 
boundaries_max_x = np.max([np.max(np.transpose(Subsets[j])[0]) for j in range(number_of_subsets)]) 
boundaries_max_y = np.max([np.max(np.transpose(Subsets[j])[1]) for j in range(number_of_subsets)]) 
boundaries_max_z = np.max([np.max(np.transpose(Subsets[j])[2]) for j in range(number_of_subsets)]) 

#Plot

for plot_idx in range(int((number_of_subsets+14)/15)):

    fig = plt.gcf()
    fig.set_size_inches(20, 20)

    #First plot containing all the filaments
    ax = fig.add_subplot(4,4,1, projection='3d')
    counter = 5
    for filament_idx in range(plot_idx * 15, np.min( [(plot_idx+1) * 15 , number_of_subsets ] )):
    #for filament_idx in range(1):

        #sc = ax.scatter(np.transpose(Subsets[filament_idx])[0],np.transpose(Subsets[filament_idx])[1],np.transpose(Subsets[filament_idx])[2], s=0.1, c = [colors_pallette[filament_idx%9]],zorder=1)
        pos = AllNodeNames(FG[filament_idx], 3)
        xi = pos[:, 0]
        yi = pos[:, 1]
        zi = pos[:, 2]
        sc = ax.scatter(xi, yi, zi , c = [tableau20[filament_idx%15]], s = 2.0)

        for tup in FG[filament_idx].edges():
            t1 = np.frombuffer(FG[filament_idx].nodes[tup[0]]['Name'])
            t2 = np.frombuffer(FG[filament_idx].nodes[tup[1]]['Name'])
            x = np.array((t1[0], t2[0]))
            y = np.array((t1[1], t2[1]))
            z = np.array((t1[2], t2[2]))
            ax.plot(x, y, z,c = "black", linewidth=0.5)

    ax.view_init(44, 125)
    ax.set_title("Filaments " + str(plot_idx * 15+1) + " to " + str(np.min( [(plot_idx+1) * 15 , number_of_subsets ])) + "\nfounded with Crawling")
    ax.set_xlabel('X-axis [Mpc]'  , labelpad=10)
    ax.set_ylabel('Y-axis [Mpc]'  , labelpad=10)
    ax.set_zlabel('Z-axis [Mpc]'  , labelpad=10)
    ax.set_xlim(boundaries_min_x,boundaries_max_x)
    ax.set_ylim(boundaries_min_y,boundaries_max_y)
    ax.set_zlim(boundaries_min_z,boundaries_max_z)


    #Filaments
    aux_idx = 1
    for filament_idx in range(plot_idx * 15, np.min( [(plot_idx+1) * 15 , number_of_subsets ] )):
        aux_idx = aux_idx + 1
        ax = fig.add_subplot(4,4,aux_idx, projection='3d')
        # sc = ax.scatter(np.transpose(Subsets[filament_idx])[0],np.transpose(Subsets[filament_idx])[1],np.transpose(Subsets[filament_idx])[2], s=0.1,c = [tableau20[filament_idx%9]],zorder=1)
        pos = AllNodeNames(FG[filament_idx], 3)
        xi = pos[:, 0]
        yi = pos[:, 1]
        zi = pos[:, 2]
        sc = ax.scatter(xi, yi, zi , c = [tableau20[filament_idx%15]], s = 2.0)

        for tup in FG[filament_idx].edges():
            t1 = np.frombuffer(FG[filament_idx].nodes[tup[0]]['Name'])
            t2 = np.frombuffer(FG[filament_idx].nodes[tup[1]]['Name'])
            x = np.array((t1[0], t2[0]))
            y = np.array((t1[1], t2[1]))
            z = np.array((t1[2], t2[2]))
            #ax.plot(x, y, z, c='black',linewidth=1,zorder=counter,marker='D')
            ax.plot(x, y, z,c = "black", linewidth=0.5)

        ax.view_init(44, 125)
        ax.set_xlabel('X-axis [Mpc]'  , labelpad=10)
        ax.set_ylabel('Y-axis [Mpc]'  , labelpad=10)
        ax.set_zlabel('Z-axis [Mpc]'  , labelpad=10)
        ax.set_xlim(boundaries_min_x,boundaries_max_x)
        ax.set_ylim(boundaries_min_y,boundaries_max_y)
        ax.set_zlim(boundaries_min_z,boundaries_max_z)
        ax.set_title("Filament " + str(filament_idx+1))


    print("\nEXPORTING THE PLOT ",plot_idx + 1)

    plt.savefig('../Images/5.-Crawling_output_set_' + str(plot_idx + 1) + '.png')


    plt.close(fig)





#Ploting full filaments
    
fig = plt.gcf()
fig.set_size_inches(20, 20)

ax = fig.add_subplot(1,1,1, projection='3d')
for filament_idx in range(number_of_subsets):
#for filament_idx in range(1):
    #sc = ax.scatter(np.transpose(Subsets[filament_idx])[0],np.transpose(Subsets[filament_idx])[1],np.transpose(Subsets[filament_idx])[2], s=0.1, c = [colors_pallette[filament_idx%9]],zorder=1)
    pos = AllNodeNames(FG[filament_idx], 3)
    xi = pos[:, 0]
    yi = pos[:, 1]
    zi = pos[:, 2]
    #sc = ax.scatter(xi, yi, zi , c = [tableau20[filament_idx%9]])
    sc = ax.scatter(xi, yi, zi)
    for tup in FG[filament_idx].edges():
        t1 = np.frombuffer(FG[filament_idx].nodes[tup[0]]['Name'])
        t2 = np.frombuffer(FG[filament_idx].nodes[tup[1]]['Name'])
        x = np.array((t1[0], t2[0]))
        y = np.array((t1[1], t2[1]))
        z = np.array((t1[2], t2[2]))
        #ax.plot(x, y, z, c='black',linewidth=0.5,zorder=counter,marker='D')
        ax.plot(x, y, z,c = "black", linewidth=0.5)


ax.view_init(44, 125)
ax.set_title("Filaments = " + str(number_of_subsets) + "\nfounded with Crawling")
ax.set_xlabel('X-axis [Mpc]'  , labelpad=10)
ax.set_ylabel('Y-axis [Mpc]'  , labelpad=10)
ax.set_zlabel('Z-axis [Mpc]'  , labelpad=10)
ax.set_xlim(boundaries_min_x,boundaries_max_x)
ax.set_ylim(boundaries_min_y,boundaries_max_y)
ax.set_zlim(boundaries_min_z,boundaries_max_z)



print("\nEXPORTING THE PLOT with full filaments ")

plt.savefig('../Images/5.-Crawling_output_full_filaments.png')

plt.close(fig)




print("\nSCRIPT FINISHED\n")