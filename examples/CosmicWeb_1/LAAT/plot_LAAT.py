##########################################################################################
##########################################################################################
######################################  LIBRARIES  #######################################
##########################################################################################
##########################################################################################

import pandas as pd
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

example_name = "CosmicWeb_1"

data = pd.read_csv("../" + example_name +  ".csv", header=None).to_numpy()
pheromone = pd.read_csv('../Output/LAAT_output_pheromone.csv', header=None).to_numpy()
selected_data =   pd.read_csv('../Output/LAAT_output_selected_data.csv', header=None).to_numpy()

# Data points for this example
total_data_points = len(data)

##########################################################################################
##########################################################################################
######################################  PLOTTING  ########################################
##########################################################################################
##########################################################################################

print("\nPLOTTING LAAT OUTPUT OF THE EXAMPLE " + example_name.upper() )

fig = plt.gcf()
fig.set_size_inches(16, 10)

#Pheromone
ax = fig.add_subplot(121, projection='3d')
cm = plt.colormaps['winter_r']
sc = ax.scatter(data[:,0], data[:,1], data[:,2], s=0.001, c=pheromone[:], cmap=cm)
ax.view_init(44, 125)
ax.set_title('Pheromone distribution')
plt.colorbar(sc,pad=0.1)
ax.set_xlabel('X-axis [Mpc]'  , labelpad=10)
ax.set_ylabel('Y-axis [Mpc]'  , labelpad=10)
ax.set_zlabel('Z-axis [Mpc]'  , labelpad=10)

#Selected Data
ax = fig.add_subplot(122, projection='3d')
sc = ax.scatter(selected_data[:,0], selected_data[:,1], selected_data[:,2], s=0.01)
ax.view_init(44, 125)
ax.set_title("Seletected data = " + str(len(selected_data)) )
ax.set_xlabel('X-axis [Mpc]'  , labelpad=10)
ax.set_ylabel('Y-axis [Mpc]'  , labelpad=10)
ax.set_zlabel('Z-axis [Mpc]'  , labelpad=10)
ax.set_xlim(min(data[:,0]),max(data[:,0]))
ax.set_ylim(min(data[:,1]),max(data[:,1]))
ax.set_zlim(min(data[:,2]),max(data[:,2]))

plt.tight_layout(pad=4.0, w_pad=4.0, h_pad=0.0)

##########################################################################################
##########################################################################################
######################################  EXPORTING  #######################################
##########################################################################################
##########################################################################################

print("\nEXPORTING THE PLOT")

plt.savefig('../Images/2.-selected_data_after_LAAT.png')

print("\nSCRIPT FINISHED\n")

##########################################################################################

#plt.show()
