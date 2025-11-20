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

example_name = "Synthetic_Manifold_2"

data = pd.read_csv(example_name + '.csv', header=None).to_numpy()

total_data_points = len(data)
manifold_points = 80000
noise_points = total_data_points - manifold_points

##########################################################################################
##########################################################################################
######################################  PLOTTING  ########################################
##########################################################################################
##########################################################################################

print("\nPLOTTING RAW DATA OF THE EXAMPLE " + example_name.upper() )

fig = plt.gcf()
fig.set_size_inches(16, 10)

#Manifold isolated
ax = fig.add_subplot(121, projection='3d')
sc = ax.scatter(data[0:manifold_points,0],data[0:manifold_points,1],data[0:manifold_points,2], s=0.1)
ax.view_init(44, 125)
ax.set_title("Manifold = " + str(manifold_points))
ax.set_xlabel('X-axis'  , labelpad=10)
ax.set_ylabel('Y-axis'  , labelpad=10)
ax.set_zlabel('Z-axis'  , labelpad=10)
ax.set_xlim(min(data[:,0]), max(data[:,0]))
ax.set_ylim(min(data[:,1]), max(data[:,1]))
ax.set_zlim(min(data[:,2]), max(data[:,2]))

#Total Data
ax = fig.add_subplot(122, projection='3d')
sc = ax.scatter(data[:,0],data[:,1],data[:,2], s=0.001)
ax.view_init(44, 125)
ax.set_title('Total Data: ' + str(total_data_points) + '\n Manifold = '  
                            + str(manifold_points/total_data_points*100)[0:4] + '%; Noise = ' 
                            + str(noise_points/total_data_points*100)[0:4] + '%')
ax.set_xlabel('X-axis'  , labelpad=10)
ax.set_ylabel('Y-axis'  , labelpad=10)
ax.set_zlabel('Z-axis'  , labelpad=10)

plt.tight_layout(pad=4.0, w_pad=4.0, h_pad=0.0)

##########################################################################################
##########################################################################################
######################################  EXPORTING  #######################################
##########################################################################################
##########################################################################################

print("\nEXPORTING THE PLOT")

plt.savefig("Images/1.-raw_data_" + example_name + ".png")

print("\nSCRIPT FINISHED\n")

##########################################################################################

#plt.show()