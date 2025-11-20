##########################################################################################
##########################################################################################
######################################  LIBRARIES  #######################################
##########################################################################################
##########################################################################################

import pandas as pd
import numpy as np



##########################################################################################
##########################################################################################
######################################  INPUT DATA  ######################################
##########################################################################################
##########################################################################################

data = pd.read_csv('../Synthetic_Manifold_2.csv', header=None).to_numpy()
pheromone = np.loadtxt('../Output/LAAT_output_pheromone.csv', delimiter=',')
  

# Data points for this example
total_data_points = len(data)
manifold_points = 80000
noise_points = total_data_points - manifold_points

##########################################################################################
##########################################################################################
#####################################  THRESHOLDING  #####################################
##########################################################################################
##########################################################################################

# Here put your threhold by percentage
pheromone_threshold_percentage = 0.15
#pheromone_threshold_percentage = 0.1 if you activated Dynamic radius.
sorted_pheromone = np.sort(-pheromone)
pheromone_threshold = -sorted_pheromone[int(pheromone_threshold_percentage * len(pheromone))]

# Here put your threhold
#pheromone_threshold = 1.5 * np.mean(pheromone)
#Selecting the data
selectedData = data[pheromone>pheromone_threshold,:]




##########################################################################################
##########################################################################################
#######################################  STATISTICS  #####################################
##########################################################################################
##########################################################################################

def Find_percentages(pheromone):
	'''
	Finds the percentage of the recovered manifold by LAAT in the example dataset. 
	'''
	
	lenths = [manifold_points,noise_points]
	names = ["Manifold","Noise"]

	#Storing Percentages of data saved
	aux_select_idx = 0
	aux = 0
	flags = np.zeros(len(pheromone), dtype=int)
	percentages=np.array([])
	for i in range(2):
	  counter = 0
	  for j in range(lenths[i]):
	    if( pheromone[j + aux] > pheromone_threshold):
	      counter += 1
	      flags[ j + aux] = 1
	  if(lenths[i] == 0):
	    aux_per = 0
	  else:
	    aux_per = 100 * counter/lenths[i]
	  aux +=  lenths[i]
	  percentages=np.append(percentages,aux_per)
  
	  aux_select_idx = lenths[i]

	return percentages	



percentages = Find_percentages(pheromone)

mean_pheromone = np.mean(pheromone)
min_pheromone = np.min(pheromone)
max_pheromone = np.max(pheromone)


print("\n\n INPUT VALUES\n\n")

print("mean pheromone = ",mean_pheromone)
print("minimum pheromone = ",min_pheromone)
print("maximum pheromone = ", max_pheromone)
print("pheromone threshold used = ", pheromone_threshold)






print("\n\n EXAMPLE VALUES\n\n")

print("Manifold points = ",manifold_points)
print("Manifold points = ",noise_points)
print("selected data lengh = ",len(selectedData))
print("%% of the Manifold recovered = ",percentages[0])
print("%% of the Noise recovered = ",percentages[1])


print("\n\n EXPORTING DATA SELECTED \n\n")

np.savetxt("../Output/LAAT_output_selected_data.csv", selectedData, delimiter=',', fmt = '%5.8f')


with open('../Output/LAAT_output_selected_data_info.txt', 'w') as f:
  f.write('mean pheromone = %f\n' % mean_pheromone)
  f.write('minimum pheromone = %f\n' % min_pheromone)
  f.write('maximum pheromone = %f\n' % max_pheromone)
  f.write('pheromone threshold used = %f\n' % pheromone_threshold)
  f.write('Manifold points = %d\n' % manifold_points)
  f.write('Noise points = %d\n' % noise_points)
  f.write('selected data lengh = %d\n' % len(selectedData))
  f.write("%% of the Manifold recovered = %f\n" % percentages[0])
  f.write('%% of the Noise recovered = %f\n' % percentages[1])

