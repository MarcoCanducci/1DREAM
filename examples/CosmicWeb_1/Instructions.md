############# DATA ##################


This three-dimensional data set consists of a cosmic web cube sample of 50^3 Mpc with 281595 data points. The file containing the data is called "CosmicWeb_1.csv" and it is localized in the "CosmicWeb_1" folder (here).


############# NOTES ##################


We run LAAT using few ants, steps and epochs. Better results in the LAAT output can be obtained increasing those parameters at the cost of increasing execution time.

If you have time or/and a good machine, you could see what happend using the following parameters:

"
LAAT_numberOfAnts = 7**3
LAAT_numberOfIterations = 100
LAAT_numberOfSteps = 12000
LAAT_numberofthreads = N # Here N is the number of cores that will be used
"

and changing the "pheromone_threshold" paramter in the "pheromone_threshold.py" file.


############# RUNNING 1DREAM ##################


To run this example you can follow the next steps.

1.- In the "LAAT" folder, run the script "run_LAAT.py" for Linux, or run the "WINDOWS_LAAT.py" for WINDOWS if you already created the LAAT.exe file. It will generate in the "Output" folder a file called "LAAT_output_pheromone.csv". This file will contain the pheromone level of every particle at the end of LAAT.

2.- To filter the data, you can run the script "pheromone_thresholding.py" localized in the "LAAT" directory. It will generate 2 files. One will contain the parameters used and a resume of the results, while the other will contain the data filtered using the pheromone of LAAT. This last file will be called "LAAT_output_selected_data.csv".

3.- Now you can run MBMS using the file "LAAT_output_selected_data.csv". To perform this, in the folder "MBMS" you can run the script "run_MBMS.py". It will generate a file called "MBMS_output.csv" (in the "Output" folder), which will contain the particles displaced towards the centers of the structures in the data, according to the MBMS method.

4.- Now you can run DimIndex. Go to the folder DimIndex, and run the script "run_DimIndex.py". It .py file will generate two .csv files, namely, the dimensionality indexes and the data filtered by those indexes. Index 0 corresponds to 1D structures, index 1 to 2D structures, and index 2 to 3D structures. We only choose the particles with an index equal to 0, i.e. 1D structures. The output file is localized in the "Output" folder and is named "Selected_data_after_DimIndex.csv".

5.- Now you can run Crawling. Go to the folder "Crawling", and run the script "run_Crawling.py". Crawling uses random parameters, so the output is variable. It generates two groups of files, both inside of the "Output" folder. The first group of files is called "Nodeposx.csv" (x is a number from 1 to N, where N is the number of 1D structures in your data). Here you can find the positions of the nodes constructing the graphs (i.e., skeletons or a group of nodes connected together by edges) that model your structures. The second group of files is called "Subsetx.csv" which is the collection of subsets of data points surrounding your graphs. Therefore, there are as many files as skeletons found by Crawling. We also export all output of Crawling in a binary file "Crawling_output.pkl" so that all properties of the graphs (like the indices of their nodes, their adjacency matrix, lengths of the edges... etc) can be reused later. 


6.- SGTM produce some errors for some filaments. We are working to fix those.


############# VISUALIZATION ##################

For matplotlib use at least version 3.2.1, previous versions might not work.

Similar to the previous steps, we can run the plot files inside of every tool folder. So for example, to run the visualization of DimIndex, just go to the "DimIndex" folder, and run the file "DimIndex_plot.py" (obviously after having obtained the output of "run_DimIndex.py"). The output of this image is shown on the screen and also is stored inside the "Images" folder.



