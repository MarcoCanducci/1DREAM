#include "LAAT.h"

int main(int argc, char *argv[])
{
	//USER VARIABLES
	vector<vector<float>> data;

	size_t numberOfAnts;
	size_t numberOfIterations;
	size_t numberOfSteps;

	size_t pso_number_particles;
	float pso_min_radii;
	float pso_max_radii;
	size_t dynamic_radius_actived;

	size_t th_neighb;
	float kappa;
	float gamma;
	size_t initialization_mode;

	size_t numberofthreads;


	//READING DATA
	printf("\n\nREADING DATA...\n\n");
	string output_file_address;
	if (reading_data(	argc, argv, data, numberOfAnts, numberOfIterations, numberOfSteps, 
										pso_number_particles, pso_min_radii, pso_max_radii, dynamic_radius_actived, th_neighb, 
										kappa, gamma, initialization_mode, numberofthreads, output_file_address) == _FAILURE_)
  {
    printf("\nError running reading_data function\n");
    return _FAILURE_;
  }
	printf("\n\nREADING DATA FINALIZED SUCCESSFULLY\n\n");

	// Initialize external_weights with default values (all ones for neutral weighting)
	vector<float> external_weights(data.size(), 1.0f);
	
	// Initialize custom_init_indices as empty (will use standard initialization by default)
	vector<size_t> custom_init_indices;

	//LAAT
	printf("\n\nPERFORMING LAAT ...\n\n");
	vector<float> pheromone(data.size(), lowerlimit);
	pheromone = LocallyAlignedAntTechnique(	data, numberOfAnts, numberOfIterations, numberOfSteps,
																					pso_number_particles, pso_min_radii, pso_max_radii, 
																					dynamic_radius_actived, th_neighb, kappa, gamma, external_weights, 
																					initialization_mode, custom_init_indices, numberofthreads);
	printf("\n\nLAAT FINALIZED SUCCESSFULLY\n\n");

	//EXPORTING DATA
	printf("\n\nEXPORTING DATA ...\n\n");
	if (exporting_data(pheromone,output_file_address) == _FAILURE_)
  {
    printf("\n\n Error running exporting_data function\n\n");
    return _FAILURE_;
  }
	printf("\n\nEXPORTING DATA FINALIZED SUCCESSFULLY\n\n");

	printf("\n\nCODE FINALIZED SUCCESSFULLY\n\n\n");

	return _SUCCESS_;
}
