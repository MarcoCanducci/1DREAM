#ifndef ANTPCA_H
#define ANTPCA_H

#define _GLOBAL_SEED_ 0 	//Used in random variables
#define _PI_ 3.1415926535897931f

#define _SUCCESS_ 0 /* integer returned after successful call of a function */
#define _FAILURE_ 1 /* integer returnd after failure in a function */

#include <vector>
#include <array>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <random>
#include <time.h>

#include <iterator>
#include <omp.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

//#include "Eigen/Core"
#include "Eigen/SVD"

#include "Eigen/Dense"
//#include <Eigen/Eigenvalues> 

#include <list>


#include <cfloat>
using namespace std;

////////////////////////////////////////////////////////////////////////////////
///////////////////	// CODE PARAMETERS ////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////


const float lowerlimit = 0.000001f;
const float upperlimit = 5.0f;
const float final_mean_pheromone_expected_per_particle = 1.0f;
const float evapRate = 0.1f;
const float beta_antmovement = 5.0f;
const float beta_antinitialization = 5.0f;
const int remove_ant_in_consecutive_steps_failed_in_pheromone_delivery = 100;


const size_t sorted_removed_type = 0;  //0 = major, 1 = random
const float percentage_removed = 30.0f;
const size_t option_score = 1;	// pp = parallel preference, op = orthogonal preference; 
										//Gr = Grasmann distance; Bh = bhattacharyya distance  
										// Options: 0 = pp Gr, 1 = op Gr, 2 = pp Bh, 3 = op Bh
										
										//pp -> Exp[-2.0 * log(K) * (9 - s^2)/9 ] / normalization
										//op -> Exp[-2.0 * log(K) * s^2] / normalization

	// Note Hellinger distance = sqrt(1 - BC); BC = exp(-D_bhattacharyya)
	// => Hellinger distance = sqrt(1 - exp(-D_bhattacharyya)) \in	[0,1)
	const size_t option_coefficientK = 1; // 0 = Log(K); 1 = Log(K_i);  2 = 1.0;
																	// 3 = 1.0/Log(K_i) ; 4 = 1.0/Log(K);

	const float beta_ph = 5.0f;

	const size_t pheromone_type = 1;  // 0 = constant, 1 = follow the alignmenet


////////////////////////////////////////////////////////////////////////////////
///////////////////	RANDOM VARIABLES ////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

const unsigned int random_deterministic = 0; //0 No activated, 1 = activated 
// Deterministic only work for 1 thread simulation becuase of the pragma reductions
extern mt19937 GLOBAL_GEN_RAND;
extern uniform_real_distribution<float> UNIFORM_DISTRIBUTION_RANDOM_GENERATOR_FLOAT;
extern uniform_real_distribution<double> UNIFORM_DISTRIBUTION_RANDOM_GENERATOR_DOUBLE;





////////////////////////////////////////////////////////////////////////////////
// main function implementing the LAAT algorithm
////////////////////////////////////////////////////////////////////////////////

size_t reading_data(int argc, char *argv[], vector<vector<float>> &data, 
	size_t &numberOfAnts,
  size_t &numberOfIterations,
  size_t &numberOfSteps,
	size_t &pso_number_particles,
	float &pso_min_radii,
	float &pso_max_radii,
	size_t &dynamic_radius_actived,
	size_t &th_neighb,
	float &kappa,
	float &gamma,
	size_t &initialization_mode,
  size_t &numberofthreads,
	string &output_file_address);

size_t exporting_data(vector<float> &pheromone, string &ouptut_folder_address);


std::vector<float> LocallyAlignedAntTechnique(
  std::vector<std::vector<float>> const &data,
  size_t numberOfAnts,
  size_t numberOfIterations,
  size_t numberOfSteps,
  size_t pso_number_particles,
  float pso_min_radii, 
  float pso_max_radii,
  size_t dynamic_radius_actived,
  size_t th_neighb,
  float kappa,
  float gamma,
  std::vector<float> const &external_weights,
  size_t initialization_mode,
  std::vector<size_t> const &custom_init_indices,
  size_t numberofthreads);

// preprocessing functions
void preprocess(std::vector<std::vector<float>> const &data,
		  size_t th_neighb, 
		  float pso_min_radii,
		  float pso_max_radii,
		  size_t pso_number_particles,
		  std::vector<std::vector<size_t>> &pso_neigbourhoods_number,
		  std::vector<std::vector<size_t>> &neighbourhoods,
		  std::vector<std::vector<float>> &neighbourhoods_distances,
		  std::vector<std::vector<std::vector<float>>> &eigenVectors,
		  std::vector<std::vector<float>> &eigenValues,
		  std::vector<std::vector<std::vector<std::vector<float>>>> &pso_eigenVectors,
		  std::vector<std::vector<std::vector<float>>> &pso_eigenValues,
		  std::vector<pair<float,size_t>> &probability_std,
		  std::vector<size_t> &interesting_particle,
		  std::vector<std::vector<float>> &preferences,
		  std::vector<std::vector<float>> &pso_radii_accumulated_probabilities,
      std::vector<std::vector<float>> &pso_radii_probabilities,
		  size_t dynamic_radius_actived,
		  std::vector<std::vector<float>> &quality_pheromone);

void rangeSearch(std::vector<std::vector<float>> const &data,
		  size_t th_neighb,
		  float pso_max_radii,
		   std::vector<std::vector<size_t>> &neighbourhoods,
		   std::vector<std::vector<float>> &neighbourhoods_distances
		   );

void computing_initial_ant_probabilities(std::vector<std::vector<float>> const &data,
		  size_t th_neighb,
		  float pso_max_radii,
		  std::vector<std::vector<size_t>> &neighbourhoods,
		  std::vector<pair<float,size_t>> &probability_std
		  );

void computing_initial_particle_properties(std::vector<std::vector<float>> const &data,
      size_t th_neighb, 
			std::vector<std::vector<size_t>> &neighbourhoods,
      std::vector<std::vector<float>> &neighbourhoods_distances,
      float pso_min_radii,
      float pso_max_radii,
			size_t pso_number_particles,
			std::vector<std::vector<size_t>> &pso_neigbourhoods_number,
			std::vector<std::vector<std::vector<std::vector<float>>>> &pso_eigenVectors,
			std::vector<std::vector<std::vector<float>>> &pso_eigenValues,
			std::vector<std::vector<float>> &pso_radii_accumulated_probabilities,
      std::vector<std::vector<float>> &pso_radii_probabilities,
			std::vector<size_t> &interesting_particle);

// iterative functions
void initializeAnts(std::vector<size_t> &antLocations,
			std::vector<pair<float,size_t>> &probability_std,
			size_t idx_epoch,
			size_t initialization_mode,
			std::vector<size_t> const &custom_init_indices);
			
void antSearch(std::vector<std::vector<float>> const &data,
	      std::vector<std::vector<size_t>> const &neighbourhoods,
	      std::vector<size_t> const &antLocations,
	      size_t numberOfSteps,
	      float kappa,
	      float gamma,
  	    std::vector<float> &pheromone,
		   	float pheromone_delivered,
		   	std::vector<size_t> &interesting_particle,
		   	std::vector<std::vector<float>> &preferences,
		   	std::vector<std::vector<float>> &quality_pheromone,
		   	std::vector<float> const &external_weights,
				size_t idx_epoch);

void antsearch_DynamicRadius(std::vector<std::vector<float>> const &data,
	      			std::vector<std::vector<size_t>> const &neighbourhoods,
	      			std::vector<size_t> &antLocations,
	      			size_t numberOfSteps,
	      			float kappa,
	      			float gamma,
							std::vector<float> &pheromone,
	      			float pheromone_delivered,
        			std::vector<size_t> &interesting_particle,
        			std::vector<std::vector<std::vector<std::vector<float>>>> &pso_eigenVectors,
        			std::vector<std::vector<std::vector<float>>> &pso_eigenValues,
        			std::vector<std::vector<size_t>> &pso_neigbourhoods_number,
	          std::vector<std::vector<float>> &pso_radii_accumulated_probabilities,
							size_t pso_number_particles,
							std::vector<float> const &external_weights,
							size_t idx_epoch);void evaporatePheromone(std::vector<float> &pheromone);

// functions to communicate with the user
void initializeProgressBar(size_t size);
void updateProgressBar(size_t loop);
void completeProgressBar();

float floatRand(size_t a);
size_t sizetRand(size_t a, size_t max_value);

// Markov Chain approximation of LAAT
std::vector<float> LocallyAlignedAntTechnique_MarkovChain(
    std::vector<std::vector<float>> const &data,
    size_t th_neighb,
    float neighbdradii,
    float kappa,
    float gamma,
    std::vector<float> const &external_weights,
    float tolerance,
    size_t max_iterations,
    size_t numberofthreads);

#endif
