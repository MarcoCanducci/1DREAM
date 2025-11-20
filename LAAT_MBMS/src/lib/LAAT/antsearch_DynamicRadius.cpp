#include "LAAT.h"

#include "nanoflann/nanoflann.hpp"
#include "utils/KDTreeVectorOfVectorsAdaptor.h"

/**
 * Perform ant search for all ants.
 *
 * When the function is done there is an increase of pheromone on data
 * points dependent on how many times they were visited by ants
 *
 * This implementation uses OpenMP to perform ant search in parallel for
 * the different ants. Synchronization in the from of atomic operations
 * is used to ensure thread safety.
 *
 * @param data vector containing the data points to search on
 * @param neighbourhoods vector containing a vector of all nearest
 *   neighbours for each data point
 * @param antLocations vector containing for each ant the initial point
 *   to start searching from
 * @param eigenVectors vector containing the eigen vectors for each data
 *   point
 * @param eigenValues vector containing the eigen values for each data
 *   point
 * @param numberOfSteps number of steps to run each ant at each iteration
 * @param beta_antmovement the inverse temperature as used in formula (8)
 * @param kappa tuning parameter for the relative importance of the
 *   influence of the alignment and pheromone terms as used in formula (7)
 * @param pheromone vector containing the pheromone of each data point, will
 *   be updated by this function
 */


void compute_local_pheromone(vector<size_t> const &neighbourhood, vector<float> &pheromone , vector<float> &local_pheromone )
{
	float aux_sum_pheromone = 0.0f;
	for (size_t idx_neighbour = 0; idx_neighbour < neighbourhood.size(); idx_neighbour++)
	{
		local_pheromone[idx_neighbour] = pheromone[neighbourhood[idx_neighbour]];
		aux_sum_pheromone += pheromone[neighbourhood[idx_neighbour]];
	}

	for (size_t idx_neighbour = 0; idx_neighbour < neighbourhood.size(); idx_neighbour++)
	{
		local_pheromone[idx_neighbour] /= aux_sum_pheromone;
	}
}

void compute_local_preference(vector<vector<float>> const &data, vector<size_t> const &neighbourhood, vector<vector<float>> &eigenVectors, vector<float> &eigenValues, vector<float> &local_prefence , size_t current_point_idx , size_t pheromone_type, vector<float> &local_quality_pheromone)
{
  vector <float> relativeDistances(3);
  vector <float> weights(3);
  float wight_sum;
  float preferences_sum = 0.0f;
  
	for (size_t neighbour_idx = 0; neighbour_idx < neighbourhood.size(); neighbour_idx++)
  {
    for (size_t dim_idx = 0; dim_idx < 3; dim_idx++)
    {
      relativeDistances[dim_idx] = data[neighbourhood[neighbour_idx]][dim_idx] - data[current_point_idx][dim_idx];
    }
    	
		float norm = sqrtf(relativeDistances[0] * relativeDistances[0] +
    						        relativeDistances[1] * relativeDistances[1] +
    						        relativeDistances[2] * relativeDistances[2]);                 
    
		for (size_t dim_idx = 0; dim_idx < 3; dim_idx++)
    {
      relativeDistances[dim_idx] = relativeDistances[dim_idx] / norm;
    }
    
		//Compute alignment between the data point and it's neighbours with the eigen-directions.
    wight_sum = 0.0f;
    for (size_t dim_idx = 0; dim_idx < 3; dim_idx++)
    {
      // matrix multiplication
    	weights[dim_idx] = fabs(
    		relativeDistances[0] * eigenVectors[0][dim_idx] +
    		relativeDistances[1] * eigenVectors[1][dim_idx] +
    		relativeDistances[2] * eigenVectors[2][dim_idx]);
        wight_sum += weights[dim_idx];
    }
    
		// Normalize alignment values to obtain relative weighting of the alignment according to formula (2).
    for (size_t dim_idx = 0; dim_idx < 3; dim_idx++)
    {
    	weights[dim_idx] = weights[dim_idx] / wight_sum;
    }
    
		for (size_t dim_idx = 0; dim_idx < 3; dim_idx++)
    {
    	local_prefence[neighbour_idx] += weights[dim_idx] * eigenValues[dim_idx]; 
    }
	
		if(pheromone_type == 1)
		{
    	local_quality_pheromone[neighbour_idx] = expf(beta_ph * local_prefence[neighbour_idx]) / expf(beta_ph);
		}
	
    preferences_sum += local_prefence[neighbour_idx];
  }

  // Normalizing the preferences.
  for (size_t neighbour_idx = 0; neighbour_idx < neighbourhood.size(); neighbour_idx++)
  {
  	local_prefence[neighbour_idx] = local_prefence[neighbour_idx] / preferences_sum;
  }
}

void compute_accumulated_jump_probability(vector<float> &local_pheromone, vector<float> &local_prefence , vector<float> &local_accum_jump_probability, float kappa )
{
	float ant_jump_probabilities = expf(beta_antmovement * ((1.0f - kappa) * local_pheromone[0] + kappa * local_prefence[0]));
	local_accum_jump_probability[0] = ant_jump_probabilities;


	for (size_t neighbour_idx = 1; neighbour_idx < local_pheromone.size(); neighbour_idx++)
	{
		ant_jump_probabilities = expf(beta_antmovement * ((1.0f - kappa) * neighbour_idx + kappa * local_prefence[neighbour_idx]));
		local_accum_jump_probability[neighbour_idx] = local_accum_jump_probability[neighbour_idx - 1] + ant_jump_probabilities;
	}

	// Normalization
	for (size_t neighbour_idx = 0; neighbour_idx < local_pheromone.size(); neighbour_idx++)
	{
		local_accum_jump_probability[neighbour_idx] /= local_accum_jump_probability[local_pheromone.size() - 1];
	}
}

size_t finding_radius(vector<float> &pso_radii_accumulated_probabilities_local, float target_probability, size_t min_idx, size_t pso_number_particles)
{
	size_t search_idx = (size_t)(target_probability * (pso_number_particles - 1 - min_idx)) + min_idx;
	if (pso_radii_accumulated_probabilities_local[search_idx] < target_probability)
	{
		for (size_t aux_idx = search_idx; aux_idx < pso_number_particles - 1; aux_idx++)
		{
			search_idx++;
			if (pso_radii_accumulated_probabilities_local[search_idx] >= target_probability)
			{
				aux_idx = pso_number_particles - 2;
			}
		}
	}
	else if (pso_radii_accumulated_probabilities_local[search_idx] > target_probability)
	{
		for (size_t aux_idx = search_idx; aux_idx > min_idx; aux_idx--)
		{
			search_idx--;
			if (pso_radii_accumulated_probabilities_local[search_idx] <= target_probability)
			{
				aux_idx = 1;
				search_idx++;
			}
		}
	}

	return search_idx;
}

#pragma omp declare reduction(vec_float_plus : vector<float> : \
                              std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<float>())) \
                    initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))

void antsearch_DynamicRadius(vector<vector<float>> const &data,
	      			vector<vector<size_t>> const &neighbourhoods,
	      			vector<size_t> &antLocations,
	      			size_t numberOfSteps,
	      			float kappa,
							vector<float> &pheromone,
	      			float pheromone_delivered,
        			vector<size_t> &interesting_particle,
        			vector<vector<vector<vector<float>>>> &pso_eigenVectors,
        			vector<vector<vector<float>>> &pso_eigenValues,
        			vector<vector<size_t>> &pso_neigbourhoods_number,
        			vector<vector<float>> &pso_radii_accumulated_probabilities,
							size_t pso_number_particles,
							size_t idx_epoch)
{
	vector<float> accumulatedPheromone(data.size(),0.0f);

	#pragma omp parallel
	{
		size_t myseed1 = _GLOBAL_SEED_ + idx_epoch * antLocations.size();
		size_t myseed2;
		size_t myseed3;

		size_t current_ant_position;
		bool check_next;
		int counter_failed_deliver_pheromone;
		float target_probability;
		size_t search_idx;
		size_t previous_ant_position;
		float quality_pheromone_aux = 0.0f;
		size_t min_idx;

		vector<float> pso_radii_accumulated_probabilities_local(pso_number_particles);

		vector<vector<float>> pso_radii_accumulated_probabilities_antstore_global(data.size());

		#pragma omp for reduction(vec_float_plus : accumulatedPheromone)
		for (size_t ant_idx = 0; ant_idx < antLocations.size(); ant_idx++)
		{
			myseed2 = myseed1 + ant_idx * numberOfSteps;
			current_ant_position = antLocations[ant_idx];

			for (size_t step_idx = 0; step_idx < numberOfSteps; step_idx++)
			{
				myseed3 = myseed2 + step_idx;

				check_next = false;
				counter_failed_deliver_pheromone = 0;

				while (check_next == false)
				{
					previous_ant_position = current_ant_position;
					vector<size_t> neighbourhood = neighbourhoods[current_ant_position];

					myseed3++;

					if(interesting_particle[current_ant_position] < pso_number_particles)
					{
						size_t radii_idx;
						vector<vector<float>> eigenVectors;
						vector<float> eigenValues;
						
						for(size_t aux_idx = 0; aux_idx < pso_number_particles; aux_idx++)
						{
              pso_radii_accumulated_probabilities_local[aux_idx] = pso_radii_accumulated_probabilities[aux_idx][current_ant_position];
						}
						
						target_probability = floatRand(myseed3);
						min_idx = interesting_particle[current_ant_position];
						radii_idx = finding_radius(pso_radii_accumulated_probabilities_local, target_probability, min_idx, pso_number_particles );

						if(radii_idx == pso_number_particles)
						{
							printf("\n\nERROR\n\n");
						}

						//Defining eigenvectors, eigenvalues and resizing the neighbourhood
						eigenVectors = pso_eigenVectors[radii_idx][current_ant_position];
						eigenValues = pso_eigenValues[radii_idx][current_ant_position];
						neighbourhood.resize(pso_neigbourhoods_number[radii_idx][current_ant_position]);

						//COMPUTING THE LOCAL PHEROMONE
						vector <float> local_pheromone(neighbourhood.size());
						compute_local_pheromone(neighbourhood, pheromone , local_pheromone );

						//COMPUTINGN LOCAL PREFERENCE
						vector <float> local_prefence(neighbourhood.size());
						vector <float> local_quality_pheromone;
						if(pheromone_type == 1)
						{
							local_quality_pheromone.resize(neighbourhood.size());
						}
						compute_local_preference(data, neighbourhood, eigenVectors, eigenValues, local_prefence , current_ant_position, pheromone_type ,local_quality_pheromone);

						//COMPUTINGN ACCUM PROBABILITY
						vector <float> local_accum_jump_probability(neighbourhood.size());
						compute_accumulated_jump_probability(local_pheromone, local_prefence , local_accum_jump_probability, kappa );

						myseed3++;

						target_probability = floatRand(myseed3);
						search_idx = (size_t)(target_probability * (neighbourhood.size() - 1 ));

						if (local_accum_jump_probability[search_idx] < target_probability)
						{
							for (size_t aux_idx = search_idx; aux_idx < neighbourhood.size() - 1; aux_idx++)
							{
								search_idx++;
								if (local_accum_jump_probability[search_idx] >= target_probability)
								{
									aux_idx = neighbourhood.size() - 2;
								}
							}
						}
						else if (local_accum_jump_probability[search_idx] > target_probability)
						{
							for (size_t aux_idx = search_idx; aux_idx > 0; aux_idx--)
							{
								search_idx--;
								if (local_accum_jump_probability[search_idx] <= target_probability)
								{
									aux_idx = 1;
									search_idx++;
								}
							}
						}

						if(pheromone_type == 1)
						{
							quality_pheromone_aux = local_quality_pheromone[search_idx];
						}
					}
					else
					{
						search_idx = sizetRand(myseed3, neighbourhood.size()) ;
					}
					
					////////////////////////////////////////////////////////////////////////////////////////////////
					///////////////////////////////// ANT NEW POSITION ////////////////////////////////////////////////
					////////////////////////////////////////////////////////////////////////////////////////////////
					current_ant_position = neighbourhood[search_idx];

					// keep track of pheromone to be added
					if (interesting_particle[current_ant_position] < pso_number_particles && interesting_particle[previous_ant_position] < pso_number_particles)
					{
						//#pragma omp atomic
						if(pheromone_type == 0)
						{
							accumulatedPheromone[current_ant_position] += 1.0f;
						}
						else if(pheromone_type == 1)
						{
							accumulatedPheromone[current_ant_position] += 1.0f * quality_pheromone_aux;
						}

						check_next = true;
					}
					else
					{
						counter_failed_deliver_pheromone += 1;
					}

					if (counter_failed_deliver_pheromone > remove_ant_in_consecutive_steps_failed_in_pheromone_delivery)
					{
						check_next = true;
						step_idx = numberOfSteps;
					}

				}
			}
		}
	} // pragma omp parallel end

	#pragma omp parallel for
	for (size_t idx = 0; idx < data.size(); ++idx)
	{
		if (accumulatedPheromone[idx])
		{
			pheromone[idx] += pheromone_delivered * accumulatedPheromone[idx];
		}
	}

	vector<float> ().swap(accumulatedPheromone);
}
