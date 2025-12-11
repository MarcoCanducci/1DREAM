#include "LAAT.h"

#include "nanoflann.hpp"
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
 * @param gamma tuning parameter for the relative importance of the
 *   external criterion (external_weights). The final weights are:
 *   lambda1 = (1-kappa)*(1-gamma), lambda2 = kappa*(1-gamma), lambda3 = gamma
 * @param pheromone vector containing the pheromone of each data point, will
 *   be updated by this function
 * @param external_weights vector containing per-point external scalar values
 *   used as a third criterion in the ant jump probability calculation
 */

#pragma omp declare reduction(vec_float_plus : std::vector<float> : \
                              std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<float>())) \
                    initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))

void antSearch(vector<vector<float>> const &data,
			   vector<vector<size_t>> const &neighbourhoods,
			   vector<size_t> const &antLocations,
			   size_t numberOfSteps,
			   float kappa,
			   float gamma,
			   vector<float> &pheromone,
			   float pheromone_delivered,
			   vector<size_t> &interesting_particle,
			   vector<vector<float>> &preferences,
			   vector<vector<float>> &quality_pheromone,
			   vector<float> const &external_weights,
				 size_t idx_epoch)
{
	vector<float> accumulatedPheromone(data.size(),0.0f);

	// Check if external weights are provided
	bool use_external_weights = !external_weights.empty() && (gamma > 0.0f);

	// Computing neighbourhoods pheromones and external weights sums for normalization
	vector<float> neighbourhoods_pheromone(data.size());
	vector<float> neighbourhoods_external_weights_sum;
	if (use_external_weights) {
		neighbourhoods_external_weights_sum.resize(data.size());
	}
	float aux_sum_pheromone;
	float aux_sum_external_weights;

	#pragma omp parallel for private(aux_sum_pheromone, aux_sum_external_weights) schedule(dynamic,10)
	for (long long idx_particle = 0; idx_particle < static_cast<long long>(data.size()); idx_particle++)
	{
		aux_sum_pheromone = 0.0f;
		aux_sum_external_weights = 0.0f;
		vector<size_t> const &neighbourhood = neighbourhoods[idx_particle];
		for (size_t idx_neighbour = 0; idx_neighbour < neighbourhood.size(); idx_neighbour++)
		{
			aux_sum_pheromone += pheromone[neighbourhood[idx_neighbour]];
			if (use_external_weights) {
				aux_sum_external_weights += external_weights[neighbourhood[idx_neighbour]];
			}
		}
		neighbourhoods_pheromone[idx_particle] = aux_sum_pheromone;
		// Avoid division by zero - if all external weights are zero, set sum to 1
		if (use_external_weights) {
			neighbourhoods_external_weights_sum[idx_particle] = (aux_sum_external_weights > 0.0f) ? aux_sum_external_weights : 1.0f;
		}
	}

	// Pre computation of accumulated probabilities
	float ant_jump_probabilities;
	vector<vector<float>> accumulated_ant_jump_probabilities(data.size());
	float normalizedPheromone;
	float normalizedExternalWeight;
	float lambda1, lambda2, lambda3;

	// Compute lambda weights based on whether external weights are used
	// If external weights not provided: lambda1 = (1-kappa), lambda2 = kappa, lambda3 = 0
	// If external weights provided: lambda1 = (1-kappa)*(1-gamma), lambda2 = kappa*(1-gamma), lambda3 = gamma
	float effective_gamma = use_external_weights ? gamma : 0.0f;
 
	#pragma omp parallel for private(normalizedPheromone,normalizedExternalWeight,lambda1,lambda2,lambda3,ant_jump_probabilities) schedule(dynamic,10)
	for (long long particle_idx = 0; particle_idx < static_cast<long long>(data.size()); particle_idx++)
	{
		if (interesting_particle[particle_idx] == 1)
		{
			vector<size_t> const &neighbourhood = neighbourhoods[particle_idx];
			accumulated_ant_jump_probabilities[particle_idx].resize(neighbourhood.size());

			lambda1 = (1.0f - kappa) * (1.0f - effective_gamma);
			lambda2 = kappa * (1.0f - effective_gamma);
			lambda3 = effective_gamma;

			normalizedPheromone = pheromone[neighbourhood[0]] / neighbourhoods_pheromone[particle_idx];
			if (use_external_weights) {
				normalizedExternalWeight = external_weights[neighbourhood[0]] / neighbourhoods_external_weights_sum[particle_idx];
				ant_jump_probabilities = expf(beta_antmovement * (lambda1 * normalizedPheromone + lambda2 * preferences[particle_idx][0] + lambda3 * normalizedExternalWeight));
			} else {
				ant_jump_probabilities = expf(beta_antmovement * (lambda1 * normalizedPheromone + lambda2 * preferences[particle_idx][0]));
			}
			accumulated_ant_jump_probabilities[particle_idx][0] = ant_jump_probabilities;

			for (size_t neighbour_idx = 1; neighbour_idx < neighbourhood.size(); neighbour_idx++)
			{
				normalizedPheromone = pheromone[neighbourhood[neighbour_idx]] / neighbourhoods_pheromone[particle_idx];
				if (use_external_weights) {
					normalizedExternalWeight = external_weights[neighbourhood[neighbour_idx]] / neighbourhoods_external_weights_sum[particle_idx];
					ant_jump_probabilities = expf(beta_antmovement * (lambda1 * normalizedPheromone + lambda2 * preferences[particle_idx][neighbour_idx] + lambda3 * normalizedExternalWeight));
				} else {
					ant_jump_probabilities = expf(beta_antmovement * (lambda1 * normalizedPheromone + lambda2 * preferences[particle_idx][neighbour_idx]));
				}
				accumulated_ant_jump_probabilities[particle_idx][neighbour_idx] = accumulated_ant_jump_probabilities[particle_idx][neighbour_idx - 1] + ant_jump_probabilities;
			}

			// Normalization
			for (size_t neighbour_idx = 0; neighbour_idx < neighbourhood.size(); neighbour_idx++)
			{
				accumulated_ant_jump_probabilities[particle_idx][neighbour_idx] = accumulated_ant_jump_probabilities[particle_idx][neighbour_idx] / accumulated_ant_jump_probabilities[particle_idx][neighbourhood.size() - 1];
			}
		}
	}

	//Ant movement
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

		#pragma omp for reduction(vec_float_plus : accumulatedPheromone)
		for (size_t ant_idx = 0; ant_idx < antLocations.size(); ant_idx++)
		{
			myseed2 = myseed1 + ant_idx * numberOfSteps;

			current_ant_position = antLocations[ant_idx];

			for (size_t step_idx = 0; step_idx < numberOfSteps; step_idx++)
			{
				myseed3 = myseed2 + step_idx;
				/*
				From current_ant_position point, select the next point from it's neighbourhood
				with probabilities as defined in formula (8).
				*/
				check_next = false;
				counter_failed_deliver_pheromone = 0;

				while (check_next == false)
				{
					vector<size_t> const &neighbourhood = neighbourhoods[current_ant_position];
					previous_ant_position = current_ant_position;

					myseed3++;

					if (interesting_particle[current_ant_position] == 1)
					{ 
						target_probability = floatRand(myseed3);

						search_idx = (size_t)(target_probability * (neighbourhood.size() - 1));

						if (accumulated_ant_jump_probabilities[current_ant_position][search_idx] < target_probability)
						{
							for (size_t aux_idx = search_idx; aux_idx < neighbourhood.size() - 1; aux_idx++)
							{
								search_idx++;
								if (accumulated_ant_jump_probabilities[current_ant_position][search_idx] >= target_probability)
								{
									aux_idx = neighbourhood.size() - 2;
								}
							}
						}
						else if (accumulated_ant_jump_probabilities[current_ant_position][search_idx] > target_probability)
						{
							for (size_t aux_idx = search_idx; aux_idx > 0; aux_idx--)
							{
								search_idx--;
								if (accumulated_ant_jump_probabilities[current_ant_position][search_idx] <= target_probability)
								{
									aux_idx = 1;
									search_idx++;
								}
							}
						}
					}
					else
					{
						search_idx = sizetRand(myseed3, neighbourhood.size()) ;
					}

					// Ant new position
					current_ant_position = neighbourhood[search_idx];

					// keep track of pheromone to be added
					if (interesting_particle[current_ant_position] == 1 && interesting_particle[previous_ant_position] == 1)
					{
						if(pheromone_type == 0)
						{
							accumulatedPheromone[current_ant_position] += 1.0f;
						}
						else if(pheromone_type == 1)
						{
							accumulatedPheromone[current_ant_position] += 1.0f * quality_pheromone[previous_ant_position][search_idx];
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
	for (long long idx = 0; idx < static_cast<long long>(data.size()); ++idx)
	{
		if (accumulatedPheromone[idx])
		{
			pheromone[idx] += pheromone_delivered * accumulatedPheromone[idx];
		}
	}

	// Free memory
	vector<float> ().swap(accumulatedPheromone);
	vector<float>().swap(neighbourhoods_pheromone);
	vector<float>().swap(neighbourhoods_external_weights_sum);
}