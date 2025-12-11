#include "LAAT.h"


/**
 * Implementation of the Ant Colony algorithm proposed by Taghribi et al.
 * (2020).
 * See Algorithm 1 in the paper for a pseudocode of this algorithm.
 *
 * In this function and all of the functions evoked by this one we will refer
 * to the formulas as described by Taghribi et al. (2020).
 *
 * Reference:
 * Taghribi, A., Bunte, K., Smith, R., Shin, J., Mastropietro, M., Peletier,
 * R. F., & Tino, P. (2020). LAAT: Locally Aligned Ant Technique for detecting
 * manifolds of varying density. arXiv preprint arXiv:2009.08326.
 *
 * @param data vector containing all the data points to search
 * @param numberOfAnts number of ants
 * @param numberOfIterations number of tiems to apply ant search
 * @param numberOfSteps number of steps to run each ant at each iteration
 * @param th minimum number of nearest neighbours needed for a datapoint
 *   to be considered
 * @param neighbdradii Neighborhood radius within which to search for nearest neighbours
 * @param beta_antmovement the inverse temperature as used in formula (8)
 * @param kappa tuning parameter for the relative importance of the influence
 * of the alignment and pheromone terms as used in formula (7)
 * @param evapRate rate at which to evaporate pheromone after each application
 *   of ant search
 * @param lowerlimit lower limit on the amount of pheromone of a data point
 * @param upperlimit upper limit on the amount of pheromone of a data point
 * @return vector containing the amount of pheromone for each data point after
 *   the application of LAAT
 */

//Random variables
mt19937 GLOBAL_GEN_RAND;
uniform_real_distribution<float> UNIFORM_DISTRIBUTION_RANDOM_GENERATOR_FLOAT;
uniform_real_distribution<double> UNIFORM_DISTRIBUTION_RANDOM_GENERATOR_DOUBLE;

vector<float> LocallyAlignedAntTechnique(
	vector<vector<float>> const &data,
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
	vector<float> const &external_weights,
	size_t initialization_mode,
	vector<size_t> const &custom_init_indices,
	size_t numberofthreads)
{
	cout << endl << endl <<  "Running LAAT, version 1.4.1 (04-04-2025)" << endl << endl;

	printf("\ndynamic_radius_actived = %d\n",(int) dynamic_radius_actived);
	printf("pso_number_particles = %d\n",(int) pso_number_particles);


	if(dynamic_radius_actived == 1 && pso_number_particles <= 1)
	{
		printf("\n\nERROR!!!, The dynamic radius is activated, but 'pso_number_particles' = %d, and it must to be >= 2\n",(int) pso_number_particles );
		printf("The code stops here!!!\n\n");
		exit(EXIT_FAILURE);
	}

	//TIME VARIABLES
	struct timespec GL_start, GL_finish;
	std::vector<float> GL_times(100,0.0f);

	//RANDOM VARIABLES
	UNIFORM_DISTRIBUTION_RANDOM_GENERATOR_FLOAT = uniform_real_distribution<float>(0.0, std::nextafter(1, DBL_MAX));
	if(random_deterministic == 0)
	{
		random_device rd;
		GLOBAL_GEN_RAND.seed(rd());
	}

	omp_set_num_threads(numberofthreads);
	Eigen::initParallel();

	vector<vector<size_t>> neighbourhoods;
	vector<vector<float>> neighbourhoods_distances;
	vector<vector<vector<float>>> eigenVectors;
	vector<vector<float>> eigenValues;
	
	vector<vector<float>> preferences;
	vector<vector<float>> quality_pheromone;

	vector<pair<float,size_t>> probability_std(data.size());

	vector<size_t> interesting_particle;
	vector<vector<size_t>> pso_neigbourhoods_number;

	vector<vector<vector<vector<float>>>> pso_eigenVectors;
	vector<vector<vector<float>>> pso_eigenValues;

	vector<vector<float>> pso_radii_accumulated_probabilities;
	vector<vector<float>> pso_radii_probabilities;


	// preprocess
	cout << "Preprocessing...\n";
	clock_gettime( CLOCK_REALTIME, &GL_start);
	preprocess(data, th_neighb, pso_min_radii, pso_max_radii,pso_number_particles, pso_neigbourhoods_number,
							neighbourhoods,neighbourhoods_distances, eigenVectors, eigenValues,pso_eigenVectors,pso_eigenValues,
							probability_std,interesting_particle,preferences,
							pso_radii_accumulated_probabilities, pso_radii_probabilities , dynamic_radius_actived,quality_pheromone);

	clock_gettime( CLOCK_REALTIME, &GL_finish);
	GL_times[0] += ( GL_finish.tv_sec - GL_start.tv_sec ) + ( GL_finish.tv_nsec - GL_start.tv_nsec )/ 1000000000.;

	// Validate and filter custom_init_indices if initialization_mode == 1
	// Only keep indices that point to "interesting" particles to avoid segmentation faults
	vector<size_t> validated_custom_init_indices;
	size_t effective_initialization_mode = initialization_mode;
	
	if (initialization_mode == 1 && custom_init_indices.size() > 0)
	{
		size_t original_size = custom_init_indices.size();
		size_t filtered_count = 0;
		
		for (size_t i = 0; i < custom_init_indices.size(); i++)
		{
			size_t idx = custom_init_indices[i];
			
			// Check if index is within bounds
			if (idx >= data.size())
			{
				filtered_count++;
				continue;
			}
			
			// Check if particle is "interesting" based on the mode
			bool is_valid = false;
			if (dynamic_radius_actived == 0)
			{
				// In static radius mode, interesting_particle[idx] == 1 means valid
				is_valid = (interesting_particle[idx] == 1);
			}
			else
			{
				// In dynamic radius mode, interesting_particle[idx] < pso_number_particles means valid
				is_valid = (interesting_particle[idx] < pso_number_particles);
			}
			
			if (is_valid)
			{
				validated_custom_init_indices.push_back(idx);
			}
			else
			{
				filtered_count++;
			}
		}
		
		if (filtered_count > 0)
		{
			printf("\nWARNING: %zu out of %zu custom initialization indices were filtered out.\n", 
				   filtered_count, original_size);
			printf("         These indices either exceed data bounds or point to particles with\n");
			printf("         insufficient neighbors (not 'interesting' particles).\n");
		}
		
		if (validated_custom_init_indices.empty())
		{
			printf("\nWARNING: All custom initialization indices were invalid!\n");
			printf("         Falling back to standard probability-based initialization (mode 0).\n\n");
			effective_initialization_mode = 0;
		}
		else
		{
			printf("\nUsing %zu valid custom initialization indices (out of %zu provided).\n", 
				   validated_custom_init_indices.size(), original_size);
		}
	}

	clock_gettime( CLOCK_REALTIME, &GL_start);
	// iterative step
	cout << "\nPerforming ant search...\n";
	// initial locations of all ants at each iteration
	vector<size_t> antLocations(numberOfAnts);

	// initialize pheromone value for each data point to lowerlimit
	vector<float> pheromone(data.size(), lowerlimit);

	/* pheromone delivered p satisfy the relations:
		Ph^(t) = ph^(0) * evp^t + K * Sum_(i=1)^t evp^i; K = p * N_ants * N_steps / N_particles; evp = (1-evapRate) 
		p = N_particles / (N_ants * N_steps) * (Final_expected_ph - initial_ph * evp^t) * (1-evp) / (evp * (1-evp^N_iter))
	*/
 	float pheromone_delivered;
	float evp = 1.0f - evapRate;
	pheromone_delivered = (float)data.size() / (numberOfAnts * numberOfSteps) * (final_mean_pheromone_expected_per_particle - lowerlimit * powf(evp,numberOfIterations)) * (1.0f - evp)/(evp*(1.0f-powf(evp,numberOfIterations)));

	initializeProgressBar(numberOfIterations / 2);

	//printf("\nStarting the loop...\n");
	for (size_t idx_epoch = 0; idx_epoch < numberOfIterations; ++idx_epoch)
	{
		// place ants on random points using the 'probability_std' variable
		clock_gettime( CLOCK_REALTIME, &GL_start);
		initializeAnts(antLocations, probability_std, idx_epoch, effective_initialization_mode, validated_custom_init_indices);
		clock_gettime( CLOCK_REALTIME, &GL_finish);
		GL_times[1] += ( GL_finish.tv_sec - GL_start.tv_sec ) + ( GL_finish.tv_nsec - GL_start.tv_nsec )/ 1000000000.;

		// perform ant search, spread pheromone on visited data points
		clock_gettime( CLOCK_REALTIME, &GL_start);
		if(dynamic_radius_actived == 0)
		{
			antSearch(data,
				neighbourhoods,
				antLocations,
				numberOfSteps,
				kappa,
				gamma,
				pheromone,
				pheromone_delivered,
				interesting_particle,
				preferences,
				quality_pheromone,
				external_weights,
				idx_epoch);
		}
		else if(dynamic_radius_actived == 1)
		{
			antsearch_DynamicRadius(data,
				neighbourhoods,
				antLocations,
				numberOfSteps,
				kappa,
				gamma,
				pheromone,
				pheromone_delivered,
				interesting_particle,
				pso_eigenVectors,
				pso_eigenValues,
				pso_neigbourhoods_number,
				pso_radii_accumulated_probabilities,
				pso_number_particles,
				external_weights,
				idx_epoch);
		}
		clock_gettime( CLOCK_REALTIME, &GL_finish);
		GL_times[2] += ( GL_finish.tv_sec - GL_start.tv_sec ) + ( GL_finish.tv_nsec - GL_start.tv_nsec )/ 1000000000.;

		// apply evaporation of pheromone as defined in formula (1)
		clock_gettime( CLOCK_REALTIME, &GL_start);
		evaporatePheromone(pheromone);
		clock_gettime( CLOCK_REALTIME, &GL_finish);
		GL_times[3] += ( GL_finish.tv_sec - GL_start.tv_sec ) + ( GL_finish.tv_nsec - GL_start.tv_nsec )/ 1000000000.;

		updateProgressBar(idx_epoch);
	}
	completeProgressBar();

	cout << "Locally Aligned Ant Technique algorithm completed\n\n";
	
	float total_time = std::accumulate(GL_times.begin(), GL_times.end(),
                                decltype(GL_times)::value_type(0));

	printf("\n\nLAAT EXECUTION TIMES ... \n\n");

	printf("TOTAL TIME = %.12f s\n\n",total_time);
	printf("LAAT Preprocess = %.12f s = %.3f %%\n",GL_times[0],GL_times[0] / total_time * 100);
	printf("LAAT Initialization of ants = %.12f s = %.3f %%\n",GL_times[1],GL_times[1] / total_time * 100);
	printf("LAAT Ant movement = %.12f s = %.3f %%\n",GL_times[2],GL_times[2] / total_time * 100);
	printf("LAAT Evaporation = %.12f s = %.3f %%\n",GL_times[3],GL_times[3] / total_time * 100);

	return pheromone;

}
