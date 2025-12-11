#include "LAAT.h"


/**
 * Find the neighborhood of every data point, exclude data points that have
 * less neighbors than the threshold.
 * 
 *
 * This function uses the nanoflann library to find all of the neighbors in
 * an efficient way.
 *
 * When the function is done data points will contain the indices of all of
 * their neighbors.
 *
 * @param data      vector containing the data to find the neighborhoods of
 * @param th_neighb minimum number of neighbors needed for the data to be
 *   included
 * @param neighbdradii    the radius of each neighborhood
 * @param neighbourhoods vector to store the neighbourhoods in
 * @return          the median number of neighbors over all data points
 */
void computing_initial_particle_properties(vector<vector<float>> const &data,
        size_t th_neighb, 
        vector<vector<size_t>> &neighbourhoods,
        vector<vector<float>> &neighbourhoods_distances,
        float pso_min_radii,
        float pso_max_radii,
        size_t pso_number_particles,
        vector<vector<size_t>> &pso_neigbourhoods_number,
        vector<vector<vector<vector<float>>>> &pso_eigenVectors,
        vector<vector<vector<float>>> &pso_eigenValues,
        vector<vector<float>> &pso_radii_accumulated_probabilities,
				vector<vector<float>> &pso_radii_probabilities,
        vector<size_t> &interesting_particle)
{

  vector<vector<float>> pso_radii(pso_number_particles);

  float aux;
  for(size_t idx = 0; idx < pso_number_particles; idx++)
  {
    aux = (pso_max_radii - pso_min_radii) * ( (float)idx /(pso_number_particles-1) ) + pso_min_radii;
    aux = aux * aux;  //The distances in the neighbourhoods_distances are  stored to the power of 2
    pso_radii[idx].resize(data.size(),aux);
  }




  //Computing Number of neighbours for the particles of the pso
  pso_neigbourhoods_number.resize(pso_number_particles);
  for(size_t idx_point = 0; idx_point < pso_number_particles; idx_point++)
  {
    pso_neigbourhoods_number[idx_point].resize(data.size(),0);
  }
  
  size_t min_neighbours_number, max_neighbours_number;
  float dt;
  size_t number_of_particles_to_use;

  //Filling pso_number_neighbours
  printf("Step 1/6 \n");
  #pragma omp parallel for private(min_neighbours_number, max_neighbours_number, number_of_particles_to_use, dt) schedule(dynamic,10)
  for(long long idx_point = 0; idx_point < static_cast<long long>(data.size()); idx_point++)
  {
    max_neighbours_number = 0;
    if(neighbourhoods[idx_point].size() > th_neighb)
    {
      max_neighbours_number = neighbourhoods[idx_point].size();
      min_neighbours_number = 0;
      while(min_neighbours_number < max_neighbours_number && neighbourhoods_distances[idx_point][min_neighbours_number] < pso_min_radii )
      {
        min_neighbours_number++;
      }
      if(min_neighbours_number < th_neighb + 1)
      {
        min_neighbours_number = th_neighb + 1;
      }
      number_of_particles_to_use = max_neighbours_number - min_neighbours_number + 1;
      dt = 1.0f;
      if(pso_number_particles < number_of_particles_to_use)
      {
        number_of_particles_to_use = pso_number_particles;
        if(pso_number_particles > 1)
        {
          dt = (float) (max_neighbours_number - min_neighbours_number) / (float) (pso_number_particles-1);
        }
      }
      //Filling pso_neigbourhoods_number
      pso_neigbourhoods_number[pso_number_particles-1][idx_point] = max_neighbours_number;
      for(size_t  idx_particle = pso_number_particles - number_of_particles_to_use; idx_particle < pso_number_particles-1; idx_particle++)
      {
        pso_neigbourhoods_number[idx_particle][idx_point] = min_neighbours_number + (size_t) (dt * (float) (idx_particle - (pso_number_particles - number_of_particles_to_use) ) );
      }
    }
  }

  //Filling interesting particle, first call
  printf("Step 2/6 \n");
  interesting_particle.resize(data.size(),0);
  size_t idx_particle_aux;
  #pragma omp parallel for schedule(dynamic,10) private(idx_particle_aux)
  for(long long idx_point = 0; idx_point < static_cast<long long>(data.size()); idx_point++)
  {
    idx_particle_aux = pso_number_particles;
    while(idx_particle_aux > 0 && pso_neigbourhoods_number[idx_particle_aux-1][idx_point] > 0)
    {
      idx_particle_aux--;
    }   
    interesting_particle[idx_point] = idx_particle_aux;
  }

  // Computing Eigen-vectors and Eigen-values using the Principal component analysis (PCA) 

  pso_eigenVectors.resize(pso_number_particles);
  pso_eigenValues.resize(pso_number_particles);
  for(size_t idx = 0; idx < pso_number_particles; idx++)
  {
    pso_eigenVectors[idx].resize(data.size());
    pso_eigenValues[idx].resize(data.size());
  }



  //Filling pso Eigenvalues and pso Eigenvectors
  printf("Step 3/6 \n");
  
  #pragma omp parallel for schedule(dynamic,10)
  for(long long idx_point = 0; idx_point < static_cast<long long>(data.size()); idx_point++)
  {
    vector<size_t> const &neighbourhood = neighbourhoods[idx_point];
    size_t aux_start_idx = interesting_particle[idx_point];

    for(size_t idx_particle = aux_start_idx; idx_particle < pso_number_particles; idx_particle++)
    {
      //printf("idx_point = %d, idx_particle = %d, total neighbours = %d, local neighbours = %d\n",(int) idx_point,(int)idx_particle,(int) neighbourhood.size(),(int)pso_neigbourhoods_number[idx_particle][idx_point]);
      
      pso_eigenVectors[idx_particle][idx_point].resize(3, vector<float>(3));
      pso_eigenValues[idx_particle][idx_point].resize(3);
      Eigen::MatrixXf X(pso_neigbourhoods_number[idx_particle][idx_point] , 3);
      Eigen::RowVectorXf meanvec(3);
      for (size_t j = 0; j < pso_neigbourhoods_number[idx_particle][idx_point]; j++)
      {
        for (size_t dim = 0; dim < 3; dim++)
        {
          X(j, dim) = data[neighbourhood[j]][dim];
        }   
      }
      for (size_t dim = 0; dim < 3; ++dim)
      {
        meanvec(dim) = X.middleCols<1>(dim).mean();
      }
      X = X.rowwise() - meanvec;
      meanvec.resize(0);  // destructor
      Eigen::BDCSVD<Eigen::MatrixXf> svd(X, Eigen::ComputeThinV);
      X.resize(0,0); // this is the destructor
      Eigen::RowVectorXf svalue(3);
      svalue = svd.singularValues();
      Eigen::Matrix3f svector;
      svector = svd.matrixV();          // eigen vectors are in columns
      svalue = svalue.array().pow(2);   // second power of singular values
      svalue = svalue.array() / svalue.sum();  // normalize eigen values
      // Filling the pso EigenValues
      for (size_t d = 0; d < 3; ++d)
      {
        pso_eigenValues[idx_particle][idx_point][d] = svalue[d];
      }
      // Filling the pso EigenVectors
      for (size_t j = 0; j < 3; j++)
      {
	      for (size_t k = 0; k < 3; k++)
        {
	        pso_eigenVectors[idx_particle][idx_point][j][k] = svector(j, k);
        }
      }
      // Free memeory
      svalue.resize(0);
    }    
  }

  printf("Step 4/6 \n");
	vector<vector<vector<vector<float>>>> pso_eigenVectors_reduced;
  vector<vector<vector<float>>> pso_eigenValues_reduced;

  pso_eigenVectors_reduced.resize(pso_number_particles);
  pso_eigenValues_reduced.resize(pso_number_particles);
  for(size_t idx_particle = 0; idx_particle < pso_number_particles; idx_particle++)
  {
    pso_eigenVectors_reduced[idx_particle].resize(data.size());
    pso_eigenValues_reduced[idx_particle].resize(data.size());
  }
  
  // Computing the Eigen-values and Eigen-vectors using percentage_removed of min data
  #pragma omp parallel
  {
    vector<pair<float,size_t>> local_preference, local_preference_sorted;
    vector<float> v_ij(3);
    float norm,wight_sum;
    vector<float> weights(3);
    size_t cut_percentage_removed_percentage;
    vector<size_t> idx_points_at_percentage_removed_percentage(data.size());
		vector<size_t> aux_rand_idx(data.size());
		size_t myseed1;
		size_t random_number;
    #pragma omp for schedule(dynamic,10)
    for(long long idx_point = 0; idx_point < static_cast<long long>(data.size()); idx_point++)
    {
      vector<size_t> const &neighbourhood = neighbourhoods[idx_point];
      size_t aux_start_idx = interesting_particle[idx_point];

      for(size_t idx_particle = aux_start_idx; idx_particle < pso_number_particles; idx_particle++)
      {
        pso_eigenVectors_reduced[idx_particle][idx_point].resize(3, vector<float>(3));
        pso_eigenValues_reduced[idx_particle][idx_point].resize(3,0.0f); 
        local_preference.resize(pso_neigbourhoods_number[idx_particle][idx_point]);
        local_preference_sorted.resize(pso_neigbourhoods_number[idx_particle][idx_point]);
        
        //Commputing local preference
        for(size_t idx_neighbour = 0; idx_neighbour < pso_neigbourhoods_number[idx_particle][idx_point]; idx_neighbour++)
        {
          for(size_t idx_dim = 0; idx_dim < 3; idx_dim++)
          {
            v_ij[idx_dim] = data[idx_point][idx_dim] - data[neighbourhood[idx_neighbour]][idx_dim];
          }
          norm = sqrtf(v_ij[0] * v_ij[0] + v_ij[1] * v_ij[1] + v_ij[2] * v_ij[2]); 
          for(size_t idx_dim = 0; idx_dim < 3; idx_dim++)
          {
            v_ij[idx_dim] = v_ij[idx_dim]/norm;
          }
          //Compute alignment between the data point and it's neighbours with the eigen-directions.
          wight_sum = 0.0f;
          for (size_t idx_dim = 0; idx_dim < 3; idx_dim++)
          {
            // matrix multiplication
          	weights[idx_dim] = fabs(
          	v_ij[0] * pso_eigenVectors[idx_particle][idx_point][0][idx_dim] +
          	v_ij[1] * pso_eigenVectors[idx_particle][idx_point][1][idx_dim] +
          	v_ij[2] * pso_eigenVectors[idx_particle][idx_point][2][idx_dim]);
            wight_sum += weights[idx_dim];
          }
          // Normalize alignment values to obtain relative weighting of the alignment according to formula (2).
          for (size_t idx_dim = 0; idx_dim < 3; idx_dim++)
          {
          	weights[idx_dim] = weights[idx_dim] / wight_sum;
          }
					local_preference[idx_neighbour].first = 0.0f;
          for (size_t idx_dim = 0; idx_dim < 3; idx_dim++)
          {
          	local_preference[idx_neighbour].first += weights[idx_dim] * pso_eigenValues[idx_particle][idx_point][idx_dim]; 
          }
          local_preference[idx_neighbour].second = idx_neighbour; 
        }
        //Sorting local preference
        local_preference_sorted = local_preference;
        cut_percentage_removed_percentage = (1.0f - percentage_removed / 100.0f)  * pso_neigbourhoods_number[idx_particle][idx_point];

        if(sorted_removed_type == 0) // // //////////////////////////max sorted
        {
          std::sort (local_preference_sorted.begin(), local_preference_sorted.end());
          for(size_t cut_idx = 0; cut_idx < cut_percentage_removed_percentage; cut_idx++)
          {
            idx_points_at_percentage_removed_percentage[cut_idx] = local_preference_sorted[cut_idx].second;
          }
        }
        else if (sorted_removed_type == 1) //////////////////////////random sorted
        {
          for(size_t aux_idx = 0; aux_idx < pso_neigbourhoods_number[idx_particle][idx_point]; aux_idx++)
          {
            aux_rand_idx[aux_idx] = aux_idx;
          }
					size_t neighbourhood_size_removed = pso_neigbourhoods_number[idx_particle][idx_point];
          
					myseed1 = _GLOBAL_SEED_ + idx_point * pso_number_particles + idx_particle * cut_percentage_removed_percentage;

					for(size_t cut_idx = 0; cut_idx < cut_percentage_removed_percentage; cut_idx++)
          {
						//random_number = rand_r(&myseed) % neighbourhood_size_removed;
						
						//random_number = sizetRand(0, 0, idx_point, idx_particle, cut_idx, neighbourhood_size_removed) ;
						random_number = sizetRand(myseed1, neighbourhood_size_removed);
            idx_points_at_percentage_removed_percentage[cut_idx] = aux_rand_idx[random_number];
						aux_rand_idx[random_number] = aux_rand_idx[neighbourhood_size_removed-1];
						neighbourhood_size_removed--;
          }
        }
        ///////////////////////////////////////////////
        // Computing the Eigenvectors and Eigenvalues
        Eigen::MatrixXf X(cut_percentage_removed_percentage, 3);
        Eigen::RowVectorXf meanvec(3);
        for (size_t j = 0; j < cut_percentage_removed_percentage; j++)
        {
          for (size_t dim = 0; dim < 3; dim++)
          {
            X(j, dim) = data[neighbourhood[idx_points_at_percentage_removed_percentage[j]]][dim];
          }   
        }
        for (size_t dim = 0; dim < 3; ++dim)
        {
          meanvec(dim) = X.middleCols<1>(dim).mean();
        }
        X = X.rowwise() - meanvec;
        meanvec.resize(0);  // destructor
        Eigen::BDCSVD<Eigen::MatrixXf> svd(X, Eigen::ComputeThinV);
        X.resize(0,0); // this is the destructor
        Eigen::RowVectorXf svalue(3);
        svalue = svd.singularValues();
        Eigen::Matrix3f svector;
        svector = svd.matrixV();          // eigen vectors are in columns
        svalue = svalue.array().pow(2);   // second power of singular values
        svalue = svalue.array() / svalue.sum();  // normalize eigen values
        
        // Filling the pso EigenValues
        for (size_t d = 0; d < 3; ++d)
        {
          pso_eigenValues_reduced[idx_particle][idx_point][d] = svalue[d];
        }
        // Filling the pso EigenVectors
        for (size_t j = 0; j < 3; j++)
        {
	        for (size_t k = 0; k < 3; k++)
          {
	          pso_eigenVectors_reduced[idx_particle][idx_point][j][k] = svector(j, k);
          }
        }
        // Free memeory
        svalue.resize(0);
      }
    }
    //FREE MEMORY
    vector<pair<float,size_t>>().swap(local_preference);
    vector<pair<float,size_t>>().swap(local_preference_sorted);
    vector<size_t>().swap(idx_points_at_percentage_removed_percentage) ;
		vector<size_t>().swap(aux_rand_idx) ;
  }

  //Radii scores
  printf("Step 5/6 \n");
	//vector<vector<float>> pso_radii_probabilities;
  vector<vector<float>> pso_radii_cloud_points_normalized_distances;

  //Computing distance between eigen vectors full and reduced
  pso_radii_cloud_points_normalized_distances.resize(pso_number_particles);
  pso_radii_accumulated_probabilities.resize(pso_number_particles);
	pso_radii_probabilities.resize(pso_number_particles);
  for(size_t idx_particle = 0; idx_particle < pso_number_particles; idx_particle++)
  {
    pso_radii_cloud_points_normalized_distances[idx_particle].resize(data.size(),0.0f);
    pso_radii_accumulated_probabilities[idx_particle].resize(data.size(),0.0f);
		pso_radii_probabilities[idx_particle].resize(data.size(),0.0f);
  }
  float aux_normalization_radii_quality;
	size_t aux_counter = 0;
	float mean_distances = 0.0f;
  #pragma omp parallel for schedule(dynamic,10) private(aux_normalization_radii_quality)
  for(long long idx_point = 0; idx_point < static_cast<long long>(data.size()); idx_point++)
  {
    aux_normalization_radii_quality = 0.0f;
    
    float aux_G = 1.0f;
		float float_aux;

    
    size_t aux_start_idx = interesting_particle[idx_point];
		if(aux_start_idx > pso_number_particles-2)
		{
			pso_radii_accumulated_probabilities[pso_number_particles-1][idx_point] = 1.0f;
			pso_radii_probabilities[pso_number_particles-1][idx_point] = 1.0f;
		}
		else
		{
    	for(size_t idx_particle = aux_start_idx; idx_particle < pso_number_particles; idx_particle++)
    	{
    	  // if (interesting_data_particles[idx_particle][idx_point] == 1)
    	  // {
				aux_counter++;
    	  if(option_score == 0 || option_score == 1)
    	  {
					float grasmann_distance = 0.0f; 
    	    for(size_t idx_dim = 0; idx_dim < 3; idx_dim++)
    	    {
						float_aux = fabs(
    	      pso_eigenVectors[idx_particle][idx_point][0][idx_dim] * pso_eigenVectors_reduced[idx_particle][idx_point][0][idx_dim] +
    	      pso_eigenVectors[idx_particle][idx_point][1][idx_dim] * pso_eigenVectors_reduced[idx_particle][idx_point][1][idx_dim] +
    	      pso_eigenVectors[idx_particle][idx_point][2][idx_dim] * pso_eigenVectors_reduced[idx_particle][idx_point][2][idx_dim] );
						if(float_aux < 1.0f)
						{
							grasmann_distance += acosf(float_aux);
						}
						
    	    }
					pso_radii_cloud_points_normalized_distances[idx_particle][idx_point] = grasmann_distance * 2.0f / (3.0f * _PI_);	// 0 <= variable  <= 1.0 
    	  }
    	  else if(option_score == 2 || option_score == 3)
    	  {
    	    Eigen::MatrixXf W_1(3,3);
    	    Eigen::MatrixXf W_2(3,3);
    	    Eigen::RowVectorXf eigenvalues_1(3);
    	    Eigen::RowVectorXf eigenvalues_2(3);
    	    for(size_t idx_row = 0; idx_row < 3; idx_row++)
    	    {
    	        for(size_t idx_column = 0; idx_column < 3; idx_column++)
    	        {
    	            W_1(idx_row,idx_column) = pso_eigenVectors[idx_particle][idx_point][idx_row][idx_column];
    	            W_2(idx_row,idx_column) = pso_eigenVectors_reduced[idx_particle][idx_point][idx_row][idx_column];
    	        }
    	    }
    	    eigenvalues_1 << pso_eigenValues[idx_particle][idx_point][0] ,
    	                    pso_eigenValues[idx_particle][idx_point][1] ,
    	                    pso_eigenValues[idx_particle][idx_point][2];
    	    eigenvalues_2 << pso_eigenValues_reduced[idx_particle][idx_point][0] ,
    	                    pso_eigenValues_reduced[idx_particle][idx_point][1] ,
    	                    pso_eigenValues_reduced[idx_particle][idx_point][2];
    	    Eigen::MatrixXf Q_1 = W_1 * eigenvalues_1.asDiagonal() * W_1.transpose();
    	    Eigen::MatrixXf Q_2 = W_2 * eigenvalues_2.asDiagonal() * W_2.transpose();
    	    Eigen::MatrixXf Qmean = (Q_1 + Q_2) * 0.5f;
    	    float bhattacharyya_distance = 0.25f * log(Qmean.determinant() * Qmean.determinant()  / (Q_1.determinant() * Q_2.determinant()) );
    	    float Hellinger_distance = sqrtf(fabs(1.0 - exp(-bhattacharyya_distance) ) );
					
					pso_radii_cloud_points_normalized_distances[idx_particle][idx_point] = Hellinger_distance ;	// 0 <= variable  < 1.0
    	  }
				mean_distances += pso_radii_cloud_points_normalized_distances[idx_particle][idx_point];

				size_t K_i = pso_number_particles - aux_start_idx; 
				if (K_i == 1)
				{
					K_i++;
				}
    	  
				//Choosing coefficient function of K,K_i
				if(option_coefficientK == 0)	//Log(K)
				{
					aux_G = logf(pso_number_particles);
				}
				else if(option_coefficientK == 1)	//Log(K_i)
				{
					aux_G = logf(K_i);
				}
				else if(option_coefficientK == 2)	//1.0
				{
					aux_G = 1.0f;
				}
				else if(option_coefficientK == 3)	//1.0/Log(K_i)
				{
					aux_G = 1.0f / logf(K_i);
				}
				else if(option_coefficientK == 4)	//1.0/Log(K)
				{
					aux_G = 1.0f / logf(pso_number_particles);
				}

				//Compurint probabilities for the different score options
    	  if(option_score == 0 || option_score == 2)	//Parallel Grasmann and Bhattacharyya (Hellinger)
    	  {
					pso_radii_probabilities[idx_particle][idx_point] = expf(- aux_G * pso_radii_cloud_points_normalized_distances[idx_particle][idx_point] * pso_radii_cloud_points_normalized_distances[idx_particle][idx_point]);
    	  }
    	  else if(option_score == 1 || option_score == 3)	//Orthogonal Grasmann and Bhattacharyya (Hellinger)
    	  {
					pso_radii_probabilities[idx_particle][idx_point] = expf(- aux_G * (1.0f - pso_radii_cloud_points_normalized_distances[idx_particle][idx_point]) * (1.0f - pso_radii_cloud_points_normalized_distances[idx_particle][idx_point]) );
    	  }
				aux_normalization_radii_quality += pso_radii_probabilities[idx_particle][idx_point];
    	}

    	//computing accumulated probability
    	for(size_t idx_particle = aux_start_idx; idx_particle < pso_number_particles; idx_particle++)
    	{
				pso_radii_probabilities[idx_particle][idx_point] /= aux_normalization_radii_quality;
    	}
    
			pso_radii_accumulated_probabilities[aux_start_idx][idx_point] = pso_radii_probabilities[aux_start_idx][idx_point];
    	for(size_t idx_particle = aux_start_idx + 1; idx_particle < pso_number_particles; idx_particle++)
    	{
    	  pso_radii_accumulated_probabilities[idx_particle][idx_point] = pso_radii_probabilities[idx_particle][idx_point] + pso_radii_accumulated_probabilities[idx_particle-1][idx_point];
    	}
		}
  }

  printf("Step 6/6 \n");
  // //FREE MEMEORY

  vector<vector<float>>().swap(pso_radii); 
  vector<vector<vector<vector<float>>>>().swap(pso_eigenVectors_reduced);
  vector<vector<vector<float>>>().swap(pso_eigenValues_reduced);
  vector<vector<float>>().swap(pso_radii_cloud_points_normalized_distances) ;
  vector<vector<float>>().swap(neighbourhoods_distances);

}



