#include "LAAT.h"

#include "nanoflann.hpp"
#include "utils/KDTreeVectorOfVectorsAdaptor.h"
/**
 * Perform preprocessing steps as defined in Algorithm 1.
 *
 * When the function is done relative preferences have been calculated for
 * each data point and it's neighbours.
 *
 * @param data vector containing the data points to preprocess
 * @param th_neighb amount of neighbours a data point should at least
 *   have to be included in the search
 * @param neighbdradii the radius of the neighbourhood for each data point
 * @param neighbourhoods vector to store the neighbourhoods in
 * @param eigenVectors vector to store the eigen vectors in
 * @param eigenValues vector to store the eigen values in
 * @return median neighbourhood size of all data points
 */

/*

struct extreme{
  float score;
	size_t flag;
  size_t gloablal_idx;
};

#pragma omp declare reduction(vec_size_t_plus : std::vector<size_t> : \
                              std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<size_t>())) \
                    initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))


#pragma omp declare reduction(get_max : extreme :\
    omp_out = omp_out.score > omp_in.score ? omp_out : omp_in)\
    initializer (omp_priv=(omp_orig))


struct vertice
{
  	size_t number_of_edges;
  	vector<size_t> edges_idx; // graph idx elements 
  	vector<float> distances;
};


struct graph
{
  	size_t number_of_vertices;
  	vector<struct vertice> vertices;
};

*/

void preprocess(vector<vector<float>> const &data,
      size_t th_neighb, 
      float pso_min_radii,
      float pso_max_radii,
      size_t pso_number_particles,
      vector<vector<size_t>> &pso_neigbourhoods_number,
		  vector<vector<size_t>> &neighbourhoods,
      vector<vector<float>> &neighbourhoods_distances,
		  vector<vector<vector<float>>> &eigenVectors,
		  vector<vector<float>> &eigenValues,
      vector<vector<vector<vector<float>>>> &pso_eigenVectors,
      vector<vector<vector<float>>> &pso_eigenValues,
      vector<pair<float,size_t>> &probability_std,
      vector<size_t> &interesting_particle,
      // vector<vector<size_t>> &interesting_data_particles,
      vector<vector<float>> &preferences,
      vector<vector<float>> &pso_radii_accumulated_probabilities,
      vector<vector<float>> &pso_radii_probabilities,
      size_t dynamic_radius_actived,
      vector<vector<float>> &quality_pheromone)
{

  // find the local neighbourhood of all data points.
  cout << "\nPerforming ant rangeSearch...\n";
  rangeSearch(data, th_neighb, pso_max_radii, neighbourhoods,neighbourhoods_distances);

  printf("\n\n Computing initial ant probabilities\n\n");
  computing_initial_ant_probabilities(data, th_neighb, pso_max_radii, neighbourhoods,probability_std);

  printf("\n\n Computing initial particle properties\n\n");
  if(dynamic_radius_actived == 1)
  {
    computing_initial_particle_properties(data,th_neighb,neighbourhoods,neighbourhoods_distances,
		pso_min_radii,pso_max_radii, pso_number_particles,pso_neigbourhoods_number,pso_eigenVectors,pso_eigenValues,
    pso_radii_accumulated_probabilities, pso_radii_probabilities, interesting_particle );
  }
 

  printf("\n\n Continuing with the preprocess ...\n\n");


  if(dynamic_radius_actived == 0)
  {
    interesting_particle.resize(data.size(),0);
    cout << "\nPerforming ant assignment of eigen values and eigen vectors...\n";
    eigenVectors.resize(data.size());
    eigenValues.resize(data.size());

    #pragma omp parallel for schedule(dynamic,10)
    for (long long idx_point = 0; idx_point < static_cast<long long>(data.size()); idx_point++)
    {
      vector<size_t> const &neighbourhood = neighbourhoods[idx_point];

      if (neighbourhood.size() > th_neighb)
      {
        Eigen::MatrixXf X(neighbourhood.size(), 3);
        Eigen::RowVectorXf meanvec(3);

        for (size_t j = 0; j < neighbourhood.size(); j++)
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

        eigenValues[idx_point].resize(3);

        for (size_t d = 0; d < 3; ++d)
        {
          eigenValues[idx_point][d] = svalue[d];
        }

        svalue.resize(0);

        eigenVectors[idx_point].resize(3, vector<float>(3));

        for (size_t j = 0; j < 3; j++)
        {
  	      for (size_t k = 0; k < 3; k++)
          {
  	        eigenVectors[idx_point][j][k] = svector(j, k);
          }
        }

        // neighbourhood.size() > th_neighb
        // Defining the particle as "interesting", i.e. if the particle deliver (and also receive) or not deliver pheromone.
        interesting_particle[idx_point] = 1;
      }
    }

    long int only_for_1core_test_counter = 0;

    preferences.resize(data.size());
    if(pheromone_type == 1)
    {
      quality_pheromone.resize(data.size());
    }

    #pragma omp parallel
    {
      vector <float> relativeDistances(3);
      vector <float> weights(3);
      float wight_sum;
      float preferences_sum;



      #pragma omp for schedule(dynamic,10)
      for (long long idx_point = 0; idx_point < static_cast<long long>(data.size()); idx_point++)
      {
        if (interesting_particle[idx_point] == 1)
        {
          preferences_sum = 0.0f;
          vector<size_t> const &neighbourhood = neighbourhoods[idx_point];
          preferences[idx_point].resize(neighbourhood.size(),0.0f);
          if(pheromone_type == 1)
          {
            quality_pheromone[idx_point].resize(neighbourhood.size(),0.0f);
          }

          only_for_1core_test_counter += neighbourhood.size();

          for (size_t neighbour_idx = 0; neighbour_idx < neighbourhood.size(); neighbour_idx++)
          {
            for (size_t dim_idx = 0; dim_idx < 3; dim_idx++)
            {
              relativeDistances[dim_idx] = data[neighbourhood[neighbour_idx]][dim_idx] - data[idx_point][dim_idx];
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
            		relativeDistances[0] * eigenVectors[idx_point][0][dim_idx] +
            		relativeDistances[1] * eigenVectors[idx_point][1][dim_idx] +
            		relativeDistances[2] * eigenVectors[idx_point][2][dim_idx]);
                wight_sum += weights[dim_idx];
            }

            // Normalize alignment values to obtain relative weighting of the alignment according to formula (2).
            for (size_t dim_idx = 0; dim_idx < 3; dim_idx++)
            {
            	weights[dim_idx] = weights[dim_idx] / wight_sum;
            }


            for (size_t dim_idx = 0; dim_idx < 3; dim_idx++)
            {
            	preferences[idx_point][neighbour_idx] += weights[dim_idx] * eigenValues[idx_point][dim_idx]; 
            }

            preferences_sum += preferences[idx_point][neighbour_idx];

            if(pheromone_type == 1)
            {
							quality_pheromone[idx_point][neighbour_idx] = expf(beta_ph * preferences[idx_point][neighbour_idx]) / expf(beta_ph);
            }
          }

          // Normalizing the preferences.
          for (size_t neighbour_idx = 0; neighbour_idx < neighbourhood.size(); neighbour_idx++)
          {
          	preferences[idx_point][neighbour_idx] = preferences[idx_point][neighbour_idx] / preferences_sum;
          }
        }
      }
    }
  }

  // //FREE MEMEORY
  vector<vector<float>>().swap(pso_radii_probabilities);
  vector<vector<vector<float>>>().swap(eigenVectors);
  vector<vector<float>>().swap(eigenValues);
  




}

