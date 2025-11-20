#include "LAAT.h"
#include "nanoflann/nanoflann.hpp"
#include "utils/KDTreeVectorOfVectorsAdaptor.h"
using namespace nanoflann;

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
void rangeSearch(vector<vector<float>> const &data,
        size_t th_neighb,
        float pso_max_radii,
  		   vector<vector<size_t>> &neighbourhoods,
         vector<vector<float>> &neighbourhoods_distances)
{

  // nanoflann library uses r^2 as the radius
  float neighbdradii2 = pso_max_radii * pso_max_radii;

  typedef KDTreeVectorOfVectorsAdaptor<
    vector<vector<float>>, float> KDTree;
  
  KDTree KNN(data[0].size() /* dim */, data);
  KNN.index->buildIndex();

  vector<size_t> sizes(data.size());
  neighbourhoods.resize(data.size());
  neighbourhoods_distances.resize(data.size());
  
#pragma omp parallel for schedule(dynamic,10)
  for (size_t idx = 0; idx < data.size(); idx++)
  {
    vector<pair<size_t, float>> matches;


    vector<size_t> &neighbourhood = neighbourhoods[idx];
    vector<float> &neighbourhood_dist = neighbourhoods_distances[idx];

    // find neighbors
    size_t nMatches = KNN.index->radiusSearch(&data[idx][0],
					      neighbdradii2,
					      matches,
					      SearchParams());
    
    size_t counter_dist_0 = 1;

    while (counter_dist_0 < nMatches && matches[counter_dist_0].second == 0.0f)
    {
      counter_dist_0++;
    }
    sizes[idx] = nMatches - counter_dist_0;

    // store neighbors
    neighbourhood.resize(nMatches - counter_dist_0);
    neighbourhood_dist.resize(nMatches - counter_dist_0);
    for (size_t j = counter_dist_0; j < nMatches; ++j)
    {
      neighbourhood[j - counter_dist_0] = matches[j].first;
      neighbourhood_dist[j - counter_dist_0] = matches[j].second;
    }
  }

  //Median value of the std
  int count = 0;
  for (size_t idx = 0; idx< data.size();idx++)
  {
    if(sizes[idx] <= th_neighb)
    count++;
  }

  printf("%d Number of points with Neighbourhoods lower than the Threshold %d\n",count,(int)th_neighb);

  // MEDIAN VALUE
  size_t n = sizes.size() / 2;
  nth_element(sizes.begin(), sizes.begin() + n, sizes.end());
  cout << "Median value of the size of the Neighbourhoods = " << sizes[n] << endl << endl;

  if(sizes[n] < th_neighb)
  {
    cout << "Warning, median value lower than threshold, we use the threshold as the median value" << endl;
  }

  // FREE MEMORY
  vector<size_t>().swap(sizes);

}