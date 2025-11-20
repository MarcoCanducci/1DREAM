#include "LAAT.h"

/**
 * choose the initial points so that their neighborhood size is bigger
 * or equal to the median number of neighbors.
 *
 * @param data vector containing all data points
 * @param ants vector of ants to choose the initial points for
 * @param antLocations vector to store the initial locations in
 */
void initializeAnts(vector<size_t> &antLocations,
                    vector<pair<float,size_t>> &probability_std,
										size_t idx_epoch)
{
  // Case where the ants use the particles use the probability associated with his
  // standard devaition 'probability_std'

  vector <float> acummprob(probability_std.size());
  acummprob[0] = probability_std[0].first;
  
  for (size_t idx = 1; idx < probability_std.size(); idx++)
  {
    acummprob[idx] = acummprob[idx-1] + probability_std[idx].first;
  }

  #pragma omp parallel
  {
    float target_probability;
    size_t search_idx;
		size_t myseed1 = _GLOBAL_SEED_ + idx_epoch * antLocations.size();
		size_t myseed2;

    #pragma omp for
    for (size_t ant_idx = 0; ant_idx < antLocations.size(); ant_idx++)
    {
			myseed2 = myseed1 + ant_idx;
			target_probability = floatRand(myseed2);
      search_idx = (size_t) (target_probability * (probability_std.size() - 1 ));
      
      if (acummprob[search_idx] < target_probability)
      {
        for(size_t aux_idx = search_idx; aux_idx < probability_std.size()-1; aux_idx++)
        {
          search_idx++;
          if (acummprob[search_idx] >= target_probability)
          {
            aux_idx = probability_std.size()-2;
          }
        }
      }
      else if(acummprob[search_idx] > target_probability)
      {
        for(size_t aux_idx = search_idx; aux_idx > 0; aux_idx--)
        {
          search_idx--;
          if (acummprob[search_idx] <= target_probability)
          {
            aux_idx = 1;
            search_idx++;
          }
        }
      }
      //antLocations[ant_idx] = search_idx;
      antLocations[ant_idx] = probability_std[search_idx].second;
    }
  }
}
