#include "LAAT.h"

/**
 * Apply evaporation of pheromone as defined in formula (1) to all data
 * points, do no let pheromone rise above the upper limit or fall below the
 * lower limit.
 *
 * @param data    vector containing the data points to apply evaporation of
 *                pheromones to.
 * @param evapRate rate at which to evaporate pheromone after each application
 *   of ant search
 * @param lowerlimit lower limit on the amount of pheromone of a data point
 * @param upperlimit upper limit on the amount of pheromone of a data point
 */
void evaporatePheromone(vector<float> &pheromone)
{
  float newPheromone;
  #pragma omp parallel for private(newPheromone)
  for (long long i = 0; i < static_cast<long long>(pheromone.size()); ++i)
  {
    newPheromone = (1 - evapRate) * pheromone[i];

    if (newPheromone < lowerlimit)
    {
      newPheromone = lowerlimit;
    }
    else if (newPheromone > upperlimit)
    {
      newPheromone = upperlimit;
    }

    pheromone[i] = newPheromone;

  }
}
