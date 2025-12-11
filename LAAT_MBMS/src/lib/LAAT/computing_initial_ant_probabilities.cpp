#define _USE_MATH_DEFINES
#include <cmath>
#include <random>
#include <algorithm>

#include "LAAT.h"

// Auxiliary functions for 3D case (original implementation)
void initializing_octants_and_flags(vector<vector<float>>  &octants_dist, vector<vector<int>>  &flags, float pos_x, float pos_y, float pos_z, float xmin, float xmax, float ymin, float ymax, float zmin, float zmax, float radius);
float compute_octante_volumen(vector<float>  &octants_dist_idx, vector<int>  &flags_idx, float radius);

// Auxiliary functions for N-dimensional case
float compute_hypersphere_volume(size_t D, float radius);
float compute_boundary_volume_ratio_monte_carlo(vector<float> const &point, vector<float> const &mins, vector<float> const &maxs, float radius, size_t D, size_t num_samples = 1000);
float compute_boundary_volume_ratio_analytical(vector<float> const &point, vector<float> const &mins, vector<float> const &maxs, float radius, size_t D);

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
void computing_initial_ant_probabilities(vector<vector<float>> const &data,
        size_t th_neighb,
        float pso_max_radii,
  		   vector<vector<size_t>> &neighbourhoods,
         vector<pair<float,size_t>> &probability_std)
{
  float standDeviation;
  
  // Get data dimensionality
  size_t D = data[0].size();

  //Computing the standard deviation
  
  //Mean values
  //Computing the densities (Note that the boundary points has a different density than the interior points)
  vector<float> densities(data.size()); 
  
  // Compute bounding box for all dimensions
  vector<float> mins(D), maxs(D);
  for (size_t d = 0; d < D; d++) {
    mins[d] = data[0][d];
    maxs[d] = data[0][d];
  }

#pragma omp parallel
  {
    vector<float> local_mins(D), local_maxs(D);
    for (size_t d = 0; d < D; d++) {
      local_mins[d] = data[0][d];
      local_maxs[d] = data[0][d];
    }
    
    #pragma omp for nowait
    for (long long idx = 1; idx < static_cast<long long>(data.size()); idx++)
    {
      for (size_t d = 0; d < D; d++) {
        local_mins[d] = std::min(local_mins[d], data[idx][d]);
        local_maxs[d] = std::max(local_maxs[d], data[idx][d]);
      }
    }
    
    #pragma omp critical
    {
      for (size_t d = 0; d < D; d++) {
        mins[d] = std::min(mins[d], local_mins[d]);
        maxs[d] = std::max(maxs[d], local_maxs[d]);
      }
    }
  }

  // For 3D, use the original optimized octant-based approach
  // For higher dimensions, use a generalized approach
  if (D == 3) {
    float xmin = mins[0], xmax = maxs[0];
    float ymin = mins[1], ymax = maxs[1];
    float zmin = mins[2], zmax = maxs[2];
    
    //Neighbourhood Density computation
    #pragma omp parallel
    {
      vector<vector<int>> flags(8, vector<int>(3));
      vector<vector<float>> octants_dist(8, vector<float>(3));

      float total_vol;
      #pragma omp for schedule(dynamic,10)
      for (long long idx = 0; idx < static_cast<long long>(data.size()); idx++)
      {
        total_vol = 0.0f;
        initializing_octants_and_flags(octants_dist, flags, data[idx][0], data[idx][1], data[idx][2], xmin, xmax, ymin, ymax, zmin, zmax, pso_max_radii);
        for (size_t hh = 0; hh < 8; hh++)
        {
          total_vol += compute_octante_volumen(octants_dist[hh], flags[hh], pso_max_radii);
        }

        densities[idx] = (float)neighbourhoods[idx].size() / total_vol;
      }
    }
  }
  else {
    // Generalized N-dimensional approach
    // Use analytical approximation for volume ratio (faster) or Monte Carlo for more accuracy
    float full_hypersphere_volume = compute_hypersphere_volume(D, pso_max_radii);
    
    #pragma omp parallel for schedule(dynamic,10)
    for (long long idx = 0; idx < static_cast<long long>(data.size()); idx++)
    {
      // Compute the fraction of the hypersphere that lies within the bounding box
      float volume_ratio = compute_boundary_volume_ratio_analytical(data[idx], mins, maxs, pso_max_radii, D);
      float effective_volume = full_hypersphere_volume * volume_ratio;
      
      // Avoid division by zero for edge cases
      if (effective_volume < 1e-10f) {
        effective_volume = full_hypersphere_volume;
      }
      
      densities[idx] = (float)neighbourhoods[idx].size() / effective_volume;
    }
  }

  float probability_std_sum = 0.0f;
  float mean;

  vector<pair<float,size_t>> probability_std_aux(data.size());
  size_t aux_counter = 0;
#pragma omp parallel for reduction(+ : probability_std_sum, aux_counter) private(mean) schedule(dynamic,10)
  for (long long idx = 0; idx < static_cast<long long>(data.size()); idx++)
  {
    if( neighbourhoods[idx].size() > th_neighb)
    {
      mean = 0.0f;
      vector<size_t> &neighbourhood = neighbourhoods[idx];
      for(size_t i = 0; i < neighbourhoods[idx].size(); i++  )
      {
        mean += densities[neighbourhood[i]];
      }

      mean = mean / neighbourhoods[idx].size() ;

      //The computation of the standard deviation using the nex expression can produce numerical error 
      //because the expresion can be squares - mean*mean * sizes[idx] < 0 for the computer computation
      //standDeviation[idx] = (1.0f/sizes[idx]) * (squares - mean*mean * sizes[idx] ) ;

			standDeviation = 0.0f;

      for(size_t i = 0; i < neighbourhoods[idx].size(); i++)
      {
        standDeviation  += (densities[neighbourhood[i]] - mean) * (densities[neighbourhood[i]] - mean);
      }
      standDeviation /= neighbourhoods[idx].size();
      standDeviation = sqrtf(standDeviation);
    
      //Normalized version
      standDeviation = standDeviation / mean;
      probability_std_aux[idx].first = expf(beta_antinitialization * standDeviation);
      probability_std_aux[idx].second = idx;
      probability_std_sum += probability_std_aux[idx].first;
      aux_counter++;
    }
  }

  probability_std.resize(aux_counter);
  //Filling the probability_std vector
  aux_counter = 0;
  for (size_t aux_idx = 0; aux_idx < data.size(); aux_idx++)
  {
    if(neighbourhoods[aux_idx].size() > th_neighb)
    {
      probability_std[aux_counter] = probability_std_aux[aux_idx];
      aux_counter++;
    } 
  }
  vector<pair<float,size_t>>().swap(probability_std_aux);


  // size_t aux_counter = 0;
  // //Removing 0 probability points of the probability_std vector
  //   for (size_t idx = 0; idx < data.size(); idx++)
  //   {
  //     if( neighbourhoods[idx].size() <= th_neighb)
  //     {
  //       probability_std.erase(probability_std.begin() + idx - aux_counter);
  //       aux_counter++;
  //     }
  //   }


  // //EXPORTING STD
  // FILE *file = NULL;
  // file = fopen("../Output/std.csv", "w");
  // if (file == NULL)
  // {
  //   printf("Error opening Output file!");
  //   exit(EXIT_FAILURE);
  // }

  // for( size_t idx = 0; idx < data.size(); idx++)
  // {
  //   fprintf(file, "%1.12f\n",probability_std[idx] );
  // }
  // fclose(file);




  //Normalizing the probability.
  #pragma omp parallel for
  for (long long idx = 0; idx < static_cast<long long>(probability_std.size()); idx++)
  {
    probability_std[idx].first /= probability_std_sum;
  }

}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////// LOCAL FUNCTIONS /////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

void initializing_octants_and_flags(vector<vector<float>>  &octants_dist, vector<vector<int>>  &flags, float pos_x, float pos_y, float pos_z, float xmin, float xmax, float ymin, float ymax, float zmin, float zmax, float radius)
{

  size_t idx;
  float projection_distance;
  for(size_t k = 0 ; k<2;k++)
  {
    for(size_t j = 0; j < 2;j++)
    {
      for(size_t i = 0; i < 2 ; i++)
      {
        idx = i + j*2 + k*4;
        octants_dist[idx][0] = (((pos_x - xmin) * (1 - i ) + (xmax - pos_x) * i) < radius) ? (pos_x - xmin) * (1 - i ) + (xmax - pos_x) * i : radius; 
        octants_dist[idx][1] = (((pos_y - ymin) * (1 - j ) + (ymax - pos_y) * j) < radius) ? (pos_y - ymin) * (1 - j ) + (ymax - pos_y) * j : radius; 
        octants_dist[idx][2] = (((pos_z - zmin) * (1 - k ) + (zmax - pos_z) * k) < radius) ? (pos_z - zmin) * (1 - k ) + (zmax - pos_z) * k : radius; 
        
        // flags
        for(size_t hh = 0; hh<3;hh++)
        {
          flags[idx][hh] = 0;
          if(octants_dist[idx][hh] < radius)
          {
            flags[idx][hh] = 1;
            if(octants_dist[idx][((int)hh+1)%3] < radius ) 
            {
              projection_distance = sqrtf(octants_dist[idx][hh] * octants_dist[idx][hh] + octants_dist[idx][((int)hh+1)%3] * octants_dist[idx][((int)hh+1)%3]);
              if(projection_distance <= radius)
              {
                flags[idx][hh] = 2;
              }
            }
            else if(octants_dist[idx][((int)hh+2)%3] < radius )
            {
              projection_distance = sqrtf(octants_dist[idx][hh] * octants_dist[idx][hh] + octants_dist[idx][((int)hh+2)%3] * octants_dist[idx][((int)hh+2)%3]);
              if(projection_distance <= radius)
              {
                flags[idx][hh] = 2;
              }
            }
          }
        }
      }
    }
  }
}


float compute_octante_volumen(vector<float>  &octants_dist_idx, vector<int>  &flags_idx, float radius)
{
  int option = flags_idx[0] + flags_idx[1] + flags_idx[2];
  float vol = 0.0f;
  float octante_normal_volumen = 1.0f / 6.0f * M_PI * radius*radius*radius;
  float X_i;

  //# Case the octante is completly inside of the box
  if(option == 0)
  {
    vol = octante_normal_volumen;
  }
  // # Only one direcction is outside of the box
  else if(option == 1)
  {
    X_i = flags_idx[0]  * octants_dist_idx[0] + flags_idx[1]  * octants_dist_idx[1]  + flags_idx[2]  * octants_dist_idx[2] ;
    vol = M_PI / 4.0f * (radius*radius * X_i - X_i*X_i*X_i / 3.0f);
  }
  //# 2 direcctions are outside but both has flag 1
  else if(option == 2)
  {
    for (int hh = 0; hh <3; hh++)
    {
      if(flags_idx[hh] == 1)
      {
        X_i = octants_dist_idx[0] * 0.5f * (hh - 1) * (hh - 2) + octants_dist_idx[1]  * hh * (2 - hh) + octants_dist_idx[2]  * 0.5f * hh * (hh - 1);
        vol += M_PI / 4.0f * (radius*radius * X_i - X_i*X_i*X_i / 3.0f);
      }
    }
    vol -= octante_normal_volumen ;
  }
  // # 3 direcctions are outside but all of them has flag 1
  else if(option == 3)
  {
    for (int hh = 0; hh <3; hh++)
    {
      X_i = octants_dist_idx[0] * 0.5f * (hh -1) * (hh - 2) + octants_dist_idx[1]  * hh * (2 - hh) + octants_dist_idx[2]  * 0.5f * hh * (hh - 1);
      vol +=  M_PI / 4.0f * (radius*radius * X_i - X_i*X_i*X_i / 3.0f);
    }
    vol -= 2.0f * octante_normal_volumen ;
  }
  // # 2 direcctions are outside but both with flag 2
  else if(option == 4)
  {
    float dist_large, dist_short;
    if (flags_idx[0] == 0)
    {
      dist_large = max(octants_dist_idx[1], octants_dist_idx[2]);
      dist_short = min(octants_dist_idx[1], octants_dist_idx[2]);
    }
    else
    {
      if (flags_idx[1] == 0)
      {
        dist_large = max(octants_dist_idx[0], octants_dist_idx[2]);
        dist_short = min(octants_dist_idx[0], octants_dist_idx[2]);
      }
      else
      {
        dist_large = max(octants_dist_idx[0], octants_dist_idx[1]);
        dist_short = min(octants_dist_idx[0], octants_dist_idx[1]);
      }
    }

    float A = dist_large * dist_short * sqrtf(radius * radius - dist_large * dist_large);
    float B = 1.0f / 6.0f * M_PI / 2.0f * (sqrtf(radius * radius - dist_short * dist_short) * (2.0f * radius * radius + dist_short * dist_short) - dist_short * (dist_short * dist_short - 3.0f * radius * radius) - 2.0f * radius * radius * radius) - 1.0f / 6.0f * (2.0f * dist_short * sqrtf(radius * radius - dist_large * dist_large) * sqrtf(dist_large * dist_large - dist_short * dist_short) - sqrtf(radius * radius - dist_large * dist_large) * (-2.0f * radius * radius - dist_large * dist_large) * atanf(dist_short / sqrtf(dist_large * dist_large - dist_short * dist_short)) - dist_short * (dist_short * dist_short - 3.0f * radius * radius) * atanf(sqrtf(radius * radius - dist_large * dist_large) / sqrtf(dist_large * dist_large - dist_short * dist_short)) - 2.0f * radius * radius * radius * atanf(dist_short * sqrtf(radius * radius - dist_large * dist_large) / (radius * sqrtf(dist_large * dist_large - dist_short * dist_short))));
    float C = M_PI / 4.0f * (2.0f * radius * radius * radius / 3.0f - radius * radius * sqrtf(radius * radius - dist_short * dist_short) + sqrtf(radius * radius - dist_short * dist_short) * sqrtf(radius * radius - dist_short * dist_short) * sqrtf(radius * radius - dist_short * dist_short) / 3.0f);
    vol = A + B + C;
  }
  //# The three direcctions are with flag = 2
  else if(option == 6)
  {
    vol = octants_dist_idx[0] * octants_dist_idx[1] * octants_dist_idx[2];
  }
        
  return vol;
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////// N-DIMENSIONAL HELPER FUNCTIONS //////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

/**
 * Compute the volume of a D-dimensional hypersphere with given radius.
 * 
 * Formula: V_D(r) = (pi^(D/2) * r^D) / Gamma(D/2 + 1)
 * 
 * @param D      dimensionality
 * @param radius radius of the hypersphere
 * @return       volume of the hypersphere
 */
float compute_hypersphere_volume(size_t D, float radius)
{
  // Use lgamma for numerical stability with large D
  float log_volume = (D / 2.0f) * std::log(M_PI) + D * std::log(radius) - std::lgamma(D / 2.0f + 1.0f);
  return std::exp(log_volume);
}

/**
 * Compute the fraction of hypersphere volume that lies within the bounding box
 * using Monte Carlo sampling.
 * 
 * This is more accurate but slower, use for validation or when analytical
 * approximation is not sufficient.
 * 
 * @param point       center of the hypersphere
 * @param mins        minimum bounds for each dimension
 * @param maxs        maximum bounds for each dimension
 * @param radius      radius of the hypersphere
 * @param D           dimensionality
 * @param num_samples number of Monte Carlo samples
 * @return            fraction of volume within bounds (0 to 1)
 */
float compute_boundary_volume_ratio_monte_carlo(vector<float> const &point, 
                                                 vector<float> const &mins, 
                                                 vector<float> const &maxs, 
                                                 float radius, 
                                                 size_t D, 
                                                 size_t num_samples)
{
  // Thread-local random number generator for reproducibility
  static thread_local std::mt19937 gen(std::random_device{}());
  std::normal_distribution<float> normal_dist(0.0f, 1.0f);
  std::uniform_real_distribution<float> uniform_dist(0.0f, 1.0f);
  
  size_t inside_count = 0;
  
  for (size_t i = 0; i < num_samples; i++)
  {
    // Generate a random point uniformly in the hypersphere
    // Method: generate D normal random variables, normalize, then scale by random radius^(1/D)
    vector<float> direction(D);
    float norm = 0.0f;
    
    for (size_t d = 0; d < D; d++)
    {
      direction[d] = normal_dist(gen);
      norm += direction[d] * direction[d];
    }
    norm = std::sqrt(norm);
    
    // Scale to random radius (uniform in volume)
    float r = radius * std::pow(uniform_dist(gen), 1.0f / D);
    
    // Generate sample point
    bool inside_bounds = true;
    for (size_t d = 0; d < D && inside_bounds; d++)
    {
      float sample_coord = point[d] + r * direction[d] / norm;
      if (sample_coord < mins[d] || sample_coord > maxs[d])
      {
        inside_bounds = false;
      }
    }
    
    if (inside_bounds)
    {
      inside_count++;
    }
  }
  
  return static_cast<float>(inside_count) / static_cast<float>(num_samples);
}

/**
 * Compute an analytical approximation of the fraction of hypersphere volume 
 * that lies within the bounding box.
 * 
 * This uses a product of 1D spherical cap corrections for each dimension.
 * This is an approximation that works well when the clipping in different
 * dimensions is relatively independent (not too much corner clipping).
 * 
 * @param point  center of the hypersphere
 * @param mins   minimum bounds for each dimension
 * @param maxs   maximum bounds for each dimension
 * @param radius radius of the hypersphere
 * @param D      dimensionality
 * @return       fraction of volume within bounds (0 to 1)
 */
float compute_boundary_volume_ratio_analytical(vector<float> const &point, 
                                                vector<float> const &mins, 
                                                vector<float> const &maxs, 
                                                float radius, 
                                                size_t D)
{
  float ratio = 1.0f;
  
  for (size_t d = 0; d < D; d++)
  {
    float dist_to_min = point[d] - mins[d];
    float dist_to_max = maxs[d] - point[d];
    
    // Compute the fraction of volume remaining after clipping by each boundary
    // Using hyperspherical cap volume formula approximation
    
    // If fully inside, no clipping
    if (dist_to_min >= radius && dist_to_max >= radius)
    {
      continue;
    }
    
    // Clipping from minimum boundary
    if (dist_to_min < radius && dist_to_min >= 0)
    {
      // Fraction of 1D "diameter" that is clipped
      // For a hypersphere, the volume fraction depends on the cap height h = r - d
      float h = radius - dist_to_min;
      // Approximate cap volume fraction using regularized incomplete beta function approximation
      // For high D, this simplifies to approximately (h/2r)^(D/2) for small h
      // For a more accurate approximation: 0.5 * I_{(h/r)(2-h/r)}((D+1)/2, 0.5)
      // We use a simpler approximation: 0.5 * (1 - (1 - h/r)^((D+1)/2))
      float x = h / radius;
      float cap_fraction = 0.5f * std::pow(x, (D + 1.0f) / 2.0f);
      ratio *= (1.0f - cap_fraction);
    }
    else if (dist_to_min < 0)
    {
      // Center is outside the boundary on this side
      // Most of the sphere is outside
      float h = radius + dist_to_min;  // This is the height of the cap INSIDE
      if (h <= 0) return 0.0f;  // Completely outside
      float x = h / radius;
      float cap_fraction = 0.5f * std::pow(x, (D + 1.0f) / 2.0f);
      ratio *= cap_fraction;
    }
    
    // Clipping from maximum boundary
    if (dist_to_max < radius && dist_to_max >= 0)
    {
      float h = radius - dist_to_max;
      float x = h / radius;
      float cap_fraction = 0.5f * std::pow(x, (D + 1.0f) / 2.0f);
      ratio *= (1.0f - cap_fraction);
    }
    else if (dist_to_max < 0)
    {
      float h = radius + dist_to_max;
      if (h <= 0) return 0.0f;
      float x = h / radius;
      float cap_fraction = 0.5f * std::pow(x, (D + 1.0f) / 2.0f);
      ratio *= cap_fraction;
    }
  }
  
  // Clamp to valid range
  return std::max(0.0f, std::min(1.0f, ratio));
}