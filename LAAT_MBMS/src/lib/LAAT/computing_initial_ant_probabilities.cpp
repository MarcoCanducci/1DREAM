#include "LAAT.h"

// auxiliary function to find the median of a vector
void initializing_octants_and_flags(vector<vector<float>>  &octants_dist, vector<vector<int>>  &flags, float pos_x, float pos_y, float pos_z, float xmin, float xmax, float ymin, float ymax, float zmin, float zmax, float radius);
float compute_octante_volumen(vector<float>  &octants_dist_idx, vector<int>  &flags_idx, float radius);

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

  //Computing the standard deviation
  
  //Mean values
  //Computing the densities (Note that the boundary points has a different density than the interior points)
  vector<float> densities(data.size()); 

  float xmin = data[0][0], xmax = data[0][0];
  float ymin = data[1][0], ymax = data[1][0];
  float zmin = data[2][0], zmax = data[2][0];

#pragma omp parallel for reduction(min : xmin, ymin, zmin) reduction(max : xmax, ymax, zmax)
  for (size_t idx = 1; idx < data.size(); idx++)
  {
    xmin = std::min(xmin, data[idx][0]);
    xmax = std::max(xmax, data[idx][0]);
    ymin = std::min(ymin, data[idx][1]);
    ymax = std::max(ymax, data[idx][1]);
    zmin = std::min(zmin, data[idx][2]);
    zmax = std::max(zmax, data[idx][2]);
  }

//Neighbourhood Density computation
#pragma omp parallel
  {
    vector<vector<int>> flags(8, vector<int>(3));
    vector<vector<float>> octants_dist(8, vector<float>(3));

  float total_vol;
  #pragma omp for schedule(dynamic,10)
    for (size_t idx = 0; idx < data.size(); idx++)
    {
      total_vol = 0.0f;
      initializing_octants_and_flags(octants_dist, flags, data[idx][0], data[idx][1], data[idx][2], xmin, xmax, ymin, ymax, zmin, zmax, pso_max_radii);
      for (size_t hh = 0; hh < 8; hh++)
      {
        total_vol += compute_octante_volumen(octants_dist[hh], flags[hh], pso_max_radii);
      }



    //   densities[idx] = (float)sizes[idx] / total_vol;

	  densities[idx] = (float)neighbourhoods[idx].size() / total_vol;
    }
  }

  float probability_std_sum = 0.0f;
  float mean;

  vector<pair<float,size_t>> probability_std_aux(data.size());
  size_t aux_counter = 0;
#pragma omp parallel for reduction(+ : probability_std_sum, aux_counter) private(mean) schedule(dynamic,10)
  for (size_t idx = 0; idx < data.size(); idx++)
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
  for (size_t idx = 0; idx < probability_std.size(); idx++)
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