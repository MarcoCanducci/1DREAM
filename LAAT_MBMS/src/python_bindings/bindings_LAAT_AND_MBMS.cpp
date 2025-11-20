#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <iostream>
#include <fstream>
#include "LAAT/LAAT.h"
#include "MBMS/MBMS.h"

namespace py = pybind11;
using namespace py::literals;

/**
 * Binding function to use the LAAT algorithm from Python.
 *
 * The function expects a Numpy array containing the data, and returns a Numpy
 * array containing the pheromone resulting from LAAT.
 */
py::array LAAT(py::array_t<float> in,
	       size_t numberOfAnts,
	       size_t numberOfIterations,
	       size_t numberOfSteps,
				 size_t pso_number_particles,
				 float pso_min_radii,
				 float pso_max_radii,
				 size_t dynamic_radius_actived,
	       size_t th_neighbors,
	       float kappa,
	       size_t numberofthreads)
{
  auto buf = in.request();
  float *npData = static_cast<float *>(buf.ptr);
  size_t size = buf.shape[0];
  std::vector<std::vector<float>> data(size, std::vector<float>(3));

  for (size_t i = 0; i < size; ++i)
    for (size_t j = 0; j < 3; ++j)
      data[i][j] = npData[i + j * size];

  py::array ret = py::cast(LocallyAlignedAntTechnique(data,
						      numberOfAnts,
						      numberOfIterations,
						      numberOfSteps,
									pso_number_particles,
									pso_min_radii,
									pso_max_radii,
									dynamic_radius_actived,
						      th_neighbors,
						      kappa,
						      numberofthreads));
  return ret;
}

/**
 * Binding function to use the MBMS algorithm from Python.
 *
 * The function expects a Numpy array containing the data, and returns a Numpy
 * array containing updated data after MBMS has been applied.
 */
py::array MBMS(py::array_t<float> in,
	       size_t iter,
	       float radius,
	       float sigma,
	       size_t k)
{
  auto buf = in.request();
  float *npData = static_cast<float *>(buf.ptr);

  size_t size = buf.shape[0];
  std::vector<std::vector<float>> data(size, std::vector<float>(3));

  for (size_t i = 0; i < size; ++i)
    for (size_t j = 0; j < 3; ++j)
      //data[i][j] = npData[i * 3 + j];
	  data[i][j] = npData[i * 3 + j];

  manifoldBlurringMeanShift(data, iter, radius, sigma, k);

  return py::array(py::cast(data));
}

PYBIND11_MODULE(LAAT_MBMS, m)
{
  m.doc() = "LAAT_MBMS module for Python. Contains the LAAT and MBMS functions";

  m.def("LAAT", &LAAT, "Locally Aligned Ant Technique algorithm",
	"in"_a,
	"numberOfAnts"_a = 343,
	"numberOfIterations"_a = 100,
	"numberOfSteps"_a = 12000,
	"pso_number_particles"_a = 100,
	"pso_min_radii"_a = 1.0,
	"pso_max_radii"_a = 1.0,
	"dynamic_radius_actived"_a = 0,
	"th_neighbors"_a = 3,
	"kappa"_a = 0.8,
	"numberofthreads"_a = 1);

  m.def("MBMS", &MBMS, "Manifold Blurring Mean Shift algorithm",
	"in"_a,
	"iter"_a = 10,
	"radius"_a = 3,
	"sigma"_a = 1.5,
	"k"_a = 10);
}
