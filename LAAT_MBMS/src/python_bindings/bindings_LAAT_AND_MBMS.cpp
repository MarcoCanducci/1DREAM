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
 * The function expects a Numpy array containing the data (N x D), and returns a Numpy
 * array containing the pheromone resulting from LAAT.
 * 
 * Now supports N-dimensional data (any number of columns).
 * Data should be passed in row-major (C) order with shape (N, D).
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
               float gamma,
               py::array_t<float> external_weights_in,
               size_t initialization_mode,
               py::array_t<size_t> custom_init_indices_in,
               size_t numberofthreads)
{
  auto buf = in.request();
  
  // Validate input dimensions
  if (buf.ndim != 2) {
    throw std::runtime_error("Input data must be 2-dimensional (N x D)");
  }
  
  float *npData = static_cast<float *>(buf.ptr);
  size_t size = buf.shape[0];  // Number of data points
  size_t ndim = buf.shape[1];  // Number of dimensions
  
  // Create data vector with dynamic dimensions
  std::vector<std::vector<float>> data(size, std::vector<float>(ndim));

  // Copy data - support both row-major (C) and column-major (Fortran) order
  // Check if array is Fortran-contiguous (column-major)
  bool is_fortran = (buf.strides[0] == sizeof(float)) && 
                    (buf.strides[1] == static_cast<ssize_t>(size * sizeof(float)));
  
  if (is_fortran) {
    // Column-major (Fortran) order: data[i][j] = npData[i + j * size]
    for (size_t i = 0; i < size; ++i)
      for (size_t j = 0; j < ndim; ++j)
        data[i][j] = npData[i + j * size];
  } else {
    // Row-major (C) order: data[i][j] = npData[i * ndim + j]
    for (size_t i = 0; i < size; ++i)
      for (size_t j = 0; j < ndim; ++j)
        data[i][j] = npData[i * ndim + j];
  }

  // Convert external_weights numpy array to std::vector
  // If not provided, initialize with all ones (neutral weighting)
  std::vector<float> external_weights;
  auto buf_ext = external_weights_in.request();
  
  if (buf_ext.size > 0) {
    float *npExtWeights = static_cast<float *>(buf_ext.ptr);
    size_t ext_size = buf_ext.shape[0];
    external_weights.resize(ext_size);
    
    for (size_t i = 0; i < ext_size; ++i)
      external_weights[i] = npExtWeights[i];
  } else {
    // Default: all weights are 1.0 (neutral)
    external_weights.resize(size, 1.0f);
  }

  // Convert custom_init_indices numpy array to std::vector
  // If not provided, will use standard initialization
  std::vector<size_t> custom_init_indices;
  auto buf_init = custom_init_indices_in.request();
  
  if (buf_init.size > 0) {
    size_t *npInitIndices = static_cast<size_t *>(buf_init.ptr);
    size_t init_size = buf_init.shape[0];
    custom_init_indices.resize(init_size);
    
    for (size_t i = 0; i < init_size; ++i)
      custom_init_indices[i] = npInitIndices[i];
  }

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
                                                      gamma,
                                                      external_weights,
                                                      initialization_mode,
                                                      custom_init_indices,
                                                      numberofthreads));
  return ret;
}

/**
 * Binding function to use the MBMS algorithm from Python.
 *
 * The function expects a Numpy array containing the data (N x D), and returns a Numpy
 * array containing updated data after MBMS has been applied.
 * 
 * Now supports N-dimensional data (any number of columns).
 */
py::array MBMS(py::array_t<float> in,
               size_t iter,
               float radius,
               float sigma,
               size_t k)
{
  auto buf = in.request();
  
  // Validate input dimensions
  if (buf.ndim != 2) {
    throw std::runtime_error("Input data must be 2-dimensional (N x D)");
  }
  
  float *npData = static_cast<float *>(buf.ptr);
  size_t size = buf.shape[0];  // Number of data points
  size_t ndim = buf.shape[1];  // Number of dimensions
  
  std::vector<std::vector<float>> data(size, std::vector<float>(ndim));

  // Copy data in row-major order
  for (size_t i = 0; i < size; ++i)
    for (size_t j = 0; j < ndim; ++j)
      data[i][j] = npData[i * ndim + j];

  manifoldBlurringMeanShift(data, iter, radius, sigma, k);

  return py::array(py::cast(data));
}

/**
 * Binding function for LAAT Markov Chain approximation.
 * Computes the stationary distribution of the transition matrix.
 */
py::array LAAT_MarkovChain(py::array_t<float> in,
                           size_t th_neighbors,
                           float radius,
                           float kappa,
                           float gamma,
                           py::array_t<float> external_weights_in,
                           float tolerance,
                           size_t max_iterations,
                           size_t numberofthreads)
{
    auto buf = in.request();
    
    if (buf.ndim != 2) {
        throw std::runtime_error("Input data must be 2-dimensional (N x D)");
    }
    
    float *npData = static_cast<float *>(buf.ptr);
    size_t size = buf.shape[0];
    size_t ndim = buf.shape[1];
    
    std::vector<std::vector<float>> data(size, std::vector<float>(ndim));

    bool is_fortran = (buf.strides[0] == sizeof(float)) && 
                      (buf.strides[1] == static_cast<ssize_t>(size * sizeof(float)));
    
    if (is_fortran) {
        for (size_t i = 0; i < size; ++i)
            for (size_t j = 0; j < ndim; ++j)
                data[i][j] = npData[i + j * size];
    } else {
        for (size_t i = 0; i < size; ++i)
            for (size_t j = 0; j < ndim; ++j)
                data[i][j] = npData[i * ndim + j];
    }

    // Convert external_weights
    std::vector<float> external_weights;
    auto buf_ext = external_weights_in.request();
    
    if (buf_ext.size > 0) {
        float *npExtWeights = static_cast<float *>(buf_ext.ptr);
        size_t ext_size = buf_ext.shape[0];
        external_weights.resize(ext_size);
        for (size_t i = 0; i < ext_size; ++i)
            external_weights[i] = npExtWeights[i];
    } else {
        external_weights.resize(size, 1.0f);
    }

    py::array ret = py::cast(LocallyAlignedAntTechnique_MarkovChain(
        data,
        th_neighbors,
        radius,
        kappa,
        gamma,
        external_weights,
        tolerance,
        max_iterations,
        numberofthreads));
    
    return ret;
}

PYBIND11_MODULE(LAAT_MBMS, m)
{
  m.doc() = "LAAT_MBMS module for Python. Contains the LAAT and MBMS functions. Supports N-dimensional data.";

  m.def("LAAT", &LAAT, "Locally Aligned Ant Technique algorithm (supports N-dimensional data)",
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
        "gamma"_a = 0.0,
        "external_weights"_a = py::array_t<float>(),
        "initialization_mode"_a = 0,
        "custom_init_indices"_a = py::array_t<size_t>(),
        "numberofthreads"_a = 1);

  m.def("MBMS", &MBMS, "Manifold Blurring Mean Shift algorithm (supports N-dimensional data)",
        "in"_a,
        "iter"_a = 10,
        "radius"_a = 3,
        "sigma"_a = 1.5,
        "k"_a = 10);

  m.def("LAAT_MarkovChain", &LAAT_MarkovChain, 
        "LAAT Markov Chain approximation - computes stationary distribution as pheromone approximation",
        "in"_a,
        "th_neighbors"_a = 3,
        "radius"_a = 1.0,
        "kappa"_a = 0.8,
        "gamma"_a = 0.0,
        "external_weights"_a = py::array_t<float>(),
        "tolerance"_a = 1e-6,
        "max_iterations"_a = 1000,
        "numberofthreads"_a = 1);
}
