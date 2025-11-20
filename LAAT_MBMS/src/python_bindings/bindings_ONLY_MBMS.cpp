#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <iostream>
#include <fstream>
#include "MBMS/MBMS.h"

namespace py = pybind11;
using namespace py::literals;

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

  m.def("MBMS", &MBMS, "Manifold Blurring Mean Shift algorithm",
	"in"_a,
	"iter"_a = 10,
	"radius"_a = 3,
	"sigma"_a = 1.5,
	"k"_a = 10);
}
