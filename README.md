# 1DREAM

1DREAM is a toolbox composed of five main Machine Learning methodologies for the detection and modeling of 1D filament-like structures demonstrated on astronomical applications (https://doi.org/10.1016/j.ascom.2022.100658). The methodologies are:

## 1.- Locally Aligned Ant Technique (LAAT):
	Remove the noise from the data set (https://doi.org/10.48550/arXiv.2009.08326)
	
## 2.- Evolutionary Manifold Alignment Aware Agents (EM3A): 
	Moves the particles towards the backbone of the filamentary structure (https://doi.org/10.1162/neco_a_01478).
	
## 3.- Dimensionality Index (DimIndex): 
	Assign to the particles their intrinsic dimension (https://doi.org/10.1016/j.artint.2021.103579).
	
## 4.- Multi-Manifold Crawling (Crawling):
	operates on the one-dimensional partition. Discover the filaments and constructs their corresponding skeletons (https://doi.org/10.1016/j.artint.2021.103579).
	
## 5.- Stream Generative Topographic Mapping (SGTM)
	Builds a probabilistic model for each extracted sub-structure, describing the transverse noise distribution along the manifold itself as a constrained Gaussian mixture model (https://doi.org/10.1016/j.artint.2021.103579).
	
 
# Installation

The C++ LAAT version and the MBMS code requires be installed. To install this module from source, you will need to install a suitable C++ compiler and CMake. Any other external dependencies of this project are included as git submodules and they must be initialized when this repository is cloned:

```sh
git clone https://git.lwp.rug.nl/cs.projects/1DREAM.git
cd 1DREAM
git submodule update --init
```

The submodules used are Eigen and Pybind11. Both are installed in the path "1DREAM/LAAT_MBMS/ext/submodulename"

Enter to the LAAT_MBMS folder.

```sh
cd LAAT_MBMS
```

Copy the 'CMakeLists.txt' file from 'python3_library_CMakeLists' folder to the 'LAAT_MBMS' folder

LINUX:
```sh
cp python_library_CMakeLists/CMakeLists_LAAT_AND_MBMS.txt CMakeLists.txt
```

WINDOWS:

```sh
Copy-Item "$PWD/python_library_CMakeLists/CMakeLists_ONLY_MBMS.txt" CMakeLists.txt
```

The project can then be installed as a Python module from the main directory of the project using pip:

```
pip install .
```

Note that for WINDOWS we only install the MBMS library, because CMake has problems finding 'OpenMP_CXX', which is necessary for 
the parallelization of LAAT.
# NOTE: For LAAT installation in WINDOWS, follows the readme in '1DREAM/LAAT_MBMS/pure_cpp_LAAT/PURE_CPP_LAAT_README.txt'

The LAAT and MBMS modules can now be imported and used in python:

```python
import LAAT_MBMS
pheromone = LAAT_MBMS.LAAT(...)
```

################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################

#Note:
1.- Some users of Linux require superuser permission (sudo) to install as well as other git repositories. We are working on solving that issue.


################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################


## DEVELOPERS: Compiling from source
You can choose to build this module from source as a static C++ library for use in C++ projects. To do so, create a build directory and run CMake:

```sh
cd LAAT_MBMS
mkdir build
cd build
cmake ..
make
```

Now the build directory will create both the static C++ library file `libLAAT_MBMSlib.a` and the Python module binary file `LAAT_MBMS.cpython`.


################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################


# How to cite this work
The manuscript associated with this software has been pusblished on 2022.
DOI: https://doi.org/10.1016/j.ascom.2022.100658


################################################################################################################################

This branch of the 1DREAM code is dedicated to implement two main variations to the original:
## 1. LAAT : Initialization of ants positions within a pool of candidates, selected by ID;
## 2. LAAT : Additional criterion for the construction of jump probabilities. Ants can be informed by:
	a. local alignment (original - PCA);
	b. density (original);
	c. external field, provided by user (NEW!)
