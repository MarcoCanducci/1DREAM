cmake_minimum_required(VERSION 3.10)

project(LAAT_MBMS VERSION 1.3.2)

# Threads
find_package(Threads REQUIRED)
# Eigen
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/ext/eigen")
# OpenMP
find_package(OpenMP REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Platform-specific compiler flags
if(MSVC)
    # MSVC flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /W3")
    # Use experimental OpenMP for min/max reductions
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp:llvm")
else()
    # GCC/Clang flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -fPIC")
endif()

add_library(LAAT_MBMSlib
    src/lib/LAAT/antsearch.cpp
		src/lib/LAAT/antsearch_DynamicRadius.cpp
		src/lib/LAAT/computing_initial_ant_probabilities.cpp
		src/lib/LAAT/computing_initial_particle_properties.cpp
    src/lib/LAAT/evaporatepheromone.cpp
    src/lib/LAAT/initializeants.cpp
    src/lib/LAAT/LAAT.cpp
    src/lib/LAAT/LAAT_MarkovChain.cpp
    src/lib/LAAT/preprocess.cpp
    src/lib/LAAT/progressbar.cpp
		src/lib/LAAT/random_generator.cpp
    src/lib/LAAT/rangesearch.cpp
    

    src/lib/MBMS/denoise.cpp
    src/lib/MBMS/dimensionalityapproximation.cpp
    src/lib/MBMS/kernelfunmean.cpp
    src/lib/MBMS/localpca.cpp
    src/lib/MBMS/MBMS.cpp
)

# add include directories to use when compiling the target
target_include_directories(LAAT_MBMSlib PRIVATE ext)
target_include_directories(LAAT_MBMSlib PRIVATE ext/nanoflann/include)
target_include_directories(LAAT_MBMSlib PUBLIC src/lib)

target_link_libraries(LAAT_MBMSlib Eigen3::Eigen OpenMP::OpenMP_CXX)

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/ext/pybind11")
pybind11_add_module(LAAT_MBMS src/python_bindings/bindings_LAAT_AND_MBMS.cpp)
target_link_libraries(LAAT_MBMS PRIVATE LAAT_MBMSlib)
