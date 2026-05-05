cmake_minimum_required(VERSION 3.18)
project(aerial_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimisation flags
add_compile_options(-O3 -march=native -ffast-math)

# Eigen
find_package(Eigen3 REQUIRED NO_MODULE)

# Force the Python that matches our venv before pybind11 searches
set(PYTHON_EXECUTABLE "${CMAKE_CURRENT_SOURCE_DIR}/../.venv/bin/python3" CACHE FILEPATH "" FORCE)

# pybind11
find_package(pybind11 REQUIRED)

# BA extension
pybind11_add_module(_ba_cpp
    ba/ba.cpp
    ba/bind.cpp
)
target_include_directories(_ba_cpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/ba)
target_link_libraries(_ba_cpp PRIVATE Eigen3::Eigen)
set_target_properties(_ba_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../src/sfm
)

# Depth parallel extension
pybind11_add_module(_depth_cpp
    depth/depth_parallel.cpp
)
target_link_libraries(_depth_cpp PRIVATE Eigen3::Eigen)
set_target_properties(_depth_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../src/mvs
)
