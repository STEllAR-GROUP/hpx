# Copyright (c) 2024-2025 Louisiana State University
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0.

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(hpxpy VERSION 0.1.0 LANGUAGES CXX)

# ------------------------------------------------------------------------------
# C++ Standard
# ------------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ------------------------------------------------------------------------------
# Build type
# ------------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ------------------------------------------------------------------------------
# Options
# ------------------------------------------------------------------------------
option(HPXPY_WITH_TESTS "Build tests" ON)
option(HPXPY_WITH_BENCHMARKS "Build benchmarks" ON)
option(HPXPY_WITH_CUDA "Enable CUDA support" OFF)
option(HPXPY_WITH_SYCL "Enable SYCL support" OFF)

# ------------------------------------------------------------------------------
# Find dependencies
# ------------------------------------------------------------------------------

# Find HPX
find_package(HPX REQUIRED)
if(NOT HPX_FOUND)
  message(FATAL_ERROR "HPX not found. Please set HPX_DIR to the HPX installation.")
endif()
message(STATUS "Found HPX: ${HPX_VERSION} at ${HPX_DIR}")

# Find Python and pybind11
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module NumPy)
message(STATUS "Found Python: ${Python_VERSION} at ${Python_EXECUTABLE}")
message(STATUS "Found NumPy: ${Python_NumPy_VERSION} at ${Python_NumPy_INCLUDE_DIRS}")

# Find pybind11 - try config mode first, then module mode
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
  # Try to find via Python
  execute_process(
    COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE pybind11_result
  )
  if(pybind11_result EQUAL 0)
    find_package(pybind11 CONFIG REQUIRED HINTS ${pybind11_DIR})
  else()
    message(FATAL_ERROR "pybind11 not found. Install with: pip install pybind11")
  endif()
endif()
message(STATUS "Found pybind11: ${pybind11_VERSION}")

# ------------------------------------------------------------------------------
# HPXPy Core Extension Module
# ------------------------------------------------------------------------------
pybind11_add_module(_core MODULE
  src/bindings/core_module.cpp
  src/bindings/runtime_bindings.cpp
  src/bindings/array_bindings.cpp
  src/bindings/algorithm_bindings.cpp
)

target_include_directories(_core PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${Python_NumPy_INCLUDE_DIRS}
)

target_link_libraries(_core PRIVATE
  HPX::hpx
  HPX::wrap_main
)

target_compile_definitions(_core PRIVATE
  HPXPY_VERSION="${PROJECT_VERSION}"
  HPXPY_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
  HPXPY_VERSION_MINOR=${PROJECT_VERSION_MINOR}
  HPXPY_VERSION_PATCH=${PROJECT_VERSION_PATCH}
)

# Enable position independent code
set_target_properties(_core PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
)

# Compiler warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

# ------------------------------------------------------------------------------
# GPU Support (Optional)
# ------------------------------------------------------------------------------
if(HPXPY_WITH_CUDA)
  find_package(HPX REQUIRED COMPONENTS cuda)
  find_package(CUDAToolkit REQUIRED)

  target_sources(_core PRIVATE
    src/bindings/gpu_bindings.cpp
  )

  target_compile_definitions(_core PRIVATE HPXPY_HAVE_CUDA)
  target_link_libraries(_core PRIVATE CUDA::cudart)

  message(STATUS "CUDA support enabled")
endif()

if(HPXPY_WITH_SYCL)
  find_package(HPX REQUIRED COMPONENTS sycl)

  target_sources(_core PRIVATE
    src/bindings/sycl_bindings.cpp
  )

  target_compile_definitions(_core PRIVATE HPXPY_HAVE_SYCL)

  message(STATUS "SYCL support enabled")
endif()

# ------------------------------------------------------------------------------
# Installation
# ------------------------------------------------------------------------------
# Install the extension module to the hpxpy package directory
install(TARGETS _core
  LIBRARY DESTINATION hpxpy
  RUNTIME DESTINATION hpxpy
)

# Install Python package files
install(DIRECTORY hpxpy/
  DESTINATION hpxpy
  FILES_MATCHING PATTERN "*.py"
)

# ------------------------------------------------------------------------------
# Testing
# ------------------------------------------------------------------------------
if(HPXPY_WITH_TESTS)
  enable_testing()

  # Add a test that runs pytest
  add_test(
    NAME pytest
    COMMAND ${Python_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/tests -v
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )

  # Make sure the extension module can be found
  set_tests_properties(pytest PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}:${CMAKE_CURRENT_SOURCE_DIR}"
  )
endif()

# ------------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------------
message(STATUS "")
message(STATUS "HPXPy ${PROJECT_VERSION} Configuration Summary")
message(STATUS "================================")
message(STATUS "  HPX version:      ${HPX_VERSION}")
message(STATUS "  Python version:   ${Python_VERSION}")
message(STATUS "  NumPy version:    ${Python_NumPy_VERSION}")
message(STATUS "  pybind11 version: ${pybind11_VERSION}")
message(STATUS "  Build type:       ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA support:     ${HPXPY_WITH_CUDA}")
message(STATUS "  SYCL support:     ${HPXPY_WITH_SYCL}")
message(STATUS "  Build tests:      ${HPXPY_WITH_TESTS}")
message(STATUS "")
