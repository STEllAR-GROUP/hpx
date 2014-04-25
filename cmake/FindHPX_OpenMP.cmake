# Copyright (c) 2007-2012 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# This is a wrapper for FindOpenMP.cmake avoiding to re-detect OpenMP during
# each cmake configuration

if(OPENMP_DISABLE)
  hpx_info("find_package.openmp" "OpenMP disabled by user.")

  set(OPENMP_FOUND OFF
      CACHE BOOL "Found OpenMP.")
  set(OpenMP_CXX_FLAGS ""
      CACHE STRING "OpenMP flags for ${language}.")
else()
  # Explicitly disabling OpenMP for clang based compilers for now as it fails
  # to detect it correctly
  if(NOT OPENMP_SEARCHED AND NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(flag_candidates
        # GCC 
        "-fopenmp"
        # MSVC
        "/openmp"
        # Intel (Windows)
        "-Qopenmp" 
        # Intel (Unix)
        "-openmp" 
        # Empty, if compiler automatically accepts OpenMP
        " "
        # Sun
        "-xopenmp"
        # HP
        "+Oopenmp"
        # IBM 
        "-qsmp"
        # PGI
        "-mp"
        )

    hpx_include(AddConfigTest)

    foreach(flag_candidate ${flag_candidates}) 
      hpx_info("find_package.openmp" "Trying flag ${flag_candidate}.")

      add_hpx_config_test("openmp" OPENMP_FOUND LANGUAGE CXX
        SOURCE cmake/tests/openmp.cpp
        FLAGS ${flag_candidate} FILE)

      if(OPENMP_FOUND)
        hpx_info("find_package.openmp" "Flag found.")
        set(OpenMP_CXX_FLAGS ${flag_candidate})
        break()
      endif()

      set(OPENMP_FOUND DETECT)
    endforeach()

    if(NOT OPENMP_FOUND)
      hpx_warn("find_package.openmp" "OpenMP support not found.")
    endif()

    set(OPENMP_FOUND ${OPENMP_FOUND}
        CACHE BOOL "Found OpenMP.")
    set(OpenMP_CXX_FLAGS ${OpenMP_CXX_FLAGS}
        CACHE STRING "OpenMP link flags for ${language}.")

    set(OPENMP_SEARCHED ON CACHE INTERNAL "Searched for OpenMP.")
  endif()
endif()

