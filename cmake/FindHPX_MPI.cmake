# Copyright (c) 207-2012 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# FIXME: This should use the standard HPX macros for an exhaustive search.

# This is a wrapper for FindMPI.cmake avoiding to re-detect MPI during each
# cmake configuration

if(MPI_DISABLE)
  hpx_info("hpx_find_package.MPI" "Library disabled by user.")
  set(MPI_FOUND OFF CACHE BOOL "Found ${name}.")
  set(MPI_LIBRARY MPI_LIBRARY-NOTFOUND CACHE FILEPATH "MPI library to link against.")
  set(MPI_EXTRA_LIBRARY MPI_EXTRA_LIBRARY-NOTFOUND CACHE FILEPATH "MPI extra libraries to link against.")
else()
  if(NOT MPI_SEARCHED)
    find_package(MPI)
    set(MPI_SEARCHED  ON CACHE INTERNAL "Searched for MPI library.")
  endif()
endif()

