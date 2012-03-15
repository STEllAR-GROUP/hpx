# Copyright (c) 2007-2012 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# This is a wrapper for FindMPI.cmake avoiding to re-detect MPI during each
# cmake configuration

set(languages CXX Fortran)

if(MPI_DISABLE)
  hpx_info("find_package.mpi" "Library disabled by user.")

  foreach(language ${languages})
    set(MPI_${language}_FOUND OFF
        CACHE BOOL "Found MPI.")
    set(MPI_${language}_COMPILER MPI_${language}_COMPILER-NOTFOUND
        CACHE FILEPATH "MPI compiler wrapper for ${language}.")
    set(MPI_${language}_COMPILE_FLAGS ""
        CACHE STRING "MPI compilation flags for ${language}.")
    set(MPI_${language}_INCLUDE_PATH MPI_${language}_INCLUDE_PATH-NOTFOUND
        CACHE FILEPATH "MPI include headers for ${language}.")
    set(MPI_${language}_LINK_FLAGS ""
        CACHE STRING "MPI link flags for ${language}.")
    set(MPI_${language}_LIBRARIES ""
        CACHE STRING "MPI libraries for ${language}.")
  endforeach()
else()
  if(NOT MPI_SEARCHED)
    set(MPI_FIND_QUIETLY TRUE)
    set(MPI_LIB_FIND_QUIETLY TRUE)
    set(MPI_HEADER_PATH_FIND_QUIETLY TRUE)
    set(MPIEXEC_FIND_QUIETLY TRUE)

    foreach(language ${languages})
      set(MPI_${language}_COMPILER_QUIETLY TRUE)
      set(MPI_${language}_FIND_QUIETLY TRUE)
    endforeach()
    
    find_package(MPI)

    if(MPI_FOUND)
      hpx_info("find_package.mpi" "Library found in system path.")
    else()
      hpx_warn("find_package.mpi" "Library not found in system path.")
    endif()

    foreach(language ${languages})
      set(MPI_${language}_FOUND ${MPI_${language}_FOUND}
          CACHE BOOL "Found MPI.")
      set(MPI_${language}_COMPILER ${MPI_${language}_COMPILER}
          CACHE FILEPATH "MPI compiler wrapper for ${language}.")
      set(MPI_${language}_COMPILE_FLAGS ${MPI_${language}_COMPILE_FLAGS}
          CACHE STRING "MPI compilation flags for ${language}.")
      set(MPI_${language}_INCLUDE_PATH ${MPI_${language}_INCLUDE_PATH}
          CACHE FILEPATH "MPI include headers for ${language}.")
      set(MPI_${language}_LINK_FLAGS ${MPI_${language}_LINK_FLAGS}
          CACHE STRING "MPI link flags for ${language}.")
      set(MPI_${language}_LIBRARIES ${MPI_${language}_LIBRARIES}
          CACHE STRING "MPI libraries for ${language}.")
    endforeach()

    set(MPI_SEARCHED ON CACHE INTERNAL "Searched for MPI library.")
  endif()
endif()

