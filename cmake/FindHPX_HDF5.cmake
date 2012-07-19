# Copyright (c) 2007-2012 Hartmut Kaiser
# Copyright (c) 2010-2011 Matt Anderson
# Copyright (c) 2011      Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT HPX_FINDPACKAGE_LOADED)
  include(HPX_FindPackage)
endif()

if(HDF5_USE_SYSTEM)
  set(HDF5_CPP_USE_SYSTEM ON)
  set(HDF5_FORTRAN_USE_SYSTEM ON)
endif()

if($ENV{HDF5_ROOT})
  if(NOT HDF5_CPP_ROOT)
    set(HDF5_CPP_ROOT $ENV{HDF5_ROOT})
  endif()
  if(NOT HDF5_FORTRAN_ROOT)
    set(HDF5_FORTRAN_ROOT $ENV{HDF5_ROOT})
  endif()
endif()

if(HDF5_ROOT)
  if(NOT HDF5_CPP_ROOT)
    set(HDF5_CPP_ROOT ${HDF5_ROOT})
  endif()
  if(NOT HDF5_FORTRAN_ROOT)
    set(HDF5_FORTRAN_ROOT ${HDF5_ROOT})
  endif()
endif()

# find main HDF5 library
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  set(hdf5_lib hdf5ddll hdf5dll)
else()
  set(hdf5_lib hdf5dll)
endif()

hpx_find_package(HDF5
  LIBRARIES hdf5 libhdf5 ${hdf5_lib}
  LIBRARY_PATHS lib64 lib
  HEADERS H5pubconf.h
  HEADER_PATHS include)

# find HDF5 C++ library
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  set(hdf5_cpp_lib hdf5_cppddll hdf5_cppdll)
else()
  set(hdf5_cpp_lib hdf5_cppdll)
endif()

hpx_find_package(HDF5_CPP
  LIBRARIES hdf5_cpp libhdf5_cpp ${hdf5_cpp_lib}
  LIBRARY_PATHS lib64 lib
  HEADERS H5Cpp.h
  HEADER_PATHS include include/cpp)

# find HDF5 Fortran library
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  set(hdf5_fortran_lib hdf5_fortranddll hdf5_fortrandll)
  set(hdf5_fortran_mod include/fortran/Debug include/fortran/Release)
else()
  set(hdf5_fortran_lib hdf5_fortrandll)
  set(hdf5_fortran_mod include/fortran/Release include/fortran/Debug)
endif()

hpx_find_package(HDF5_FORTRAN
  LIBRARIES hdf5_fortran libhdf5_fortran ${hdf5_fortran_lib}
  LIBRARY_PATHS lib64 lib
  HEADERS hdf5.mod
  HEADER_PATHS include include/fortran ${hdf5_fortran_mod})

