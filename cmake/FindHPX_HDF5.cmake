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
  set(HDF5_CPP_ROOT $ENV{HDF5_ROOT})
  set(HDF5_FORTRAN_ROOT $ENV{HDF5_ROOT})
endif()

if(HDF5_ROOT)
  set(HDF5_CPP_ROOT ${HDF5_ROOT})
  set(HDF5_FORTRAN_ROOT ${HDF5_ROOT})
endif()

hpx_find_package(HDF5
  LIBRARIES hdf5 libhdf5 hdf5dll
  LIBRARY_PATHS lib64 lib
  HEADERS H5pubconf.h
  HEADER_PATHS include)

hpx_find_package(HDF5_CPP
  LIBRARIES hdf5_cpp libhdf5_cpp hdf5_cppdll
  LIBRARY_PATHS lib64 lib
  HEADERS H5Cpp.h
  HEADER_PATHS include include/cpp)

hpx_find_package(HDF5_FORTRAN
  LIBRARIES hdf5_fortran libhdf5_fortran hdf5_fortrandll
  LIBRARY_PATHS lib64 lib
  HEADERS hdf5.mod
  HEADER_PATHS include)

