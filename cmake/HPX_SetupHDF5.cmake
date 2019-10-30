# Copyright (c) 2019 Ste||ar Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(HPX_WITH_EXAMPLES_HDF5)
  find_package(HDF5 COMPONENTS CXX)
  if(NOT HDF5_FOUND)
    set(HPX_WITH_EXAMPLES_HDF5 OFF)
  else()
    add_library(hpx::hdf5 INTERFACE IMPORTED)
    set_property(TARGET hpx::hdf5 PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${HDF5_INCLUDE_DIR})
    set_property(TARGET hpx::hdf5 PROPERTY INTERFACE_LINK_LIBRARIES
      ${HDF5_LIBRARIES})
  endif()
endif()
