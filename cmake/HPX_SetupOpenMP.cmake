# Copyright (c) 2019 Ste||ar Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(HPX_WITH_EXAMPLES_OPENMP)
  find_package(OpenMP)
  if(NOT OPENMP_FOUND)
    set(HPX_WITH_EXAMPLES_OPENMP OFF)
  else()
    add_library(hpx::openmp INTERFACE IMPORTED)
    set_property(TARGET hpx::openmp PROPERTY
      INTERFACE_INCLUDE_DIRECTORIES ${OPENMP_INCLUDE_DIRS})
    set_property(TARGET hpx::openmp PROPERTY
      INTERFACE_LINK_LIBRARIES ${OPENMP_LIBRARIES})
  endif()
endif()
