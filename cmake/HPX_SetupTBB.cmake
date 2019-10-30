# Copyright (c) 2019 Ste||ar Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(HPX_WITH_EXAMPLES_TBB)
  find_package(TBB)
  if(NOT TBB_FOUND)
    set(HPX_WITH_EXAMPLES_TBB OFF)
  else()
    add_library(hpx::tbb INTERFACE IMPORTED)
    set_property(TARGET hpx::tbb PROPERTY
      INTERFACE_INCLUDE_DIRECTORIES ${TBB_INCLUDE_DIRS})
    set_property(TARGET hpx::tbb PROPERTY
      INTERFACE_LINK_LIBRARIES ${TBB_LIBRARIES})
  endif()
endif()
