# Copyright (c) 2019 Ste||ar Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(HPX_WITH_EXAMPLES_QTHREADS)
  find_package(QThreads)
  if(NOT QTHREADS_FOUND)
    set(HPX_WITH_EXAMPLES_QTHREADS OFF)
  else()
    add_library(hpx::qthreads INTERFACE IMPORTED)
    set_property(TARGET hpx::qthreads PROPERTY
      INTERFACE_INCLUDE_DIRECTORIES ${QTHREADS_INCLUDE_DIRS})
    set_property(TARGET hpx::qthreads PROPERTY
      INTERFACE_LINK_LIBRARIES ${QTHREADS_LIBRARIES})
  endif()
endif()
