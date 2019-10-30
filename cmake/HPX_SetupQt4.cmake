# Copyright (c) 2019 Ste||ar Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(HPX_WITH_EXAMPLES_QT4)
  find_package(Qt4)
  if(NOT QT4_FOUND)
    set(HPX_WITH_EXAMPLES_QT4 OFF)
  else()
    add_library(hpx::qt4 INTERFACE IMPORTED)
    set_property(TARGET hpx::qt4 PROPERTY
      INTERFACE_INCLUDE_DIRECTORIES ${QT4_INCLUDE_DIRS})
    set_property(TARGET hpx::qt4 PROPERTY
      INTERFACE_LINK_LIBRARIES ${QT4_LIBRARIES})
  endif()
endif()
