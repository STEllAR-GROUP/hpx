# Copyright (c) 2007-2021 Hartmut Kaiser
# Copyright (c) 2014-2015 Thomas Heller
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(parcelport_plugins)

if(HPX_WITH_NETWORKING)
  set(parcelport_plugins ${parcelport_plugins})
endif()

foreach(type ${parcelport_plugins})
  add_hpx_pseudo_target(plugins.parcelport.${type})
  add_subdirectory(${type})
  list(FIND HPX_STATIC_PARCELPORT_PLUGINS ${type} pos)
  if(NOT ${pos} EQUAL -1)
    target_link_libraries(hpx_full PRIVATE ${type})
  endif()
  add_hpx_pseudo_dependencies(plugins.parcelport plugins.parcelport.${type})
endforeach()
