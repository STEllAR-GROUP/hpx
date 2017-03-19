# Copyright (c) 2007-2017 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(parcel_plugins)

if(HPX_WITH_NETWORKING)
  set(parcel_plugins ${parcel_plugins}
    coalescing)
endif()

foreach(type ${parcel_plugins})
  add_hpx_pseudo_target(plugins.parcel.${type})
  add_subdirectory(${type})
  add_hpx_pseudo_dependencies(plugins.parcel plugins.parcel.${type})
endforeach()

macro(add_parcel_modules)
  if(HPX_WITH_NETWORKING)
    add_coalescing_module()
  endif()
endmacro()
