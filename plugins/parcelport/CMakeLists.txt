# Copyright (c) 2007-2017 Hartmut Kaiser
# Copyright (c) 2014-2015 Thomas Heller
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(parcelport_plugins)

if(HPX_WITH_NETWORKING)
  set(parcelport_plugins ${parcelport_plugins} libfabric mpi tcp)
endif()

set(HPX_STATIC_PARCELPORT_PLUGINS
    ""
    CACHE INTERNAL "" FORCE
)
set(HPX_STATIC_PARCELPORT_PLUGINS_HEADERS
    ""
    CACHE INTERNAL "" FORCE
)
set(HPX_STATIC_PARCELPORT_PLUGINS_SOURCES
    ""
    CACHE INTERNAL "" FORCE
)
set(HPX_STATIC_PARCELPORT_PLUGINS_PROPERTIES
    ""
    CACHE INTERNAL "" FORCE
)

foreach(type ${parcelport_plugins})
  add_hpx_pseudo_target(plugins.parcelport.${type})
  add_subdirectory(${type})
  add_hpx_pseudo_dependencies(plugins.parcelport plugins.parcelport.${type})
endforeach()

hpx_debug("creating static_parcelports.hpp: " ${HPX_STATIC_PARCELPORT_PLUGINS})

# add_subdirectory is called before to insure HPX_STATIC_PARCELPORT_PLUGINS
# cache variable is filled
set(_parcelport_export)
set(_parcelport_init)
foreach(parcelport ${HPX_STATIC_PARCELPORT_PLUGINS})
  set(_parcelport_export
      "${_parcelport_export}HPX_EXPORT hpx::plugins::parcelport_factory_base *${parcelport}_factory_init(\n"
  )
  set(_parcelport_export
      "${_parcelport_export}    std::vector<hpx::plugins::parcelport_factory_base *>& factories);\n"
  )
  set(_parcelport_init
      "${_parcelport_init}        ${parcelport}_factory_init(factories);\n"
  )
endforeach()

configure_file(
  "${PROJECT_SOURCE_DIR}/cmake/templates/static_parcelports.hpp.in"
  "${PROJECT_BINARY_DIR}/hpx/runtime/parcelset/static_parcelports.hpp" @ONLY
)

# ------------------------------------------------------------------------------
# Write parcelport_defines.hpp
# ------------------------------------------------------------------------------
# write config to build dir
write_config_defines_file(
  TEMPLATE ""
  NAMESPACE "parcelport"
  FILENAME "${PROJECT_BINARY_DIR}/hpx/config/parcelport_defines.hpp"
)

# write config to install dir
write_config_defines_file(
  TEMPLATE ""
  NAMESPACE "parcelport"
  FILENAME
    "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/hpx/config/parcelport_defines.hpp"
)
