# Copyright (c) 2007-2013 Hartmut Kaiser
# Copyright (c) 2014-2015 Thomas Heller
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

include(HPX_AddLibrary)

################################################################################
# Decide whether to use the VERBS based parcelport
################################################################################
if(HPX_PARCELPORT_MINIMAL)
  hpx_add_config_define(HPX_PARCELPORT_MINIMAL)

  macro(add_parcelport_minimal_module)
    hpx_debug("add_parcelport_minimal_module")

    add_parcelport(minimal
      STATIC
      SOURCES
        "${hpx_SOURCE_DIR}/plugins/parcelport/minimal/parcelport_minimal.cpp"
      HEADERS
      DEPENDENCIES
      FOLDER "Core/Plugins/Parcelport/minimal")
  endmacro()
else()
  macro(add_parcelport_minimal_module)
  endmacro()
endif()
