# Copyright (c) 2007-2013 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

include(HPX_AddLibrary)

hpx_option(HPX_USE_PARCEL_COALESCING BOOL "Enable plugin for basic coalescing of parcel data (default: OFF)." OFF ADVANCED)

macro(add_coalescing_module)
  hpx_debug("add_coalescing_module" "HPX_USE_PARCEL_COALESCING: ${HPX_USE_PARCEL_COALESCING}")
  if(HPX_USE_PARCEL_COALESCING)
    add_hpx_library(parcel_coalescing
      SOURCES ${hpx_SOURCE_DIR}/plugins/parcel/coalescing/coalescing_message_handler.cpp
      HEADERS ${hpx_SOURCE_DIR}/hpx/plugins/parcel/coalescing_message_handler.hpp
      FOLDER "Core/Plugins/MessageHandler")

    set_property(TARGET parcel_coalescing_lib APPEND
      PROPERTY COMPILE_DEFINITIONS
      "HPX_LIBRARY_EXPORTS"
      "HPX_PLUGIN_NAME=parcel_coalescing")

    add_hpx_pseudo_dependencies(plugins.parcel.coalescing parcel_coalescing_lib)
  endif()
endmacro()

