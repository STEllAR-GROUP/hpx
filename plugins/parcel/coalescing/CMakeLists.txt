# Copyright (c) 2007-2016 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

include(HPX_AddLibrary)

macro(add_coalescing_module)
  if(HPX_WITH_PARCEL_COALESCING)
    hpx_debug("add_coalescing_module")
    add_hpx_library(parcel_coalescing
        PLUGIN
        SOURCES "${PROJECT_SOURCE_DIR}/plugins/parcel/coalescing/coalescing_message_handler.cpp"
        HEADERS
              "${PROJECT_SOURCE_DIR}/hpx/plugins/parcel/coalescing_message_handler.hpp"
              "${PROJECT_SOURCE_DIR}/hpx/plugins/parcel/message_buffer.hpp"
        FOLDER "Core/Plugins/MessageHandler"
        )

    add_hpx_pseudo_dependencies(plugins.parcel.coalescing parcel_coalescing_lib)
    add_hpx_pseudo_dependencies(core plugins.parcel.coalescing)
  endif()
endmacro()

