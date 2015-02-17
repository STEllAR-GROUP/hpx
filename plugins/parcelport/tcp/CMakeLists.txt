# Copyright (c) 2007-2013 Hartmut Kaiser
# Copyright (c) 2014-2015 Thomas Heller
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

include(HPX_AddLibrary)

if(HPX_PARCELPORT_TCP)
  hpx_add_config_define(HPX_PARCELPORT_TCP)

  macro(add_parcelport_tcp_module)
    hpx_debug("add_parcelport_tcp_module")
    add_parcelport(
        tcp
        STATIC
        SOURCES "${hpx_SOURCE_DIR}/plugins/parcelport/tcp/connection_handler_tcp.cpp"
                "${hpx_SOURCE_DIR}/plugins/parcelport/tcp/parcelport_tcp.cpp"
        HEADERS
              "${hpx_SOURCE_DIR}/hpx/plugins/parcelport/tcp/connection_handler.hpp"
              "${hpx_SOURCE_DIR}/hpx/plugins/parcelport/tcp/locality.hpp"
              "${hpx_SOURCE_DIR}/hpx/plugins/parcelport/tcp/receiver.hpp"
              "${hpx_SOURCE_DIR}/hpx/plugins/parcelport/tcp/sender.hpp"
        FOLDER "Core/Plugins/Parcelport/Tcp"
        )
  endmacro()
else()
  macro(add_parcelport_tcp_module)
  endmacro()
endif()
