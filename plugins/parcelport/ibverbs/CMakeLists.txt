# Copyright (c) 2007-2013 Hartmut Kaiser
# Copyright (c) 2014-2015 Thomas Heller
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

include(HPX_AddLibrary)

################################################################################
# Decide whether to use the MPI based parcelport
################################################################################
if(HPX_WITH_PARCELPORT_IBVERBS)
  find_package(Ibverbs)
  if(NOT IBVERBS_FOUND)
    hpx_error("The ibverbs libraries could not be found and HPX_WITH_PARCELPORT_IBVERBS=ON, please specify IBVERBS_ROOT to point to the correct location or set HPX_PARCELPORT_IBVERBS to OFF")
  endif()
  include_directories(${IBVERBS_INCLUDE_DIR})

  find_package(Rdmacm)
  if(NOT RDMACM_FOUND)
    hpx_error("The ibverbs libraries could not be found and HPX_WITH_PARCELPORT_IBVERBS=ON, please specify RDMACM_ROOT to point to the correct location or set HPX_PARCELPORT_IBVERBS to OFF")
  endif()
  include_directories(${RDMACM_INCLUDE_DIR})
  hpx_add_config_define(HPX_HAVE_PARCELPORT_IBVERBS)
  hpx_add_config_define(HPX_HAVE_PARCELPORT_IBVERBS_IFNAME "\"${HPX_WITH_PARCELPORT_IBVERBS_IFNAME}\"")
  hpx_add_config_define(HPX_HAVE_PARCELPORT_IBVERBS_MESSAGE_PAYLOAD "${HPX_WITH_PARCELPORT_IBVERBS_MESSAGE_PAYLOAD}")
  hpx_add_config_define(HPX_HAVE_PARCELPORT_IBVERBS_MEMORY_CHUNK_SIZE "${HPX_WITH_PARCELPORT_IBVERBS_MEMORY_CHUNK_SIZE}")
  hpx_add_config_define(HPX_HAVE_PARCELPORT_IBVERBS_MAX_MEMORY_CHUNKS "${HPX_WITH_PARCELPORT_IBVERBS_MAX_MEMORY_CHUNKS}")

  macro(add_parcelport_ibverbs_module)
    hpx_debug("add_parcelport_ibverbs_module")
    add_parcelport(ibverbs
        SOURCES "${PROJECT_SOURCE_DIR}/plugins/parcelport/ibverbs/connection_handler_ibverbs.cpp"
                "${PROJECT_SOURCE_DIR}/plugins/parcelport/ibverbs/parcelport_ibverbs.cpp"
        HEADERS
              "${PROJECT_SOURCE_DIR}/hpx/plugins/parcelport/ibverbs/acceptor.hpp"
              "${PROJECT_SOURCE_DIR}/hpx/plugins/parcelport/ibverbs/allocator.hpp"
              "${PROJECT_SOURCE_DIR}/hpx/plugins/parcelport/ibverbs/client.hpp"
              "${PROJECT_SOURCE_DIR}/hpx/plugins/parcelport/ibverbs/connection_handler.hpp"
              "${PROJECT_SOURCE_DIR}/hpx/plugins/parcelport/ibverbs/context.hpp"
              "${PROJECT_SOURCE_DIR}/hpx/plugins/parcelport/ibverbs/data_buffer.hpp"
              "${PROJECT_SOURCE_DIR}/hpx/plugins/parcelport/ibverbs/helper.hpp"
              "${PROJECT_SOURCE_DIR}/hpx/plugins/parcelport/ibverbs/ibverbs_errors.hpp"
              "${PROJECT_SOURCE_DIR}/hpx/plugins/parcelport/ibverbs/locality.hpp"
              "${PROJECT_SOURCE_DIR}/hpx/plugins/parcelport/ibverbs/messages.hpp"
              "${PROJECT_SOURCE_DIR}/hpx/plugins/parcelport/ibverbs/receiver.hpp"
              "${PROJECT_SOURCE_DIR}/hpx/plugins/parcelport/ibverbs/sender.hpp"
              "${PROJECT_SOURCE_DIR}/hpx/plugins/parcelport/ibverbs/server.hpp"
        DEPENDENCIES
              ${IBVERBS_LIBRARIES}
              ${RDMACM_LIBRARIES}
        FOLDER "Core/Plugins/Parcelport/Ibverbs"
        )
  endmacro()
else()
  macro(add_parcelport_ibverbs_module)
  endmacro()
endif()
