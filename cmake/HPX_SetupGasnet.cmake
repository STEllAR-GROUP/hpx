# Copyright (c) 2019-2022 Ste||ar Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

include(HPX_Message)
include(FindGasnet)

macro(hpx_setup_gasnet)

  if(NOT GASNET_FOUND)

    find_gasnet()

    set(GASNET_CXXFLAGS)
    set(GASNET_LDFLAGS_TMP ${GASNET_LDFLAGS})
    set(GASNET_LDFLAGS)

    foreach(TOKEN ${GASNET_CFLAGS})
       list(APPEND GASNET_CXXFLAGS ${TOKEN})
    endforeach()

    foreach(TOKEN ${GASNET_LDFLAGS_TMP})
       list(APPEND GASNET_LDFLAGS ${TOKEN})
    endforeach()

    add_library(gasnet INTERFACE IMPORTED)
    target_compile_options(gasnet INTERFACE ${GASNET_CXXFLAGS})
    target_include_directories(gasnet INTERFACE ${GASNET_LIBRARY_DIRS})
    target_link_options(gasnet INTERFACE ${GASNET_LDFLAGS})
    target_link_libraries(gasnet INTERFACE ${GASNET_LIBRARIES})

    if(GASNET_MPI_FOUND)
       # Setup PMI imported target
       find_package(PMI)
       if(PMI_FOUND)
          hpx_add_config_define_namespace(
             DEFINE HPX_PARCELPORT_GASNET_HAVE_PMI NAMESPACE PARCELPORT_GASNET
          )
          add_library(Pmi::pmi INTERFACE IMPORTED)
          target_include_directories(Pmi::pmi SYSTEM INTERFACE ${PMI_INCLUDE_DIR})
          target_link_libraries(Pmi::pmi INTERFACE ${PMI_LIBRARY})
       endif()
    endif()

  endif()
endmacro()
