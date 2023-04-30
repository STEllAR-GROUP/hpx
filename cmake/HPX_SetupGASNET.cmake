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
    set(GASNET_CXXDEFS)
    set(GASNET_INCLUDES)
    set(GASNET_LDFLAGS)
    set(GASNET_LIBFLAGS)

    foreach(TOKEN ${GASNET_CFLAGS})
       string(FIND "${TOKEN}" "-I" incpos)
       string(FIND "${TOKEN}" "-l" lnkpos)
       string(FIND "${TOKEN}" "-L" libpos)
       string(FIND "${TOKEN}" "-D" defpos)

       if(NOT ${incpos} EQUAL -1)
          string(REPLACE "-I" "" TOKEN "${TOKEN}")
          list(APPEND GASNET_INCLUDES ${TOKEN})
       elseif(NOT ${lnkpos} EQUAL -1)
          string(REPLACE "-l" "" TOKEN "${TOKEN}")
          list(APPEND GASNET_LDFLAGS ${TOKEN})
       elseif(NOT ${libpos} EQUAL -1)
          string(REPLACE "-L" "" TOKEN "${TOKEN}")
          list(APPEND GASNET_LIBFLAGS ${TOKEN})
       elseif(NOT ${defpos} EQUAL -1)
          string(REPLACE "-D" "" TOKEN "${TOKEN}")
          list(APPEND GASNET_CXXDEFS ${TOKEN})
       else()
          list(APPEND GASNET_CXXFLAGS ${TOKEN})
       endif()
    endforeach()

    foreach(TOKEN ${GASNET_LDFLAGS})
       string(FIND "${TOKEN}" "-l" lnkpos)
       if(NOT ${lnkpos} EQUAL -1)
          list(APPEND GASNET_LIBFLAGS ${TOKEN})
       endif()
    endforeach()

    list(APPEND GASNET_CXXDEFS "-DGASNET_PAR")

    #string (REPLACE ";" " " GASNET_CXXFLAGS "${GASNET_CXXFLAGS}")
    #string (REPLACE ";" " " GASNET_CXXDEFS "${GASNET_CXXDEFS}")
    #string (REPLACE ";" " " GASNET_INCLUDES "${GASNET_INCLUDES}")
    #string (REPLACE ";" " " GASNET_LDFLAGS "${GASNET_LDFLAGS}")
    #string (REPLACE ";" " " GASNET_LIBFLAGS "${GASNET_LIBFLAGS}")

    set(gasnet_libraries ${GASNET_LIBFLAGS})

    if(GASNET_MPI_FOUND AND HPX_WITH_NETWORKING)
       add_library(gasnet-mpi-par INTERFACE IMPORTED)
       target_compile_options(gasnet-mpi-par INTERFACE ${GASNET_CXXFLAGS})
       target_compile_definitions(gasnet-mpi-par INTERFACE ${GASNET_CXXDEFS})
       target_include_directories(gasnet-mpi-par SYSTEM INTERFACE ${GASNET_INCLUDES})
       target_link_options(gasnet-mpi-par INTERFACE ${GASNET_LDFLAGS})
       target_link_libraries(gasnet-mpi-par INTERFACE ${GASNET_LIBFLAGS})
    elseif(GASNET_UDP_FOUND AND HPX_WITH_NETWORKING)
       add_library(gasnet-udp-par INTERFACE IMPORTED)
       target_compile_options(gasnet-udp-par INTERFACE ${GASNET_CXXFLAGS})
       target_compile_definitions(gasnet-udp-par INTERFACE ${GASNET_CXXDEFS})
       target_include_directories(gasnet-udp-par SYSTEM INTERFACE${GASNET_INCLUDES})
       target_link_options(gasnet-udp-par INTERFACE ${GASNET_LDFLAGS})
       target_link_libraries(gasnet-udp-par INTERFACE ${GASNET_LIBFLAGS})
    else()
       add_library(gasnet-smp-par INTERFACE IMPORTED)
       target_compile_options(gasnet-smp-par INTERFACE ${GASNET_CXXFLAGS})
       target_compile_definitions(gasnet-smp-par INTERFACE ${GASNET_CXXDEFS})
       target_include_directories(gasnet-smp-par SYSTEM INTERFACE ${GASNET_INCLUDES})
       target_link_options(gasnet-smp-par INTERFACE ${GASNET_LDFLAGS})
       target_link_libraries(gasnet-smp-par INTERFACE ${GASNET_LIBFLAGS})
    endif()

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
