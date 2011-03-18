# Copyright (c) 2007-2011 Hartmut Kaiser
# Copyright (c)      2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

add_definitions(-DHPX_COMPONENT_NAME=hpx)
add_definitions(-DHPX_EXPORTS)
add_definitions(-DBOOST_COROUTINE_EXPORTS)

add_hpx_library_sources(hpx
  exception.cpp
  exception_list.cpp
  runtime.cpp)

add_subdirectory(lcos)
add_subdirectory(performance_counters)
add_subdirectory(runtime)
add_subdirectory(util)

# add source file from coroutine library 
if(NOT MSVC)
  add_hpx_library_sources(hpx
    ${hpx_SOURCE_DIR}/external/coroutine/libs/coroutine/src/swapcontext.cpp)
endif()

# Boost Chrono sources (as long as this library is not in Boost)
if(HPX_INTERNAL_CHRONO AND ${BOOST_MINOR_VERSION} LESS 47)
  add_hpx_library_sources(hpx
    ${hpx_SOURCE_DIR}/external/chrono/libs/chrono/src/chrono.cpp)
endif()

###############################################################################
# build instructions for serialization support library
add_library(hpx_serialization SHARED ${hpx_serialization_SOURCES})
set_target_properties(hpx_serialization PROPERTIES
  INSTALL_NAME_DIR "${CMAKE_INSTALL_PREFIX}/lib"
  VERSION ${HPX_VERSION}
  SOVERSION ${HPX_SOVERSION}
  CLEAN_DIRECT_OUTPUT 1
  OUTPUT_NAME hpx_serialization)
target_link_libraries(hpx_serialization ${BOOST_FOUND_LIBRARIES})

###############################################################################
# build instructions for main shared HPX library
add_library(hpx SHARED ${hpx_SOURCES})
set_target_properties(hpx PROPERTIES
  INSTALL_NAME_DIR "${CMAKE_INSTALL_PREFIX}/lib"
  VERSION ${HPX_VERSION}
  SOVERSION ${HPX_SOVERSION}
  CLEAN_DIRECT_OUTPUT 1
  OUTPUT_NAME hpx)
target_link_libraries(hpx hpx_serialization ${BOOST_FOUND_LIBRARIES})

###############################################################################
# installation instructions
install(TARGETS hpx hpx_serialization
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    COMPONENT core
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                GROUP_READ GROUP_EXECUTE
                WORLD_READ WORLD_EXECUTE)

