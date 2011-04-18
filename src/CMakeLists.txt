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

target_link_libraries(hpx_serialization
  ${hpx_MALLOC_LIBRARY} ${BOOST_FOUND_LIBRARIES})

get_target_property(path hpx_serialization LOCATION)
get_filename_component(name "${path}" NAME)
set_property(TARGET hpx_serialization APPEND
  PROPERTY COMPILE_DEFINITIONS
  "HPX_LIBRARY_NAME=${name}"
  "HPX_LIBRARY_STRING=\"${name}\""
  "BOOST_SERIALIZATION_DYN_LINK"
  "BOOST_ARCHIVE_DYN_LINK")

if(NOT HPX_DISABLE_WARNINGS AND NOT MSVC)
  set_property(TARGET hpx_serialization APPEND
    PROPERTY COMPILE_FLAGS
    "-Wall -Wno-strict-aliasing -Wsign-promo")
endif()

###############################################################################
# build instructions for main shared HPX library
add_library(hpx SHARED ${hpx_SOURCES})
set_target_properties(hpx PROPERTIES
  INSTALL_NAME_DIR "${CMAKE_INSTALL_PREFIX}/lib"
  VERSION ${HPX_VERSION}
  SOVERSION ${HPX_SOVERSION}
  CLEAN_DIRECT_OUTPUT 1
  OUTPUT_NAME hpx)

target_link_libraries(hpx hpx_serialization
  ${hpx_MALLOC_LIBRARY} ${BOOST_FOUND_LIBRARIES})
 
get_target_property(path hpx LOCATION)
get_filename_component(name "${path}" NAME)
set_property(TARGET hpx APPEND
  PROPERTY COMPILE_DEFINITIONS
  "HPX_LIBRARY_NAME=${name}"
  "HPX_LIBRARY_STRING=\"${name}\"")

if(NOT HPX_DISABLE_WARNINGS AND NOT MSVC)
  set_property(TARGET hpx APPEND
    PROPERTY COMPILE_FLAGS
    "-Wall -Wno-strict-aliasing -Wsign-promo")
endif()

if(MSVC)
  set_property(TARGET hpx APPEND
    PROPERTY COMPILE_FLAGS
    "-bigobj")
  if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    set_property(TARGET hpx APPEND
      PROPERTY COMPILE_FLAGS
      "-DHPX_EMULATE_SWAP_CONTEXT=1")
  endif()
endif()

################################################################################
# Source Groups
source_group("Header Files" REGULAR_EXPRESSION "${hpx_SOURCE_DIR}/hpx/[^.]*hpp")
source_group("Header Files\\components\\distributing_factory" REGULAR_EXPRESSION "${hpx_SOURCE_DIR}/hpx/components/distributing_factory/[^.]*hpp")
source_group("Header Files\\components\\distributing_factory\\server" REGULAR_EXPRESSION "${hpx_SOURCE_DIR}/hpx/components/distributing_factory/server/[^.]*hpp")
source_group("Header Files\\components\\distributing_factory\\stubs" REGULAR_EXPRESSION "${hpx_SOURCE_DIR}/hpx/components/distributing_factory/stubs/[^.]*hpp")
source_group("Header Files\\config" REGULAR_EXPRESSION "${hpx_SOURCE_DIR}/hpx/config/[^.]*hpp")
source_group("Header Files\\include" REGULAR_EXPRESSION "${hpx_SOURCE_DIR}/hpx/include/[^.]*hpp")
source_group("Header Files\\lcos" REGULAR_EXPRESSION "${hpx_SOURCE_DIR}/hpx/lcos/[^.]*hpp")
source_group("Header Files\\lcos\\server" REGULAR_EXPRESSION "${hpx_SOURCE_DIR}/hpx/lcos/server/[^.]*hpp")
source_group("Header Files\\lcos\\stubs" REGULAR_EXPRESSION "${hpx_SOURCE_DIR}/hpx/lcos/stubs/[^.]*hpp")
source_group("Header Files\\performance_counters" REGULAR_EXPRESSION "${hpx_SOURCE_DIR}/hpx/performance_counters/[^.]*hpp")
source_group("Header Files\\performance_counters\\server" REGULAR_EXPRESSION "${hpx_SOURCE_DIR}/hpx/performance_counters/server/[^.]*hpp")
source_group("Header Files\\performance_counters\\stubs" REGULAR_EXPRESSION "${hpx_SOURCE_DIR}/hpx/performance_counters/stubs/[^.]*hpp")
source_group("Header Files\\runtime" REGULAR_EXPRESSION "${hpx_SOURCE_DIR}/hpx/runtime/[^.]*hpp")
source_group("Header Files\\runtime\\actions" REGULAR_EXPRESSION "${hpx_SOURCE_DIR}/hpx/runtime/actions/[^.]*hpp")
source_group("Header Files\\runtime\\applier" REGULAR_EXPRESSION "${hpx_SOURCE_DIR}/hpx/runtime/applier/[^.]*hpp")
source_group("Header Files\\runtime\\components" REGULAR_EXPRESSION "${hpx_SOURCE_DIR}/hpx/runtime/components/[^.]*hpp")
source_group("Header Files\\runtime\\components\\server" REGULAR_EXPRESSION "${hpx_SOURCE_DIR}/hpx/runtime/components/server/[^.]*hpp")
source_group("Header Files\\runtime\\components\\stubs" REGULAR_EXPRESSION "${hpx_SOURCE_DIR}/hpx/runtime/components/stubs/[^.]*hpp")
source_group("Header Files\\runtime\\naming" REGULAR_EXPRESSION "${hpx_SOURCE_DIR}/hpx/runtime/naming/[^.]*hpp")
source_group("Header Files\\runtime\\naming\\detail" REGULAR_EXPRESSION "${hpx_SOURCE_DIR}/hpx/runtime/naming/detail/[^.]*hpp")
source_group("Header Files\\runtime\\naming\\server" REGULAR_EXPRESSION "${hpx_SOURCE_DIR}/hpx/runtime/naming/server/[^.]*hpp")
source_group("Header Files\\runtime\\parcelset" REGULAR_EXPRESSION "${hpx_SOURCE_DIR}/hpx/runtime/parcelset/[^.]*hpp")
source_group("Header Files\\runtime\\parcelset\\server" REGULAR_EXPRESSION "${hpx_SOURCE_DIR}/hpx/runtime/parcelset/server/[^.]*hpp")
source_group("Header Files\\runtime\\threads" REGULAR_EXPRESSION "${hpx_SOURCE_DIR}/hpx/runtime/threads/[^.]*hpp")
source_group("Header Files\\runtime\\threads\\detail" REGULAR_EXPRESSION "${hpx_SOURCE_DIR}/hpx/runtime/threads/detail/[^.]*hpp")
source_group("Header Files\\runtime\\threads\\policies" REGULAR_EXPRESSION "${hpx_SOURCE_DIR}/hpx/runtime/threads/policies/[^.]*hpp")
source_group("Header Files\\util" REGULAR_EXPRESSION "${hpx_SOURCE_DIR}/hpx/util/[^.]*hpp")
source_group("Header Files\\util\\hardware" REGULAR_EXPRESSION "${hpx_SOURCE_DIR}/hpx/util/hardware/[^.]*hpp")

source_group("Source Files" REGULAR_EXPRESSION "${hpx_SOURCE_DIR}/src/[^.]*cpp")
source_group("Source Files\\components\\distributing_factory" REGULAR_EXPRESSION "${hpx_SOURCE_DIR}/src/components/distributing_factory/[^.]*cpp")
source_group("Source Files\\components\\distributing_factory\\server" REGULAR_EXPRESSION "${hpx_SOURCE_DIR}/src/components/distributing_factory/server/[^.]*cpp")
source_group("Source Files\\components\\distributing_factory\\stubs" REGULAR_EXPRESSION "${hpx_SOURCE_DIR}/src/components/distributing_factory/stubs/[^.]*cpp")
source_group("Source Files\\config" REGULAR_EXPRESSION "${hpx_SOURCE_DIR}/src/config/[^.]*cpp")
source_group("Source Files\\include" REGULAR_EXPRESSION "${hpx_SOURCE_DIR}/src/include/[^.]*cpp")
source_group("Source Files\\lcos" REGULAR_EXPRESSION "${hpx_SOURCE_DIR}/src/lcos/[^.]*cpp")
source_group("Source Files\\lcos\\server" REGULAR_EXPRESSION "${hpx_SOURCE_DIR}/src/lcos/server/[^.]*cpp")
source_group("Source Files\\lcos\\stubs" REGULAR_EXPRESSION "${hpx_SOURCE_DIR}/src/lcos/stubs/[^.]*cpp")
source_group("Source Files\\performance_counters" REGULAR_EXPRESSION "${hpx_SOURCE_DIR}/src/performance_counters/[^.]*cpp")
source_group("Source Files\\performance_counters\\server" REGULAR_EXPRESSION "${hpx_SOURCE_DIR}/src/performance_counters/server/[^.]*cpp")
source_group("Source Files\\performance_counters\\stubs" REGULAR_EXPRESSION "${hpx_SOURCE_DIR}/src/performance_counters/stubs/[^.]*cpp")
source_group("Source Files\\runtime" REGULAR_EXPRESSION "${hpx_SOURCE_DIR}/src/runtime/[^.]*cpp")
source_group("Source Files\\runtime\\actions" REGULAR_EXPRESSION "${hpx_SOURCE_DIR}/src/runtime/actions/[^.]*cpp")
source_group("Source Files\\runtime\\applier" REGULAR_EXPRESSION "${hpx_SOURCE_DIR}/src/runtime/applier/[^.]*cpp")
source_group("Source Files\\runtime\\components" REGULAR_EXPRESSION "${hpx_SOURCE_DIR}/src/runtime/components/[^.]*cpp")
source_group("Source Files\\runtime\\components\\server" REGULAR_EXPRESSION "${hpx_SOURCE_DIR}/src/runtime/components/server/[^.]*cpp")
source_group("Source Files\\runtime\\components\\stubs" REGULAR_EXPRESSION "${hpx_SOURCE_DIR}/src/runtime/components/stubs/[^.]*cpp")
source_group("Source Files\\runtime\\naming" REGULAR_EXPRESSION "${hpx_SOURCE_DIR}/src/runtime/naming/[^.]*cpp")
source_group("Source Files\\runtime\\naming\\detail" REGULAR_EXPRESSION "${hpx_SOURCE_DIR}/src/runtime/naming/detail/[^.]*cpp")
source_group("Source Files\\runtime\\naming\\server" REGULAR_EXPRESSION "${hpx_SOURCE_DIR}/src/runtime/naming/server/[^.]*cpp")
source_group("Source Files\\runtime\\parcelset" REGULAR_EXPRESSION "${hpx_SOURCE_DIR}/src/runtime/parcelset/[^.]*cpp")
source_group("Source Files\\runtime\\parcelset\\server" REGULAR_EXPRESSION "${hpx_SOURCE_DIR}/src/runtime/parcelset/server/[^.]*cpp")
source_group("Source Files\\runtime\\threads" REGULAR_EXPRESSION "${hpx_SOURCE_DIR}/src/runtime/threads/[^.]*cpp")
source_group("Source Files\\runtime\\threads\\detail" REGULAR_EXPRESSION "${hpx_SOURCE_DIR}/src/runtime/threads/detail/[^.]*cpp")
source_group("Source Files\\runtime\\threads\\policies" REGULAR_EXPRESSION "${hpx_SOURCE_DIR}/src/runtime/threads/policies/[^.]*cpp")
source_group("Source Files\\util" REGULAR_EXPRESSION "${hpx_SOURCE_DIR}/src/util/[^.]*cpp")
source_group("Source Files\\util\\hardware" REGULAR_EXPRESSION "${hpx_SOURCE_DIR}/src/util/hardware/[^.]*cpp")

###############################################################################
# installation instructions
install(TARGETS hpx hpx_serialization
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    COMPONENT core
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                GROUP_READ GROUP_EXECUTE
                WORLD_READ WORLD_EXECUTE)

