# Copyright (c) 2007-2012 Hartmut Kaiser
# Copyright (c)      2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

add_subdirectory(components)

if(UNIX AND HPX_WARNINGS)
  hpx_use_flag_if_available(-Wextra)

  # We never want trampolines
  hpx_use_flag_if_available(-Wtrampolines)

  # These are usually benign and can't be suppressed because of
  # interface requirements
  hpx_use_flag_if_available(-Wno-unused-parameter)

  # There are ignored qualifiers in Boost, so we have to ignore them
  hpx_use_flag_if_available(-Wno-ignored-qualifiers)

  # Be extra strict about format checks
  # Boost.Logging is built on fprintf, sadly
  hpx_use_flag_if_available(-Wformat=2)
  hpx_use_flag_if_available(-Wno-format-nonliteral)

  # Self initialization is dangerous
  hpx_use_flag_if_available(-Winit-self)

  # For portability
  hpx_use_flag_if_available(-Wdouble-promotion)

  # Warn about casting that violates qualifiers or alignment
  hpx_use_flag_if_available(-Wcast-qual)
  hpx_use_flag_if_available(-Wcast-align)

  hpx_use_flag_if_available(-Werror=trampolines)
  hpx_use_flag_if_available(-Werror=parentheses)
  hpx_use_flag_if_available(-Werror=reorder)
  hpx_use_flag_if_available(-Werror=return-type)
  hpx_use_flag_if_available(-Werror=sequence-point)
  hpx_use_flag_if_available(-Werror=uninitialized)
  hpx_use_flag_if_available(-Werror=missing-field-initializers)
  hpx_use_flag_if_available(-Werror=format)
  hpx_use_flag_if_available(-Werror=missing-braces)
  hpx_use_flag_if_available(-Werror=sign-compare)
endif()

foreach(lib ${hpx_LIBRARIES})
  set(${lib}_SOURCES "" CACHE INTERNAL "Sources for lib${lib}." FORCE)

  if(MSVC)
    set(${lib}_HEADERS "" CACHE INTERNAL "Headers for lib${lib}." FORCE)
  endif()
endforeach()

################################################################################
# gather sources

# libhpx_serialization sources
add_hpx_library_sources(hpx_serialization
  GLOB_RECURSE GLOBS "${hpx_SOURCE_DIR}/src/util/portable_binary_*archive.cpp")

# libhpx sources
add_hpx_library_sources(hpx
  GLOB GLOBS "${hpx_SOURCE_DIR}/src/*.cpp"
  EXCLUDE ".*hpx_init.cpp")
add_hpx_library_sources(hpx
  GLOB_RECURSE GLOBS "${hpx_SOURCE_DIR}/src/runtime/*.cpp"
  APPEND)
add_hpx_library_sources(hpx
  GLOB_RECURSE GLOBS "${hpx_SOURCE_DIR}/src/performance_counters/*.cpp"
  APPEND)
add_hpx_library_sources(hpx
  GLOB_RECURSE GLOBS "${hpx_SOURCE_DIR}/src/util/*.cpp"
  EXCLUDE ".*portable_binary_.*archive[.]cpp"
  APPEND)
add_hpx_library_sources(hpx
  GLOB_RECURSE GLOBS "${hpx_SOURCE_DIR}/src/lcos/*.cpp"
  APPEND)

# libhpx_init sources
add_hpx_library_sources(hpx_init
  GLOB GLOBS "${hpx_SOURCE_DIR}/src/*hpx_init.cpp")

set(hpx_external_SOURCES "" CACHE INTERNAL "Sources for libhpx_external." FORCE)

# Boost Chrono sources (as long as this library is not in Boost)
if(HPX_INTERNAL_CHRONO)
  add_hpx_library_sources(hpx_external
    GLOB GLOBS "${hpx_SOURCE_DIR}/external/chrono/libs/chrono/src/chrono.cpp"
    APPEND)
endif()

# Add Boost.Backtrace sources if stack traces are enabled
if(HPX_STACKTRACES)
  add_hpx_library_sources(hpx_external
    GLOB GLOBS "${hpx_SOURCE_DIR}/external/backtrace/libs/backtrace/src/backtrace.cpp"
    APPEND)
endif()

if(NOT MSVC)
  add_hpx_library_sources(hpx_external
    GLOB GLOBS "${hpx_SOURCE_DIR}/external/coroutine/libs/coroutine/src/swapcontext.cpp"
    APPEND)
else()
  if(HPX_EMULATE_SWAP_CONTEXT)
    set(switch_to_fiber_source "${hpx_SOURCE_DIR}/external/coroutine/libs/coroutine/src/switch_to_fiber.asm")
    add_hpx_library_sources(hpx_external_objects
      GLOB GLOBS "${switch_to_fiber_source}")
    add_custom_command(
      OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/switch_to_fiber.obj
      COMMAND ${CMAKE_ASM_MASM_COMPILER} /Fo ${CMAKE_CURRENT_BINARY_DIR}/switch_to_fiber.obj /nologo /c ${switch_to_fiber_source}
      DEPENDS ${switch_to_fiber_source})
    set(hpx_external_OBJECTS ${CMAKE_CURRENT_BINARY_DIR}/switch_to_fiber.obj
      CACHE INTERNAL "Object files for libhpx_external." FORCE)
    add_definitions(-DHPX_EMULATE_SWAP_CONTEXT)
  endif()
endif()

################################################################################
# gather headers

# libhpx_serialization headers
add_hpx_library_headers(hpx_serialization
  GLOB_RECURSE GLOBS "${hpx_SOURCE_DIR}/hpx/util/portable_binary_*archive.hpp")

# libhpx headers
add_hpx_library_headers(hpx
  GLOB GLOBS "${hpx_SOURCE_DIR}/hpx/*.hpp"
  EXCLUDE ".*hpx_init.hpp")
add_hpx_library_headers(hpx
  GLOB_RECURSE GLOBS "${hpx_SOURCE_DIR}/hpx/runtime/*.hpp"
  APPEND)
add_hpx_library_headers(hpx
  GLOB_RECURSE GLOBS "${hpx_SOURCE_DIR}/hpx/config/*.hpp"
  APPEND)
add_hpx_library_headers(hpx
  GLOB_RECURSE GLOBS "${hpx_SOURCE_DIR}/hpx/include/*.hpp"
  APPEND)
add_hpx_library_headers(hpx
  GLOB_RECURSE GLOBS "${hpx_SOURCE_DIR}/hpx/lcos/*.hpp"
  APPEND)
add_hpx_library_headers(hpx
  GLOB_RECURSE GLOBS "${hpx_SOURCE_DIR}/hpx/performance_counters/*.hpp"
  APPEND)
add_hpx_library_headers(hpx
  GLOB_RECURSE GLOBS "${hpx_SOURCE_DIR}/hpx/traits/*.hpp"
  APPEND)
add_hpx_library_headers(hpx
  GLOB_RECURSE GLOBS "${hpx_SOURCE_DIR}/hpx/util/*.h*"
  EXCLUDE ".*portable_binary_.*archive[.]hpp"
  APPEND)

# libhpx_init headers
add_hpx_library_headers(hpx_init
  GLOB GLOBS "${hpx_SOURCE_DIR}/hpx/hpx_init.hpp")

################################################################################
# make source groups
add_hpx_source_group(
  NAME hpx_serialization
  CLASS "Source Files"
  ROOT ${hpx_SOURCE_DIR}/src/util
  TARGETS ${hpx_serialization_SOURCES})

add_hpx_source_group(
  NAME hpx_serialization
  CLASS "Header Files"
  ROOT ${hpx_SOURCE_DIR}/hpx/util
  TARGETS ${hpx_serialization_HEADERS})

add_hpx_source_group(
  NAME hpx CLASS "Source Files"
  ROOT ${hpx_SOURCE_DIR}/src
  TARGETS ${hpx_SOURCES})

add_hpx_source_group(
  NAME hpx CLASS "External Source Files"
  ROOT ${hpx_SOURCE_DIR}/external
  TARGETS ${hpx_external_SOURCES})

add_hpx_source_group(
  NAME hpx
  CLASS "Header Files"
  ROOT ${hpx_SOURCE_DIR}/hpx
  TARGETS ${hpx_HEADERS})

add_hpx_source_group(
  NAME hpx_init
  CLASS "Source Files"
  ROOT ${hpx_SOURCE_DIR}/src
  TARGETS ${hpx_init_SOURCES})

add_hpx_source_group(
  NAME hpx_init
  CLASS "Header Files"
  ROOT ${hpx_SOURCE_DIR}/hpx
  TARGETS ${hpx_init_HEADERS})

################################################################################
# libhpx_serialization
add_library(hpx_serialization SHARED EXCLUDE_FROM_ALL
  ${hpx_serialization_SOURCES} ${hpx_serialization_HEADERS})

set_target_properties(hpx_serialization
                      PROPERTIES SKIP_BUILD_RPATH TRUE
                                 BUILD_WITH_INSTALL_RPATH TRUE
                                 INSTALL_RPATH_USE_LINK_PATH TRUE 
                                 INSTALL_RPATH ${HPX_RPATH})

set_target_properties(hpx_serialization PROPERTIES
  VERSION ${HPX_VERSION}
  SOVERSION ${HPX_SOVERSION}
  CLEAN_DIRECT_OUTPUT 1
  OUTPUT_NAME hpx_serialization
  FOLDER "Core"
  RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}
  RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}
  RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL}
  RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO})

set(hpx_serialization_lib_dependencies
  ${hpx_MALLOC_LIBRARY}
  ${BOOST_serialization_LIBRARY})

if(NOT MSVC)
  set(hpx_serialization_lib_dependencies
      ${hpx_serialization_lib_dependencies}
      ${BOOST_system_LIBRARY})
endif()

target_link_libraries(hpx_serialization
  ${hpx_serialization_lib_dependencies})

hpx_get_target_file(name hpx_serialization)

set_property(TARGET hpx_serialization APPEND
  PROPERTY COMPILE_DEFINITIONS
  "HPX_LIBRARY=\"${name}\""
  "HPX_EXPORTS"
  "BOOST_COROUTINE_EXPORTS")

if(NOT HPX_NO_INSTALL)
  hpx_library_install(hpx_serialization)
endif()

################################################################################
# libhpx
add_library(hpx SHARED EXCLUDE_FROM_ALL
  ${hpx_SOURCES} ${hpx_external_SOURCES} ${hpx_external_OBJECTS} ${hpx_HEADERS})

set_target_properties(hpx PROPERTIES
  VERSION ${HPX_VERSION}
  SOVERSION ${HPX_SOVERSION}
  CLEAN_DIRECT_OUTPUT 1
  OUTPUT_NAME hpx
  FOLDER "Core"
  RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}
  RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}
  RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL}
  RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO})

set_target_properties(hpx
                      PROPERTIES SKIP_BUILD_RPATH TRUE
                                 BUILD_WITH_INSTALL_RPATH TRUE
                                 INSTALL_RPATH_USE_LINK_PATH TRUE 
                                 INSTALL_RPATH ${HPX_RPATH})

if(MSVC AND HPX_EMULATE_SWAP_CONTEXT)
  set_target_properties(hpx PROPERTIES LINK_FLAGS "/EXPORT:switch_to_fiber")
endif()

target_link_libraries(hpx
  hpx_serialization
  ${hpx_MALLOC_LIBRARY}
  ${hpx_RUNTIME_LIBRARIES}
  ${BOOST_FOUND_LIBRARIES})

hpx_get_target_file(name hpx)

set_property(TARGET hpx APPEND
  PROPERTY COMPILE_DEFINITIONS
  "HPX_LIBRARY=\"${name}\""
  "HPX_EXPORTS"
  "HPX_ACTION_ARGUMENT_LIMIT=3"
  "BOOST_ENABLE_ASSERT_HANDLER"
  "BOOST_COROUTINE_EXPORTS") # Do we still need this?

if(NOT HPX_NO_INSTALL)
  hpx_library_install(hpx)
endif()

################################################################################
# libhpx_init
add_library(hpx_init STATIC EXCLUDE_FROM_ALL
  ${hpx_init_SOURCES} ${hpx_init_HEADERS})

set_property(TARGET hpx_init APPEND
  PROPERTY COMPILE_DEFINITIONS
  "HPX_APPLICATION_EXPORTS"
  "HPX_ACTION_ARGUMENT_LIMIT=3"
  "BOOST_ENABLE_ASSERT_HANDLER")

set_property(TARGET hpx_init PROPERTY FOLDER "Core")

if(NOT HPX_NO_INSTALL)
  hpx_archive_install(hpx_init)
endif()

