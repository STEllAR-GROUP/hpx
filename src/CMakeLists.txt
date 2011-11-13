# Copyright (c) 2007-2011 Hartmut Kaiser
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

  # For some reason the detection tests give false positives for this flag.
  if(040600 LESS ${GCC_VERSION} OR 040600 EQUAL ${GCC_VERSION})
    hpx_use_flag_if_available(-Wno-unused-but-set-parameter)
  endif()

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
  GLOB_RECURSE GLOBS "${hpx_SOURCE_DIR}/src/runtime/*.cpp")
add_hpx_library_sources(hpx
  GLOB_RECURSE GLOBS "${hpx_SOURCE_DIR}/src/performance_counters/*.cpp")
add_hpx_library_sources(hpx
  GLOB_RECURSE GLOBS "${hpx_SOURCE_DIR}/src/util/*.cpp"
  EXCLUDE ".*portable_binary_.*archive[.]cpp")
add_hpx_library_sources(hpx
  GLOB_RECURSE GLOBS "${hpx_SOURCE_DIR}/src/lcos/*.cpp")

# libhpx_init sources
add_hpx_library_sources(hpx_init
  GLOB GLOBS "${hpx_SOURCE_DIR}/src/*hpx_init.cpp")

# Boost Chrono sources (as long as this library is not in Boost)
if(HPX_INTERNAL_CHRONO AND ${BOOST_MINOR_VERSION} LESS 47)
  add_hpx_library_sources(hpx
    GLOB GLOBS "${hpx_SOURCE_DIR}/external/chrono/libs/chrono/src/chrono.cpp")
endif()

# Add Boost.Backtrace sources if stack traces are enabled
if(HPX_STACKTRACES)
  add_hpx_library_sources(hpx
    GLOB GLOBS "${hpx_SOURCE_DIR}/external/backtrace/libs/backtrace/src/backtrace.cpp")
endif()

add_hpx_library_sources(hpx
  GLOB GLOBS "${hpx_SOURCE_DIR}/external/coroutine/libs/coroutine/src/swapcontext.cpp")

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
  GLOB_RECURSE GLOBS "${hpx_SOURCE_DIR}/hpx/runtime/*.hpp")
add_hpx_library_headers(hpx
  GLOB_RECURSE GLOBS "${hpx_SOURCE_DIR}/hpx/include/*.hpp")
add_hpx_library_headers(hpx
  GLOB_RECURSE GLOBS "${hpx_SOURCE_DIR}/hpx/config/*.hpp")
add_hpx_library_headers(hpx
  GLOB_RECURSE GLOBS "${hpx_SOURCE_DIR}/hpx/performance_counters/*.hpp")
add_hpx_library_headers(hpx
  GLOB_RECURSE GLOBS "${hpx_SOURCE_DIR}/hpx/util/*.h*"
  EXCLUDE ".*portable_binary_.*archive[.]hpp")
add_hpx_library_headers(hpx
  GLOB_RECURSE GLOBS "${hpx_SOURCE_DIR}/hpx/lcos/*.hpp")

# libhpx_init headers
add_hpx_library_sources(hpx_init
  GLOB GLOBS "${hpx_SOURCE_DIR}/hpx/*hpx_init.hpp")

################################################################################
# make source groups
add_hpx_source_group(CLASS "Source Files" ROOT ${hpx_SOURCE_DIR}/src
  TARGETS ${hpx_serialization_SOURCES} ${hpx_SOURCES} ${hpx_init_SOURCES})

add_hpx_source_group(CLASS "Header Files" ROOT ${hpx_SOURCE_DIR}/hpx
  TARGETS ${hpx_serialization_HEADERS} ${hpx_HEADERS} ${hpx_init_HEADERS})

################################################################################
# libhpx_serialization
add_library(hpx_serialization SHARED
  ${hpx_serialization_SOURCES} ${hpx_serialization_HEADERS})

set_target_properties(hpx_serialization PROPERTIES
  VERSION ${HPX_VERSION}
  SOVERSION ${HPX_SOVERSION}
  CLEAN_DIRECT_OUTPUT 1
  OUTPUT_NAME hpx_serialization)

target_link_libraries(hpx_serialization
  ${hpx_MALLOC_LIBRARY}
  ${BOOST_serialization_LIBRARY}
  ${BOOST_system_LIBRARY})

hpx_mangle_name(name hpx_serialization)

set_property(TARGET hpx_serialization APPEND
  PROPERTY COMPILE_DEFINITIONS
  "HPX_LIBRARY=\"${name}\""
  "HPX_COMPONENT_NAME=hpx"
  "HPX_EXPORTS"
  "BOOST_COROUTINE_EXPORTS")

hpx_library_install(${name})

################################################################################
# libhpx
add_library(hpx SHARED ${hpx_SOURCES} ${hpx_HEADERS})
set_target_properties(hpx PROPERTIES
  VERSION ${HPX_VERSION}
  SOVERSION ${HPX_SOVERSION}
  CLEAN_DIRECT_OUTPUT 1
  OUTPUT_NAME hpx)

target_link_libraries(hpx
  hpx_serialization
  ${hpx_MALLOC_LIBRARY}
  ${hpx_RUNTIME_LIBRARIES}
  ${BOOST_FOUND_LIBRARIES})

hpx_mangle_name(name hpx)

set_property(TARGET hpx APPEND
  PROPERTY COMPILE_DEFINITIONS
  "HPX_LIBRARY=\"${name}\""
  "HPX_COMPONENT_NAME=hpx"
  "HPX_EXPORTS"
  "HPX_ACTION_ARGUMENT_LIMIT=2"
  "BOOST_ENABLE_ASSERT_HANDLER"
  "BOOST_COROUTINE_EXPORTS") # Do we still need this?

hpx_library_install(${name})

################################################################################
# libhpx_init
add_library(hpx_init STATIC ${hpx_init_SOURCES} ${hpx_init_HEADERS})

set_property(TARGET hpx_init APPEND
  PROPERTY COMPILE_DEFINITIONS
  "HPX_APPLICATION_EXPORTS"
  "HPX_ACTION_ARGUMENT_LIMIT=2"
  "BOOST_ENABLE_ASSERT_HANDLER")

hpx_mangle_name(name hpx_init)

hpx_archive_install(${name})

