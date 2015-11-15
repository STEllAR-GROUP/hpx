# Copyright (c) 2014 Thomas Heller
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# We first try to find the required minimum set of Boost libraries. This will
# also give us the version of the found boost installation
if(HPX_WITH_STATIC_LINKING)
  set(Boost_USE_STATIC_LIBS ON)
endif()

# Add additional version to recognize
set(Boost_ADDITIONAL_VERSIONS
    ${Boost_ADDITIONAL_VERSIONS}
    "1.60.0" "1.60"
    "1.59.0" "1.59"
    "1.58.0" "1.58"
    "1.57.0" "1.57")

find_package(Boost
  1.49
  REQUIRED
  COMPONENTS
  chrono
  date_time
  filesystem
  program_options
  regex
  system
  thread
  )

if(NOT Boost_FOUND)
  hpx_error("Could not find Boost. Please set BOOST_ROOT to point to your Boost installation.")
endif()

set(Boost_TMP_LIBRARIES ${Boost_LIBRARIES})
if(UNIX AND NOT CYGWIN)
  find_library(BOOST_UNDERLYING_THREAD_LIBRARY NAMES pthread DOC "The threading library used by boost.thread")
  if(NOT BOOST_UNDERLYING_THREAD_LIBRARY AND (HPX_PLATFORM_UC STREQUAL "XEONPHI"))
    set(BOOST_UNDERLYING_THREAD_LIBRARY "-pthread")
  endif()
  set(Boost_TMP_LIBRARIES ${Boost_TMP_LIBRARIES} ${BOOST_UNDERLYING_THREAD_LIBRARY})
endif()

# Set configuration option to use Boost.Context or not. This depends on the Boost
# version (Boost.Context was included with 1.51) and the Platform
set(use_generic_coroutine_context OFF)
if(Boost_VERSION GREATER 105000)
  find_package(Boost 1.50 QUIET COMPONENTS context)
  if(Boost_CONTEXT_FOUND)
    hpx_info("  context")
  endif()
  if(APPLE)
    set(use_generic_coroutine_context ON)
  endif()
  if(HPX_PLATFORM_UC STREQUAL "BLUEGENEQ" AND Boost_VERSION GREATER 105500)
    set(use_generic_coroutine_context ON)
  endif()
endif()

hpx_option(
  HPX_WITH_GENERIC_CONTEXT_COROUTINES
  BOOL
  "Use Boost.Context as the underlying coroutines context switch implementation."
  ${use_generic_coroutine_context} ADVANCED)

set(Boost_TMP_LIBRARIES ${Boost_TMP_LIBRARIES} ${Boost_LIBRARIES})

if(HPX_WITH_COMPRESSION_BZIP2 OR HPX_WITH_COMPRESSION_ZLIB)
  find_package(Boost 1.49 QUIET COMPONENTS iostreams)
  if(Boost_IOSTREAMS_FOUND)
    hpx_info("  iostreams")
  else()
    hpx_error("Could not find Boost.Iostreams but HPX_WITH_COMPRESSION_BZIP2=On or HPX_WITH_COMPRESSION_LIB=On. Either set it to off or provide a boost installation including the iostreams library")
  endif()
  set(Boost_TMP_LIBRARIES ${Boost_TMP_LIBRARIES} ${Boost_LIBRARIES})
endif()

# attempt to load Boost.Random (if available), it's needed for one example only
find_package(Boost 1.49 QUIET COMPONENTS random)
if(Boost_RANDOM_FOUND)
  hpx_info("  random")
  set(Boost_TMP_LIBRARIES ${Boost_TMP_LIBRARIES} ${Boost_LIBRARIES})
endif()

# If the found Boost installation is < 1.53, we need to include our packaged
# atomic library
if(Boost_VERSION LESS 105300)
  set(Boost_INCLUDE_DIRS ${Boost_INCLUDE_DIRS} "${PROJECT_SOURCE_DIR}/external/atomic")
  set(Boost_INCLUDE_DIRS ${Boost_INCLUDE_DIRS} "${PROJECT_SOURCE_DIR}/external/lockfree")
else()
  find_package(Boost 1.53 QUIET REQUIRED COMPONENTS atomic)
  if(Boost_ATOMIC_FOUND)
    hpx_info("  atomic")
  endif()

  set(Boost_TMP_LIBRARIES ${Boost_TMP_LIBRARIES} ${Boost_LIBRARIES})
endif()

set(Boost_LIBRARIES ${Boost_TMP_LIBRARIES})
set(Boost_INCLUDE_DIRS ${Boost_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/external/cache)

# If we compile natively for the MIC, we need some workarounds for certain
# Boost headers
# FIXME: push changes upstream
if(HPX_PLATFORM_UC STREQUAL "XEONPHI")
  set(Boost_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/external/asio ${Boost_INCLUDE_DIRS})
endif()

# Boost preprocessor definitions
hpx_add_config_define(BOOST_PARAMETER_MAX_ARITY 7)
if(MSVC)
  HPX_option(HPX_WITH_BOOST_ALL_DYNAMIC_LINK BOOL "Add BOOST_ALL_DYN_LINK to compile flags" OFF)
  if (HPX_WITH_BOOST_ALL_DYNAMIC_LINK)
    hpx_add_config_define(BOOST_ALL_DYN_LINK)
  endif()
else()
  hpx_add_config_define(HPX_COROUTINE_NO_SEPARATE_CALL_SITES)
endif()
hpx_add_config_define(HPX_HAVE_LOG_NO_TSS)
hpx_add_config_define(HPX_HAVE_LOG_NO_TS)
hpx_add_config_define(BOOST_BIGINT_HAS_NATIVE_INT64)

# Disable usage of std::atomics in lockfree
if(Boost_VERSION LESS 105300)
  hpx_add_config_define(BOOST_NO_0X_HDR_ATOMIC)
endif()

include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})
if(NOT MSVC)
  hpx_libraries(${Boost_LIBRARIES})
else()
  hpx_library_dir(${Boost_LIBRARY_DIRS})
endif()
