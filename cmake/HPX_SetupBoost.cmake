# Copyright (c) 2014 Thomas Heller
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# We first try to find the required minimum set of Boost libraries. This will
# also give us the version of the found boost installation
find_package(Boost
  1.49
  REQUIRED
  COMPONENTS
  chrono
  date_time
  filesystem
  program_options
  regex
  serialization
  system
  thread
  )

set(Boost_TMP_LIBRARIES ${Boost_LIBRARIES})

# Set configuration option to use Boost.Context or not. This depends on the Boost
# version (Boost.Context was included with 1.51) and the Platform
if(Boost_VERSION GREATER 105000)
  find_package(Boost 1.50 QUIET REQUIRED COMPONENTS context)
  if(Boost_CONTEXT_FOUND)
    hpx_info("  context")
  endif()
  set(use_generic_coroutine_context OFF)
  if(APPLE)
    set(use_generic_coroutine_context ON)
  endif()
  if(HPX_PLATFORM_UC STREQUAL "BLUEGENEQ" AND Boost_VERSION GREATER 105500)
    set(use_generic_coroutine_context ON)
  endif()
endif()

set(Boost_TMP_LIBRARIES ${Boost_TMP_LIBRARIES} ${Boost_LIBRARIES})

hpx_option(
  WITH_GENERIC_COROUTINE_CONTEXT BOOL ${use_generic_coroutine_context}
  "Use Boost.Context as the underlying coroutines context switch implementation."
)

if(WITH_GENERIC_COROUTINE_CONTEXT)
  if(NOT Boost_CONTEXT_FOUND)
    hpx_error("The usage of Boost.Context was selected but Boost.Context was not found (Version 1.51 or higher is required).")
  endif()
  if(HPX_PLATFORM_UC STREQUAL "BLUEGENEQ")
    if(Boost_VERSION LESS 105600)
      hpx_error("On BlueGene/Q, Boost.Context can only be used with a Boost >=1.56")
    endif()
  endif()
  hpx_add_config_define(HPX_HAVE_GENERIC_CONTEXT_COROUTINES)
endif()

# If the found Boost installation is < 1.53, we need to include our packaged
# atomic library
if(Boost_VERSION LESS 105300)
  set(Boost_INCLUDE_DIRS ${Boost_INCLUDE_DIRS} "${hpx_SOURCE_DIR}/external/atomic")
  set(Boost_INCLUDE_DIRS ${Boost_INCLUDE_DIRS} "${hpx_SOURCE_DIR}/external/lockfree")
else()
  find_package(Boost 1.53 QUIET REQUIRED COMPONENTS atomic)
  if(Boost_ATOMIC_FOUND)
    hpx_info("  atomic")
  endif()

  set(Boost_TMP_LIBRARIES ${Boost_TMP_LIBRARIES} ${Boost_LIBRARIES})
endif()

set(Boost_LIBRARIES ${Boost_TMP_LIBRARIES})
set(Boost_INCLUDE_DIRS ${Boost_INCLUDE_DIRS} ${hpx_SOURCE_DIR}/external/cache)
set(Boost_INCLUDE_DIRS ${Boost_INCLUDE_DIRS} ${hpx_SOURCE_DIR}/external/endian)

# If we compile natively for the MIC, we need some workarounds for certain
# Boost headers
# FIXME: push changes upstream
if(HPX_PLATFORM_UC STREQUAL "XEONPHI")
  set(Boost_INCLUDE_DIRS ${hpx_SOURCE_DIR}/external/asio ${Boost_INCLUDE_DIRS})
endif()

# Boost preprocessor definitions
hpx_add_config_define(BOOST_PARAMETER_MAX_ARITY 7)
hpx_add_config_define(HPX_COROUTINE_ARG_MAX 1)
if(NOT MSVC)
  hpx_add_config_define(HPX_COROUTINE_NO_SEPARATE_CALL_SITES)
endif()
hpx_add_config_define(HPX_LOG_NO_TSS)
hpx_add_config_define(HPX_LOG_NO_TS)
hpx_add_config_define(BOOST_BIGINT_HAS_NATIVE_INT64)

# Disable usage of std::atomics in lockfree
if(Boost_VERSION LESS 105300)
  hpx_add_config_define(BOOST_NO_0X_HDR_ATOMIC)
endif()

