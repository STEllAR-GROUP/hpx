# Copyright (c) 2021 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(HPX_REQUIRED_HPXLOCAL_VERSION 0.1.0)

if(NOT HPX_WITH_FETCH_HPXLOCAL)
  find_package(HPXLocal ${HPX_REQUIRED_HPXLOCAL_VERSION} EXACT REQUIRED)
elseif(NOT TARGET HPX::hpx_local AND NOT HPX_FIND_PACKAGE)
  # When HPXLocal is used through fetchcontent, we will forward variables that
  # belong in HPXLocal with a warning.
  set(HPX_HPXLOCAL_COMPATIBILITY_OPTIONS
      ASYNC_MPI
      ATTACH_DEBUGGER_ON_TEST_FAILURE
      CLANG_CUDA
      COROUTINE_COUNTERS
      CUDA
      EXAMPLES_OPENMP
      EXAMPLES_QTHREADS
      EXAMPLES_TBB
      FETCH_ASIO
      GENERIC_CONTEXT_COROUTINES
      HIP
      ITTNOTIFY
      MALLOC
      NICE_THREADLEVEL
      PARCELPORT_MPI_ENV
      PARCELPORT_MPI_MULTITHREADED
      SANITIZERS
      SPINLOCK_DEADLOCK_DETECTION
      VALGRIND
      VERIFY_LOCKS
      VERIFY_LOCKS_BACKTRACE
  )
  foreach(compat_option ${HPX_HPXLOCAL_COMPATIBILITY_OPTIONS})
    if(DEFINED HPX_WITH_${compat_option})
      if(NOT DEFINED HPXLocal_WITH_${compat_option})
        hpx_warn(
          "HPX_WITH_${compat_option} is set (\"${HPX_WITH_${compat_option}}\"), but HPXLocal_WITH_${compat_option} should be used instead. Forwarding HPX_WITH_${compat_option} to HPXLocal_WITH_${compat_option} for compatibility, but HPX_WITH_${compat_option} will be ignored in the future. Unsetting HPX_WITH_${compat_option}."
        )

        get_property(
          option_type
          CACHE "HPX_WITH_${compat_option}"
          PROPERTY TYPE
        )
        if(NOT option_type)
          set(option_type BOOL)
        endif()

        get_property(
          option_description
          CACHE "HPX_WITH_${config_option}"
          PROPERTY HELPSTRING
        )
        if(NOT option_type)
          set(option_description "")
        endif()

        set(HPXLocal_WITH_${compat_option}
            "${HPX_WITH_${compat_option}}"
            CACHE ${option_type} ${option_description} FORCE
        )

        unset(HPX_WITH_${compat_option} CACHE)
      else()
        hpx_warn(
          "HPX_WITH_${compat_option} is set (\"${HPX_WITH_${compat_option}}\"), but HPXLocal_WITH_${compat_option} should be used instead. HPXLocal_WITH_${compat_option} is set as well (\"${HPXLocal_WITH_${compat_option}}\"). Unsetting HPX_WITH_${compat_option}."
        )

        unset(HPX_WITH_${compat_option} CACHE)
      endif()
    endif()
  endforeach()

  # Propagate options to HPXLocal, if needed
  set(HPX_HPXLOCAL_CONFIGURATION_OPTIONS
      APEX
      CHECK_MODULE_DEPENDENCIES
      COMPILE_ONLY_TESTS
      COMPILER_WARNINGS
      COMPILER_WARNINGS_AS_ERRORS
      DEPRECATION_WARNINGS
      DISABLED_SIGNAL_EXCEPTION_HANDLERS
      EXAMPLES
      EXECUTABLE_PREFIX
      FAIL_COMPILE_TESTS
      FULL_RPATH
      GCC_VERSION_CHECK
      HIDDEN_VISIBILITY
      LOGGING
      PARALLEL_TESTS_BIND_NONE
      PRECOMPILED_HEADERS
      TESTS
      TESTS_BENCHMARKS
      TESTS_DEBUG_LOG
      TESTS_DEBUG_LOG_DESTINATION
      TESTS_EXAMPLES
      TESTS_EXTERNAL_BUILD
      TESTS_HEADERS
      TESTS_MAX_THREADS_PER_LOCALITY
      TESTS_REGRESSIONS
      TESTS_UNIT
      TOOLS
      UNITY_BUILD
      VS_STARTUP_PROJECT
  )
  foreach(config_option ${HPX_HPXLOCAL_CONFIGURATION_OPTIONS})
    if(DEFINED HPX_WITH_${config_option})
      if(NOT DEFINED HPXLocal_WITH_${config_option})
        hpx_debug(
          "HPX_WITH_${config_option} is (\"${HPX_WITH_${config_option}}\"). Forwarding HPX_WITH_${config_option} to HPXLocal_WITH_${config_option}."
        )

        get_property(
          option_type
          CACHE "HPX_WITH_${config_option}"
          PROPERTY TYPE
        )
        if(NOT option_type)
          set(option_type BOOL)
        endif()

        get_property(
          option_description
          CACHE "HPX_WITH_${config_option}"
          PROPERTY HELPSTRING
        )
        if(NOT option_type)
          set(option_description "")
        endif()

        set(HPXLocal_WITH_${config_option}
            "${HPX_WITH_${config_option}}"
            CACHE ${option_type} "${option_description}" FORCE
        )
      endif()
    endif()
  endforeach()

  # Handle HPX_WITH_CXXxx variables separately. Forward them to
  # HPXLocal_CXX_STANDARD.
  if(DEFINED HPX_WITH_CXX20)
    hpx_warn(
      "HPX_WITH_CXX20 deprecated, but set. Setting HPXLocal_CXX_STANDARD=20 automatically for compatibility."
    )
    set(HPXLocal_CXX_STANDARD
        "20"
        CACHE STRING "" FORCE
    )
  elseif(DEFINED HPX_WITH_CXX17)
    hpx_warn(
      "HPX_WITH_CXX17 deprecated, but set. Setting HPXLocal_CXX_STANDARD=17 automatically for compatibility."
    )
    set(HPXLocal_CXX_STANDARD
        "17"
        CACHE STRING "" FORCE
    )
  endif()

  # The MPI parcelport HPXLocal to be built using MPI support.
  if(HPX_WITH_PARCELPORT_MPI)
    set(HPXLocal_WITH_ASYNC_MPI
        ON
        CACHE BOOL "" FORCE
    )
  endif()

  # Place HPXLocal binaries in HPX's binary directory by default when on Windows
  if(MSVC AND NOT DEFINED HPXLocal_WITH_BINARY_DIR)
    set(HPXLocal_WITH_BINARY_DIR
        "${PROJECT_BINARY_DIR}"
        CACHE STRING "" FORCE
    )
  endif()

  if(FETCHCONTENT_SOURCE_DIR_HPXLOCAL)
    hpx_info(
      "HPX_WITH_FETCH_HPXLOCAL=${HPX_WITH_FETCH_HPXLOCAL}, HPXLocal will be used through CMake's FetchContent and installed alongside HPX (FETCHCONTENT_SOURCE_DIR_HPXLOCAL=${FETCHCONTENT_SOURCE_DIR_HPXLOCAL})"
    )
  else()
    hpx_info(
      "HPX_WITH_FETCH_HPXLOCAL=${HPX_WITH_FETCH_HPXLOCAL}, HPXLocal will be fetched using CMake's FetchContent and installed alongside HPX (HPX_WITH_HPXLOCAL_TAG=${HPX_WITH_HPXLOCAL_TAG})"
    )
  endif()
  include(FetchContent)

  # TODO: Forcing a fixed tag until an official release.
  hpx_warn(
    "Forcing a fixed tag for HPXLocal. This is temporary until an official release is made."
  )
  set(HPX_WITH_HPXLOCAL_TAG
      "${HPX_WITH_HPXLOCAL_TAG_DEFAULT}"
      CACHE BOOL "" FORCE
  )
  fetchcontent_declare(
    HPXLocal
    GIT_REPOSITORY https://github.com/STEllAR-GROUP/hpx-local.git
    GIT_TAG ${HPX_WITH_HPXLOCAL_TAG}
  )

  fetchcontent_makeavailable(HPXLocal)

  if(HPXLocal_WITH_CUDA)
    enable_language(CUDA)
  endif()
endif()
