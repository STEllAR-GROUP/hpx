# Copyright (c) 2024 Panos Syskakis
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Fetches Boost using CMake's FetchContent and builds it.
# Installs Boost alongside HPX and creates a file that, when included, will point find_package to that Boost installation

set(__boost_libraries "")
if(HPX_PARCELPORT_LIBFABRIC_WITH_LOGGING
   OR HPX_PARCELPORT_LIBFABRIC_WITH_DEV_MODE
)
  set(__boost_libraries ${__boost_libraries} log log_setup date_time chrono
                        thread
  )
endif()

if(HPX_WITH_GENERIC_CONTEXT_COROUTINES)
  # if context is needed, we should still link with boost thread and chrono
  set(__boost_libraries ${__boost_libraries} context thread chrono)
endif()

set(HPX_WITH_BOOST_VERSION "1.84.0")
hpx_info(
  "HPX_WITH_FETCH_BOOST=${HPX_WITH_FETCH_BOOST}, Boost v${HPX_WITH_BOOST_VERSION} will be fetched using CMake's FetchContent"
)
include(FetchContent)
fetchcontent_declare(
  Boost
  URL https://github.com/boostorg/boost/releases/download/boost-${HPX_WITH_BOOST_VERSION}/boost-${HPX_WITH_BOOST_VERSION}.tar.gz
  TLS_VERIFY true
  DOWNLOAD_EXTRACT_TIMESTAMP true
)

if(NOT Boost_POPULATED)
  fetchcontent_populate(Boost)
endif()

if(NOT _HPX_IS_FETCHED_BOOST_BUILT)
  set(boost_with_libraries "--with-libraries=headers")

  if(__boost_libraries)
    list(JOIN ${__boost_libraries} "," __boost_libraries_comma_sep)
    set(boost_with_libraries
        "${boost_with_libraries},${__boost_libraries_comma_sep}"
    )
  endif()

  if(WIN32)
    execute_process(
      COMMAND
        cmd /C
        ".\\bootstrap.bat --prefix=${boost_BINARY_DIR} ${boost_with_libraries} &&\
                                .\\b2 install cxxflags=/std:c++${HPX_CXX_STANDARD}"
      WORKING_DIRECTORY "${boost_SOURCE_DIR}"
      RESULT_VARIABLE _result
    )
  else()
    execute_process(
      COMMAND
        sh -c
        "./bootstrap.sh --prefix=${boost_BINARY_DIR} ${boost_with_libraries} &&\
                               ./b2 install cxxflags=--std=c++${HPX_CXX_STANDARD}"
      WORKING_DIRECTORY "${boost_SOURCE_DIR}"
      RESULT_VARIABLE _result
    )
  endif()

  if(NOT _result EQUAL 0)
    hpx_error("Failed to build Boost")
  endif()

  set(_HPX_IS_FETCHED_BOOST_BUILT
      ON
      CACHE INTERNAL ""
  )
endif()

# Copy the Boost build directory to the install directory as-is
install(
  DIRECTORY "${boost_BINARY_DIR}/"
  DESTINATION "${CMAKE_INSTALL_PREFIX}"
  USE_SOURCE_PERMISSIONS
  COMPONENT core
)

# Create a file on the build tree to point to the Boost build
file(WRITE "${CMAKE_BINARY_DIR}/hpx_boost_root.cmake"
     "set(hpx_boost_root \"${boost_BINARY_DIR}\")"
)

# Create a file on the install tree to point to the Boost install
file(WRITE "${CMAKE_BINARY_DIR}/hpx_boost_root-install.cmake"
     "set(hpx_boost_root \"${CMAKE_INSTALL_PREFIX}\")"
)

install(
  FILES "${CMAKE_BINARY_DIR}/hpx_boost_root-install.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${HPX_PACKAGE_NAME}"
  COMPONENT cmake
)
