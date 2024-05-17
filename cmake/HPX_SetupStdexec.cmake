#  Copyright (c) 2024 Isidoros Tsaousis-Seiras
#
#  SPDX-License-Identifier: BSL-1.0
#  Distributed under the Boost Software License, Version 1.0. (See accompanying
#  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(STDEXEC_ROOT AND NOT Stdexec_ROOT)
  set(Stdexec_ROOT ${STDEXEC_ROOT})
  # remove STDEXEC_ROOT from the cache
  unset(STDEXEC_ROOT CACHE)
endif()

if(Stdexec_ROOT OR HPX_WITH_FETCH_STDEXEC)
  # explicitly enable HPX_WITH_STDEXEC
  set(HPX_WITH_STDEXEC ON)

  # prefer Stdexec_ROOT over HPX_WITH_FETCH_STDEXEC by default
  if(Stdexec_ROOT AND HPX_WITH_FETCH_STDEXEC)
    set(HPX_WITH_FETCH_STDEXEC OFF)
    hpx_warn(
      "Both Stdexec_ROOT and HPX_WITH_FETCH_STDEXEC are provided. HPX_WITH_FETCH_STDEXEC is set to OFF."
    )
  endif()
elseif(HPX_WITH_STDEXEC)
  hpx_error(
    "HPX_WITH_STDEXEC is set to ON, but Stdexec_ROOT is not provided and HPX_WITH_FETCH_STDEXEC is not enabled. Please provide Stdexec_ROOT or set HPX_WITH_FETCH_STDEXEC to ON."
  )
endif()

# STDEXEC requires C++20
if(HPX_WITH_STDEXEC AND HPX_WITH_CXX_STANDARD LESS 20)
  hpx_error(
    "HPX_WITH_STDEXEC is set to ON, but HPX_WITH_CXX_STANDARD is less than 20. Please set HPX_WITH_CXX_STANDARD to 20 or higher."
  )
endif()

if(HPX_WITH_STDEXEC AND NOT TARGET STDEXEC::stdexec)
  hpx_add_config_define(HPX_HAVE_STDEXEC)

  if(HPX_WITH_FETCH_STDEXEC)
    hpx_info(
      "HPX_WITH_FETCH_STDEXEC=${HPX_WITH_FETCH_STDEXEC}, Stdexec will be fetched using CMake's FetchContent."
    )
    if(UNIX)
      include(FetchContent)
      message("FETCHING STDEXEC")
      fetchcontent_declare(
        Stdexec
        GIT_REPOSITORY https://github.com/NVIDIA/stdexec.git
        GIT_TAG ${HPX_WITH_STDEXEC_TAG}
      )
      fetchcontent_makeavailable(Stdexec)
    endif()

    # add_library(STDEXEC::stdexec INTERFACE IMPORTED)
    # target_include_directories(STDEXEC::stdexec INTERFACE
    # ${stdexec_SOURCE_DIR}) target_link_libraries(STDEXEC::stdexec INTERFACE
    # ${Stdexec_LIBRARY})
  else()
    find_package(Stdexec REQUIRED)

    if(NOT Stdexec_FOUND)
      hpx_error(
        "Stdexec could not be found, please specify Stdexec_ROOT to point to the correct location"
      )
    endif()
  endif()
endif()
