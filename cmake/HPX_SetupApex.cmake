# Copyright (c) 2007-2022 Hartmut Kaiser
# Copyright (c) 2011-2014 Thomas Heller
# Copyright (c) 2007-2008 Chirag Dekate
# Copyright (c)      2011 Bryce Lelbach
# Copyright (c)      2011 Vinay C Amatya
# Copyright (c)      2013 Jeroen Habraken
# Copyright (c) 2014-2016 Andreas Schaefer
# Copyright (c) 2017      Abhimanyu Rawat
# Copyright (c) 2017      Google
# Copyright (c) 2017      Taeguk Kwon
# Copyright (c) 2018 Christopher Hinz
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(HPX_WITH_APEX AND NOT TARGET APEX::apex)
  if(NOT HPX_FIND_PACKAGE)
    if(NOT "${APEX_ROOT}" AND "$ENV{APEX_ROOT}")
      set(APEX_ROOT "$ENV{APEX_ROOT}")
    endif()

    # We want to track parent dependencies
    hpx_add_config_define(HPX_HAVE_THREAD_PARENT_REFERENCE)

    if(APEX_ROOT)
      # Use given (external) APEX
      set(HPX_APEX_ROOT ${APEX_ROOT})

    else()
      # If APEX_ROOT not specified, local clone into hpx source dir
      include(FetchContent)
      fetchcontent_declare(
        apex
        GIT_REPOSITORY https://github.com/UO-OACISS/apex.git
        GIT_TAG ${HPX_WITH_APEX_TAG}
      )

      fetchcontent_getproperties(apex)
      if(NOT apex_POPULATED)
        # maintain compatibility
        if(HPX_WITH_APEX_NO_UPDATE)
          set(FETCHCONTENT_UPDATES_DISCONNECTED_APEX ON)
        endif()
        fetchcontent_populate(apex)
      endif()

      # check again to make sure we have received a copy of APEX
      fetchcontent_getproperties(apex)
      if(NOT apex_POPULATED)
        hpx_error("APEX could not be populated with HPX_WITH_APEX=On")
      endif()
      set(APEX_ROOT ${apex_SOURCE_DIR})

      hpx_info("APEX_ROOT is not set. Cloning APEX into ${apex_SOURCE_DIR}.")
    endif()

    list(APPEND CMAKE_MODULE_PATH "${APEX_ROOT}/cmake/Modules")
    add_subdirectory(${APEX_ROOT}/src/apex ${CMAKE_BINARY_DIR}/apex/src/apex)
    if(AMPLIFIER_FOUND)
      hpx_error("AMPLIFIER_FOUND has been set. Please disable the use of the \
        Intel Amplifier (WITH_AMPLIFIER=Off) in order to use APEX"
      )
    endif()
  endif()

  add_library(APEX::apex INTERFACE IMPORTED)
  if(HPX_FIND_PACKAGE)
    target_link_libraries(APEX::apex INTERFACE HPX::apex)
  else()
    target_link_libraries(APEX::apex INTERFACE apex)
  endif()
  if(UNIX AND NOT APPLE)
    target_link_options(APEX::apex INTERFACE "-Wl,-no-as-needed")
  endif()

  if((HPX_WITH_NETWORKING AND HPX_WITH_PARCELPORT_MPI) OR HPX_WITH_ASYNC_MPI)
    # APEX now depends on MPI itself
    if(NOT TARGET Mpi::mpi)
      include(HPX_SetupMPI)
      hpx_setup_mpi()
    endif()
    target_link_libraries(APEX::apex INTERFACE Mpi::mpi)
  endif()

  # handle optional ITTNotify library (private dependency, skip when called in
  # find_package(HPX))
  if(HPX_WITH_ITTNOTIFY AND NOT HPX_FIND_PACKAGE)
    add_subdirectory(${APEX_ROOT}/src/ITTNotify)
    if(NOT ITTNOTIFY_FOUND)
      hpx_error("ITTNotify could not be found and HPX_WITH_ITTNOTIFY=On")
    endif()

    add_library(ITTNotify::ittnotify INTERFACE IMPORTED)
    target_include_directories(
      ITTNotify::ittnotify SYSTEM INTERFACE ${ITTNOTIFY_SOURCE_DIR}
    )
    target_link_libraries(APEX::apex INTERFACE ITTNotify::ittnotify)
    hpx_add_config_define(HPX_HAVE_ITTNOTIFY 1)
  endif()
endif()
