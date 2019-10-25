# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

include(HPX_Option)
# Compatibility with using Boost.ProgramOptions, introduced in V1.4.0
hpx_option(HPX_PROGRAM_OPTIONS_WITH_BOOST_PROGRAM_OPTIONS_COMPATIBILITY
  BOOL "Enable Boost.ProgramOptions compatibility. (default: ON)"
  ON ADVANCED CATEGORY "Modules")

if(HPX_PROGRAM_OPTIONS_WITH_BOOST_PROGRAM_OPTIONS_COMPATIBILITY)
  set(__boost_program_options hpx::boost::program_options)
endif()

# Creates imported hpx::boost::program_options target
if(HPX_PROGRAM_OPTIONS_WITH_BOOST_PROGRAM_OPTIONS_COMPATIBILITY)

  hpx_add_config_define_namespace(
    DEFINE HPX_PROGRAM_OPTIONS_HAVE_BOOST_PROGRAM_OPTIONS_COMPATIBILITY
    NAMESPACE PROGRAM_OPTIONS)

  find_package(Boost ${Boost_MINIMUM_VERSION}
    QUIET MODULE
    COMPONENTS program_options)

  if(NOT Boost_PROGRAM_OPTIONS_FOUND)
    hpx_error("Could not find Boost.ProgramOptions. Provide a boost installation including the program_options library")
  endif()

  add_library(hpx::boost::program_options INTERFACE IMPORTED)

  set_property(TARGET hpx::boost::program_options APPEND PROPERTY
    INTERFACE_INCLUDE_DIRECTORIES ${Boost_INCLUDE_DIRS})
  if(${CMAKE_VERSION} VERSION_LESS "3.12.0")
    set_property(TARGET hpx::boost::program_options PROPERTY INTERFACE_LINK_LIBRARIES
      ${Boost_PROGRAM_OPTIONS_LIBRARIES})
  else()
    target_link_libraries(hpx::boost::program_options INTERFACE
      ${Boost_PROGRAM_OPTIONS_LIBRARIES})
  endif()

endif()
