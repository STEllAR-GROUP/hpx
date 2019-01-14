# Copyright (c) 2016 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Locate the Vc template library.
# Vc can be found at https://github.com/VcDevel/Vc
#
# This file is meant to be copied into projects that want to use Vc. It will
# search for VcConfig.cmake, which ships with Vc and will provide up-to-date
# buildsystem changes. Thus there should not be any need to update FindVc.cmake
# again after you integrated it into your project.
#
# This module defines the following variables:
# Vc_FOUND
# Vc_INCLUDE_DIR
# Vc_LIBRARIES
# Vc_DEFINITIONS
# Vc_COMPILE_FLAGS
# Vc_ARCHITECTURE_FLAGS
# Vc_ALL_FLAGS (the union of the above three variables)
# Vc_VERSION_MAJOR
# Vc_VERSION_MINOR
# Vc_VERSION_PATCH
# Vc_VERSION
# Vc_VERSION_STRING
# Vc_INSTALL_DIR
# Vc_LIB_DIR
# Vc_CMAKE_MODULES_DIR
#
# The following two variables are set according to the compiler used. Feel free
# to use them to skip whole compilation units.
# Vc_SSE_INTRINSICS_BROKEN
# Vc_AVX_INTRINSICS_BROKEN

find_package(Vc ${Vc_FIND_VERSION} QUIET)

if(NOT Vc_FOUND)
  if(NOT Vc_VERSION_STRING)
    # didn't find any version of Vc
    hpx_error("Vc was not found while datapar support was requested. Set Vc_DIR to the installation path of Vc")
  elseif(${Vc_VERSION_STRING} VERSION_LESS "1.70.0")
    # didn't find current version of Vc
    hpx_error("The Vc was found for requested datapar support was too old. Set Vc_DIR to the installation path of Vc (V1.70.0 is required)")
  endif()
endif()

if(Vc_VERSION_STRING AND (NOT ${Vc_VERSION_STRING} VERSION_LESS "1.70.0"))
  # found Vc V2
  if(NOT Vc_INCLUDE_DIR)
    hpx_error("Vc was not found while datapar support was requested. Set Vc_DIR to the installation path of Vc")
  endif()
  set(HPX_WITH_DATAPAR_VC_NO_LIBRARY On)
endif()

add_library(hpx::vc INTERFACE IMPORTED)
set_property(TARGET hpx::vc PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${Vc_INCLUDE_DIR})

if(NOT HPX_WITH_DATAPAR_VC_NO_LIBRARY)
  set_property(TARGET hpx::vc PROPERTY INTERFACE_LINK_LIBRARIES ${Vc_LIBRARIES})
endif()

foreach(_flag ${Vc_DEFINITIONS})
  # remove leading '-D'
  string(STRIP ${_flag} _flag)
  string(FIND ${_flag} "-D" _flagpos)
  if(${_flagpos} EQUAL 0)
    string(SUBSTRING ${_flag} 2 -1 _flag)
  endif()

  set_property(TARGET hpx::vc APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS ${_flag})
endforeach()

# do not include Vc build flags for MSVC builds as this breaks building the
# core HPX libraries itself
if(NOT MSVC)
  set_property(TARGET hpx::vc PROPERTY INTERFACE_COMPILE_OPTIONS ${Vc_COMPILE_FLAGS} ${Vc_ARCHITECTURE_FLAGS})
 else()
  hpx_add_config_cond_define(_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS)
endif()

hpx_add_config_define(HPX_HAVE_DATAPAR)
hpx_add_config_define(HPX_HAVE_DATAPAR_VC)

hpx_info("Found Vc (vectorization):" ${Vc_INCLUDE_DIR} "- version:" ${Vc_VERSION_STRING})

