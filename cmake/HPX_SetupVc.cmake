# Copyright (c) 2016 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Locate the Vc template library. Vc can be found at https://github.com/VcDevel/Vc
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

find_package(Vc ${Vc_FIND_VERSION} QUIET NO_MODULE PATHS ${Vc_ROOT})

if(Vc_FOUND)
  include_directories(SYSTEM ${Vc_INCLUDE_DIR})
  link_directories(${Vc_LIB_DIR})
  hpx_libraries(${Vc_LIBRARIES})

  hpx_add_config_define(HPX_HAVE_VC_DATAPAR)
endif()

