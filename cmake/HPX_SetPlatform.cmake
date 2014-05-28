# Copyright (c) 2014 Thomas Heller
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(HPX_PLATFORM_CHOICES "Choices are: native, Android, XeonPhi, BlueGeneQ.")

if(NOT HPX_PLATFORM)
  set(HPX_PLATFORM "native" CACHE STRING "Sets special compilation flags for specific platforms. ${HPX_PLATFORM_CHOICES}")
else()
  set(HPX_PLATFORM "${HPX_PLATFORM}" CACHE STRING "Sets special compilation flags for specific platforms. ${HPX_PLATFORM_CHOICES}")
endif()

if(NOT HPX_PLATFORM STREQUAL "")
  string(TOUPPER ${HPX_PLATFORM} HPX_PLATFORM_UC)
else()
  set(HPX_PLATFORM "native" CACHE STRING "Sets special compilation flags for specific platforms. ${HPX_PLATFORM_CHOICES}" FORCE)
  set(HPX_PLATFORM_UC "NATIVE")
endif()

if(HPX_PLATFORM_UC STREQUAL "NATIVE")
  hpx_info("Compiling with the native toolset")
elseif(HPX_PLATFORM_UC STREQUAL "ANDROID")
  hpx_info("Compiling for Android devices")
elseif(HPX_PLATFORM_UC STREQUAL "XEONPHI")
  hpx_info("Compiling for Intel Xeon Phi devices")
  if(NOT ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel"))
    hpx_error("HPX on the MIC can only be compiled with the Intel compiler.")
  endif()
  hpx_add_config_define(HPX_NATIVE_MIC)
elseif(HPX_PLATFORM_UC STREQUAL "BLUEGENEQ")
  hpx_info("Compiling for BlueGene/Q")
else()
  hpx_error("Unknown platform in HPX_PLATFORM. ${HPX_PLATFORM_CHOICES}")
endif()
