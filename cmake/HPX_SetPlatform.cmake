# Copyright (c) 2014 Thomas Heller
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(HPX_PLATFORM_CHOICES "Choices are: native, Android, XeonPhi, BlueGeneQ.")
set(HPX_PLATFORMS_UC "NATIVE;ANDROID;XEONPHI;BLUEGENEQ")

if(NOT HPX_PLATFORM)
  set(HPX_PLATFORM
      "native"
      CACHE
        STRING
        "Sets special compilation flags for specific platforms. ${HPX_PLATFORM_CHOICES}"
  )
else()
  set(HPX_PLATFORM
      "${HPX_PLATFORM}"
      CACHE
        STRING
        "Sets special compilation flags for specific platforms. ${HPX_PLATFORM_CHOICES}"
  )
endif()

if(NOT HPX_PLATFORM STREQUAL "")
  string(TOUPPER ${HPX_PLATFORM} HPX_PLATFORM_UC)
else()
  set(HPX_PLATFORM
      "native"
      CACHE
        STRING
        "Sets special compilation flags for specific platforms. ${HPX_PLATFORM_CHOICES}"
        FORCE
  )
  set(HPX_PLATFORM_UC "NATIVE")
endif()

string(FIND "${HPX_PLATFORMS_UC}" "${HPX_PLATFORM_UC}" _PLATFORM_FOUND)
if(_PLATFORM_FOUND EQUAL -1)
  hpx_error("Unknown platform in HPX_PLATFORM. ${HPX_PLATFORM_CHOICES}")
endif()
