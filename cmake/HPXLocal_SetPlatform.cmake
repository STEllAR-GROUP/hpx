# Copyright (c) 2014 Thomas Heller
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(HPXLocal_PLATFORM_CHOICES
    "Choices are: native, Android, XeonPhi, BlueGeneQ."
)
set(HPXLocal_PLATFORMS_UC "NATIVE;ANDROID;XEONPHI;BLUEGENEQ")

if(NOT HPXLocal_PLATFORM)
  set(HPXLocal_PLATFORM
      "native"
      CACHE
        STRING
        "Sets special compilation flags for specific platforms. ${HPXLocal_PLATFORM_CHOICES}"
  )
else()
  set(HPXLocal_PLATFORM
      "${HPXLocal_PLATFORM}"
      CACHE
        STRING
        "Sets special compilation flags for specific platforms. ${HPXLocal_PLATFORM_CHOICES}"
  )
endif()

if(NOT HPXLocal_PLATFORM STREQUAL "")
  string(TOUPPER ${HPXLocal_PLATFORM} HPXLocal_PLATFORM_UC)
else()
  set(HPXLocal_PLATFORM
      "native"
      CACHE
        STRING
        "Sets special compilation flags for specific platforms. ${HPXLocal_PLATFORM_CHOICES}"
        FORCE
  )
  set(HPXLocal_PLATFORM_UC "NATIVE")
endif()

string(FIND "${HPXLocal_PLATFORMS_UC}" "${HPXLocal_PLATFORM_UC}"
            _PLATFORM_FOUND
)
if(_PLATFORM_FOUND EQUAL -1)
  hpx_local_error(
    "Unknown platform in HPXLocal_PLATFORM. ${HPXLocal_PLATFORM_CHOICES}"
  )
endif()
