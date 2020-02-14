# Copyright (c) 2019 Mikael Simberg
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# This is a dummy target that we add compile flags to all tests will depend on
# this target and inherit the flags but user code linking against hpx will not
add_library(hpx_private_flags INTERFACE)
add_library(hpx_public_flags INTERFACE)

# Default unnamed config (not Debug/Release/etc) are in this var
get_property(_temp_flags GLOBAL PROPERTY HPX_CMAKE_FLAGS_CXX_)
target_compile_options(hpx_private_flags INTERFACE ${_temp_flags})

# Could potentially use CMAKE_CONFIGURATION_TYPES in case a user defined config exists
foreach(_config "DEBUG" "RELEASE" "RELWITHDEBINFO" "MINSIZEREL")
  get_property(_temp_flags GLOBAL PROPERTY HPX_CMAKE_FLAGS_CXX_${_config})
  target_compile_options(hpx_private_flags INTERFACE $<$<CONFIG:${_config}>:${_temp_flags}>)
endforeach()

foreach(_keyword PUBLIC;PRIVATE)
  set(_target)
  if(_keyword STREQUAL "PUBLIC")
    set(_target hpx_public_flags)
  else()
    set(_target hpx_private_flags)
  endif()

  get_property(HPX_TARGET_COMPILE_DEFINITIONS_VAR
    GLOBAL PROPERTY HPX_TARGET_COMPILE_DEFINITIONS_${_keyword})
  foreach(_flag ${HPX_TARGET_COMPILE_DEFINITIONS_VAR})
    target_compile_definitions(${_target} ${_keyword} ${_flag})
  endforeach()

  get_property(HPX_TARGET_COMPILE_OPTIONS_VAR
    GLOBAL PROPERTY HPX_TARGET_COMPILE_OPTIONS_${_keyword})
  foreach(_flag ${HPX_TARGET_COMPILE_OPTIONS_VAR})
    target_compile_options(${_target} INTERFACE ${_flag})
  endforeach()
endforeach()

target_compile_features(hpx_private_flags INTERFACE cxx_std_${HPX_CXX_STANDARD})
target_compile_features(hpx_public_flags INTERFACE cxx_std_${HPX_CXX_STANDARD})

target_compile_definitions(hpx_public_flags PUBLIC $<$<CONFIG:Debug>:HPX_DEBUG>)

include(HPX_ExportTargets)
# Modules can't link to this if not exported
install(TARGETS hpx_private_flags EXPORT HPXModulesTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  COMPONENT hpx_private_flags
)
install(TARGETS hpx_public_flags EXPORT HPXModulesTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  COMPONENT hpx_public_flags
)
hpx_export_modules_targets(hpx_private_flags)
hpx_export_modules_targets(hpx_public_flags)
