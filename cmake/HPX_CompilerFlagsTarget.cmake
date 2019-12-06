# Copyright (c) 2019 Mikael Simberg
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# This is a dummy target that we add compile flags to all tests will depend on
# this target and inherit the flags but user code linking against hpx will not
add_library(hpx_internal_flags INTERFACE)

# Default unnamed config (not Debug/Release/etc) are in this var
get_property(_temp_flags GLOBAL PROPERTY HPX_CMAKE_FLAGS_CXX_)
target_compile_options(hpx_internal_flags INTERFACE ${_temp_flags})

# Could potentially use CMAKE_CONFIGURATION_TYPES in case a user defined config exists
foreach(_config "DEBUG" "RELEASE" "RELWITHDEBINFO" "MINSIZEREL")
  get_property(_temp_flags GLOBAL PROPERTY HPX_CMAKE_FLAGS_CXX_${_config})
  target_compile_options(hpx_internal_flags INTERFACE $<$<CONFIG:${_config}>:${_temp_flags}>)
endforeach()

foreach(_keyword PUBLIC;PRIVATE)
  get_property(HPX_TARGET_COMPILE_OPTIONS_VAR
    GLOBAL PROPERTY HPX_TARGET_COMPILE_OPTIONS_${_keyword})
  foreach(_flag ${HPX_TARGET_COMPILE_OPTIONS_VAR})
    target_compile_options(hpx_internal_flags INTERFACE ${_flag})
  endforeach()
endforeach()

include(HPX_ExportTargets)
# Modules can't link to this if not exported
install(TARGETS hpx_internal_flags EXPORT HPXModulesTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  COMPONENT hpx_internal_flags
)
hpx_export_modules_targets(hpx_internal_flags)
