# Copyright (c) 2019 Mikael Simberg
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# These are a dummy targets that we add compile flags to. All HPX targets should
# link to them.
add_library(hpx_private_flags INTERFACE)
add_library(hpx_public_flags INTERFACE)

# Set C++ standard
target_compile_features(hpx_private_flags INTERFACE cxx_std_${HPX_CXX_STANDARD})
target_compile_features(hpx_public_flags INTERFACE cxx_std_${HPX_CXX_STANDARD})

# Set other flags that should always be set

# HPX_DEBUG must be set without a generator expression as it determines ABI
# compatibility. Projects in Release mode using HPX in Debug mode must have
# HPX_DEBUG set, and projects in Debug mode using HPX in Release mode must not
# have HPX_DEBUG set. HPX_DEBUG must also not be set by projects using HPX.
if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  target_compile_definitions(hpx_private_flags INTERFACE HPX_DEBUG)
  target_compile_definitions(hpx_public_flags INTERFACE HPX_DEBUG)
endif()

target_compile_definitions(
  hpx_private_flags
  INTERFACE $<$<CONFIG:MinSizeRel>:NDEBUG>
  INTERFACE $<$<CONFIG:Release>:NDEBUG>
  INTERFACE $<$<CONFIG:RelWithDebInfo>:NDEBUG>
)

# Remaining flags are set through the macros in cmake/HPX_AddCompileFlag.cmake

include(HPX_ExportTargets)
# Modules can't link to this if not exported
install(
  TARGETS hpx_private_flags
  EXPORT HPXInternalTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT hpx_private_flags
)
install(
  TARGETS hpx_public_flags
  EXPORT HPXInternalTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT hpx_public_flags
)
hpx_export_internal_targets(hpx_private_flags)
hpx_export_internal_targets(hpx_public_flags)
