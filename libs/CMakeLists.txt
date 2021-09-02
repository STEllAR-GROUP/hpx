# Copyright (c) 2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

include(HPXLocal_Message)
include(HPXLocal_AddPseudoDependencies)
include(HPXLocal_AddPseudoTarget)

# add example pseudo targets needed for modules
if(HPXLocal_WITH_EXAMPLES)
  hpx_local_add_pseudo_target(examples.modules)
  hpx_local_add_pseudo_dependencies(examples examples.modules)
endif()

# add test pseudo targets needed for modules
hpx_local_warn("HPXLocal_WITH_TESTS=${HPXLocal_WITH_TESTS}")
if(HPXLocal_WITH_TESTS)
  if(HPXLocal_WITH_TESTS_UNIT)
    hpx_local_add_pseudo_target(tests.unit.modules)
    hpx_local_add_pseudo_dependencies(tests.unit tests.unit.modules)
  endif()

  if(HPXLocal_WITH_EXAMPLES AND HPXLocal_WITH_TESTS_EXAMPLES)
    hpx_local_add_pseudo_target(tests.examples.modules)
    hpx_local_add_pseudo_dependencies(tests.examples tests.examples.modules)
  endif()

  if(HPXLocal_WITH_TESTS_REGRESSIONS)
    hpx_local_add_pseudo_target(tests.regressions.modules)
    hpx_local_add_pseudo_dependencies(
      tests.regressions tests.regressions.modules
    )
  endif()

  if(HPXLocal_WITH_TESTS_BENCHMARKS)
    hpx_local_add_pseudo_target(tests.performance.modules)
    hpx_local_add_pseudo_dependencies(
      tests.performance tests.performance.modules
    )
  endif()

  if(HPXLocal_WITH_TESTS_HEADERS)
    hpx_local_add_pseudo_target(tests.headers.modules)
    hpx_local_add_pseudo_dependencies(tests.headers tests.headers.modules)
  endif()
endif()

hpx_local_info("")
hpx_local_info("Configuring modules:")

# Generate a file that lists all enabled modules for checks that might be
# optional based on the presence of a module or not
set(MODULE_ENABLED_MODULE_DEFINES)

# variables needed for config_strings_modules.hpp
set(CONFIG_STRINGS_MODULES_INCLUDES)
set(CONFIG_STRINGS_MODULES_ENTRIES)

string(TOUPPER local uppercase_lib)
add_library(hpx_local SHARED src/dummy.cpp)
target_compile_definitions(hpx_local PRIVATE HPX_LOCAL_EXPORTS)
set_target_properties(hpx_local PROPERTIES FOLDER "Local")
set_target_properties(
  hpx_local
  PROPERTIES VERSION ${HPXLocal_VERSION}
             SOVERSION ${HPXLocal_SOVERSION}
             CLEAN_DIRECT_OUTPUT 1
             OUTPUT_NAME hpx_local
             FOLDER "Local"
)
install(
  TARGETS hpx_local
  EXPORT HPXLocalTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT local
)
hpx_local_export_targets(hpx_local)

add_subdirectory(core)

# hpx_local_add_module populates HPXLocal_ENABLED_MODULES
foreach(module ${HPXLocal_ENABLED_MODULES})
  string(TOUPPER ${module} uppercase_module)

  set(MODULE_ENABLED_MODULE_DEFINES
      "${MODULE_ENABLED_MODULE_DEFINES}#define HPX_HAVE_MODULE_${uppercase_module}\n"
  )
endforeach()

configure_file(
  "${PROJECT_SOURCE_DIR}/cmake/templates/modules_enabled.hpp.in"
  "${PROJECT_BINARY_DIR}/libs/core/config_local/include/hpx/local/config/modules_enabled.hpp"
  @ONLY
)

if(MSVC AND HPXLocal_COROUTINES_WITH_SWAP_CONTEXT_EMULATION)
  target_link_options(hpx_local PRIVATE "/EXPORT:switch_to_fiber")
endif()

if(HPXLocal_WITH_DATAPAR_VC)
  target_link_libraries(hpx_local PUBLIC Vc::vc)
endif()

if(HPXLocal_WITH_ITTNOTIFY)
  target_link_libraries(hpx_local PUBLIC Amplifier::amplifier)
endif()
