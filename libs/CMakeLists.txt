# Copyright (c) 2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

include(HPX_Message)
include(HPX_AddPseudoDependencies)
include(HPX_AddPseudoTarget)

# add example pseudo targets needed for modules
if(HPX_WITH_EXAMPLES)
  add_hpx_pseudo_target(examples.modules)
  add_hpx_pseudo_dependencies(examples examples.modules)
endif()

# add test pseudo targets needed for modules
if(HPX_WITH_TESTS)
  if(HPX_WITH_TESTS_UNIT)
    add_hpx_pseudo_target(tests.unit.modules)
    add_hpx_pseudo_dependencies(tests.unit tests.unit.modules)
  endif()

  if(HPX_WITH_EXAMPLES AND HPX_WITH_TESTS_EXAMPLES)
    add_hpx_pseudo_target(tests.examples.modules)
    add_hpx_pseudo_dependencies(tests.examples tests.examples.modules)
  endif()

  if(HPX_WITH_TESTS_REGRESSIONS)
    add_hpx_pseudo_target(tests.regressions.modules)
    add_hpx_pseudo_dependencies(tests.regressions tests.regressions.modules)
  endif()

  if(HPX_WITH_TESTS_BENCHMARKS)
    add_hpx_pseudo_target(tests.performance.modules)
    add_hpx_pseudo_dependencies(tests.performance tests.performance.modules)
  endif()

  if(HPX_WITH_TESTS_HEADERS)
    add_hpx_pseudo_target(tests.headers.modules)
    add_hpx_pseudo_dependencies(tests.headers tests.headers.modules)
  endif()
endif()

hpx_info("")
hpx_info("Configuring modules:")

# Generate a file that lists all enabled modules for checks that might be
# optional based on the presence of a module or not
set(MODULE_ENABLED_MODULE_DEFINES)

# variables needed for config_strings_modules.hpp
set(CONFIG_STRINGS_MODULES_INCLUDES)
set(CONFIG_STRINGS_MODULES_ENTRIES)

set(HPX_LIBS
    core full parallelism
    CACHE INTERNAL "List of HPX libraries" FORCE
)

foreach(lib ${HPX_LIBS})
  # NOTE: hpx_full (i.e. the target representing libhpx.so) is still created in
  # src/CMakeLists.txt. When all of hpx_full has been modularized the target
  # creation can move here as well.
  if(NOT ${lib} STREQUAL "full")
    string(TOUPPER ${lib} uppercase_lib)
    add_library(hpx_${lib} SHARED src/dummy.cpp)
    target_compile_definitions(hpx_${lib} PRIVATE HPX_${uppercase_lib}_EXPORTS)

    set_target_properties(hpx_${lib} PROPERTIES FOLDER "Core")

    install(
      TARGETS hpx_${lib}
      EXPORT HPXInternalTargets
      LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
      ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
      RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT ${lib}
    )

    hpx_export_internal_targets(hpx_${lib})
  endif()

  add_subdirectory(${lib})
endforeach()

# add_hpx_module populates HPX_ENABLED_MODULES
foreach(module ${HPX_ENABLED_MODULES})
  string(TOUPPER ${module} uppercase_module)

  set(MODULE_ENABLED_MODULE_DEFINES
      "${MODULE_ENABLED_MODULE_DEFINES}#define HPX_HAVE_MODULE_${uppercase_module}\n"
  )
endforeach()

configure_file(
  "${PROJECT_SOURCE_DIR}/cmake/templates/modules_enabled.hpp.in"
  "${PROJECT_BINARY_DIR}/libs/core/config/include/hpx/config/modules_enabled.hpp"
  @ONLY
)

# FIXME : temporary add public dependencies to the boost libraries which are
# only linked in the modules (which are themselves privately linked to hpx), it
# should be removed when we make the modules object libraries that we publicly
# link to hpx
target_link_libraries(hpx_core PUBLIC hpx_dependencies_boost)
target_link_libraries(hpx_core PUBLIC hpx_dependencies_allocator)
target_link_libraries(hpx_core PUBLIC Hwloc::hwloc)

if(HPX_FILESYSTEM_WITH_BOOST_FILESYSTEM_COMPATIBILITY)
  target_link_libraries(hpx_core PUBLIC Boost::filesystem)
endif()

if(MSVC AND HPX_COROUTINES_WITH_SWAP_CONTEXT_EMULATION)
  target_link_options(hpx_core PRIVATE "/EXPORT:switch_to_fiber")
endif()
