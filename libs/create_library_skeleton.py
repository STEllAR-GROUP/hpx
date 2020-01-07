#!/usr/bin/env python3
'''
Copyright (c) 2018 Thomas Heller

SPDX-License-Identifier: BSL-1.0
Distributed under the Boost Software License, Version 1.0. (See accompanying
file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

create_library_skeleton.py - A tool to generate a library skeleton to be used
as library component of HPX
'''

import sys, os

if len(sys.argv) != 2:
    print('Usage: %s <lib_name>' % sys.argv[0])
    print('Generates the skeleton for lib_name in the current working directory')
    sys.exit(1)

lib_name = sys.argv[1]
lib_name_upper = lib_name.upper()
header_str = '=' * len(lib_name)

# CMake minimum version
cmake_version = '3.3.2'

cmake_header = f'''# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
'''

readme_template = f'''
..
    Copyright (c) 2019 The STE||AR-Group

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

{header_str}
{lib_name}
{header_str}

This library is part of HPX.

Documentation can be found `here
<https://stellar-group.github.io/hpx/docs/sphinx/latest/html/libs/{lib_name}/docs/index.html>`__.
'''

index_rst = f'''..
    Copyright (c) 2019 The STE||AR-Group

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _libs_{lib_name}:

{header_str}
{lib_name}
{header_str}

TODO: High-level description of the library.

See the :ref:`API reference <libs_{lib_name}_api>` of this module for more
details.

'''

root_cmakelists_template = cmake_header + f'''
cmake_minimum_required(VERSION {cmake_version} FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${{CMAKE_CURRENT_SOURCE_DIR}}/cmake")

# Default location is $HPX_ROOT/libs/{lib_name}/include
set({lib_name}_headers)

# Default location is $HPX_ROOT/libs/{lib_name}/include_compatibility
set({lib_name}_compat_headers)

set({lib_name}_sources)

include(HPX_AddModule)
add_hpx_module({lib_name}
  COMPATIBILITY_HEADERS OFF
  DEPRECATION_WARNINGS
  FORCE_LINKING_GEN
  GLOBAL_HEADER_GEN OFF
  SOURCES ${{{lib_name}_sources}}
  HEADERS ${{{lib_name}_headers}}
  COMPAT_HEADERS ${{{lib_name}_compat_headers}}
  DEPENDENCIES
  CMAKE_SUBDIRS examples tests
)
'''

examples_cmakelists_template = cmake_header + f'''
if (HPX_WITH_EXAMPLES)
  add_hpx_pseudo_target(examples.modules.{lib_name})
  add_hpx_pseudo_dependencies(examples.modules examples.modules.{lib_name})
  if (HPX_WITH_TESTS AND HPX_WITH_TESTS_EXAMPLES AND HPX_{lib_name_upper}_WITH_TESTS)
    add_hpx_pseudo_target(tests.examples.modules.{lib_name})
    add_hpx_pseudo_dependencies(tests.examples.modules tests.examples.modules.{lib_name})
  endif()
endif()
'''

tests_cmakelists_template = cmake_header + f'''
include(HPX_Message)
include(HPX_Option)

if (NOT HPX_WITH_TESTS AND HPX_TOP_LEVEL)
  hpx_set_option(HPX_{lib_name_upper}_WITH_TESTS VALUE OFF FORCE)
  return()
endif()

if (HPX_{lib_name_upper}_WITH_TESTS)
    if (HPX_WITH_TESTS_UNIT)
      add_hpx_pseudo_target(tests.unit.modules.{lib_name})
      add_hpx_pseudo_dependencies(tests.unit.modules tests.unit.modules.{lib_name})
      add_subdirectory(unit)
    endif()

    if (HPX_WITH_TESTS_REGRESSIONS)
      add_hpx_pseudo_target(tests.regressions.modules.{lib_name})
      add_hpx_pseudo_dependencies(tests.regressions.modules tests.regressions.modules.{lib_name})
      add_subdirectory(regressions)
    endif()

    if (HPX_WITH_TESTS_BENCHMARKS)
      add_hpx_pseudo_target(tests.performance.modules.{lib_name})
      add_hpx_pseudo_dependencies(tests.performance.modules tests.performance.modules.{lib_name})
      add_subdirectory(performance)
    endif()

    if (HPX_WITH_TESTS_HEADERS)
      add_hpx_header_tests(
        modules.{lib_name}
        HEADERS ${{{lib_name}_headers}}
        HEADER_ROOT ${{PROJECT_SOURCE_DIR}}/include
        NOLIBS
        DEPENDENCIES hpx_{lib_name})
    endif()
endif()
'''

if lib_name != '--recreate-index':
    def mkdir(path):
        if not os.path.exists(path):
            os.makedirs(path)

    mkdir(lib_name)

    ################################################################################
    # Generate basic directory structure
    for subdir in ['docs', 'examples', 'include', 'src', 'tests']:
        path = os.path.join(lib_name, subdir)
        mkdir(path)
    # Generate include directory structure
    # Normalize path...
    include_path = ''.join(lib_name)
    path = os.path.join(lib_name, 'include', 'hpx', include_path)
    mkdir(path)
    path = os.path.join(lib_name, 'tests', 'unit')
    mkdir(path)
    path = os.path.join(lib_name, 'tests', 'regressions')
    mkdir(path)
    path = os.path.join(lib_name, 'tests', 'performance')
    mkdir(path)
    ################################################################################

    ################################################################################
    # Generate README skeleton
    f = open(os.path.join(lib_name, 'README.rst'), 'w')
    f.write(readme_template)
    ################################################################################

    ################################################################################
    # Generate CMakeLists.txt skeletons

    # Generate top level CMakeLists.txt
    f = open(os.path.join(lib_name, 'CMakeLists.txt'), 'w')
    f.write(root_cmakelists_template)

    # Generate docs/index.rst
    f = open(os.path.join(lib_name, 'docs', 'index.rst'), 'w')
    f.write(index_rst)

    # Generate examples/CMakeLists.txt
    f = open(os.path.join(lib_name, 'examples', 'CMakeLists.txt'), 'w')
    f.write(examples_cmakelists_template)

    # Generate tests/CMakeLists.txt
    f = open(os.path.join(lib_name, 'tests', 'CMakeLists.txt'), 'w')
    f.write(tests_cmakelists_template)

    # Generate tests/unit/CMakeLists.txt
    f = open(os.path.join(lib_name, 'tests', 'unit', 'CMakeLists.txt'), 'w')
    f.write(cmake_header)

    # Generate tests/regressions/CMakeLists.txt
    f = open(os.path.join(lib_name, 'tests', 'regressions', 'CMakeLists.txt'), 'w')
    f.write(cmake_header)
    f.write('\n')

    # Generate tests/performance/CMakeLists.txt
    f = open(os.path.join(lib_name, 'tests', 'performance', 'CMakeLists.txt'), 'w')
    f.write(cmake_header)
    ################################################################################

################################################################################

# Scan directory to get all libraries...
cwd = os.getcwd()
libs = sorted([ lib for lib in os.listdir(cwd) if os.path.isdir(lib) ])

# Adapting top level CMakeLists.txt
libs_cmakelists = cmake_header + f'''
# This file is auto generated. Please do not edit manually.
'''

libs_cmakelists += '''
include(HPX_Message)
include(HPX_AddPseudoDependencies)
include(HPX_AddPseudoTarget)

set(HPX_CANDIDATE_LIBS
'''
for lib in libs:
    if not lib.startswith('_'):
        libs_cmakelists += f'  {lib}\n'
libs_cmakelists += ')\n\n'

libs_cmakelists += '''
# add example pseudo targets needed for modules
if(HPX_WITH_EXAMPLES)
  add_hpx_pseudo_target(examples.modules)
  add_hpx_pseudo_dependencies(examples examples.modules)
endif()

# add test pseudo targets needed for modules
if(HPX_WITH_TESTS)
  if (HPX_WITH_TESTS_UNIT)
    add_hpx_pseudo_target(tests.unit.modules)
    add_hpx_pseudo_dependencies(tests.unit tests.unit.modules)
  endif()

  if (HPX_WITH_EXAMPLES AND HPX_WITH_TESTS_EXAMPLES)
    add_hpx_pseudo_target(tests.examples.modules)
    add_hpx_pseudo_dependencies(tests.examples tests.examples.modules)
  endif()

  if (HPX_WITH_TESTS_REGRESSIONS)
    add_hpx_pseudo_target(tests.regressions.modules)
    add_hpx_pseudo_dependencies(tests.regressions tests.regressions.modules)
  endif()

  if (HPX_WITH_TESTS_BENCHMARKS)
    add_hpx_pseudo_target(tests.performance.modules)
    add_hpx_pseudo_dependencies(tests.performance tests.performance.modules)
  endif()

  if (HPX_WITH_TESTS_HEADERS)
    add_custom_target(tests.headers.modules)
    add_hpx_pseudo_dependencies(tests.headers tests.headers.modules)
  endif()
endif()

'''

libs_cmakelists += '''
hpx_info("")
hpx_info("Configuring modules:")

# variables needed for modules.cpp
set(MODULE_FORCE_LINKING_INCLUDES)
set(MODULE_FORCE_LINKING_CALLS)

# variables needed for config_strings_modules.hpp
set(CONFIG_STRINGS_MODULES_INCLUDES)
set(CONFIG_STRINGS_MODULES_ENTRIES)

foreach(lib ${HPX_CANDIDATE_LIBS})
  # if the module is successfully added, xxx_LIBRARY_ENABLED will be ON
  add_subdirectory(${lib})

  get_property(HPX_${lib}_LIBRARY_ENABLED GLOBAL PROPERTY HPX_${lib}_LIBRARY_ENABLED)
  if (HPX_${lib}_LIBRARY_ENABLED)
    set(HPX_LIBS ${HPX_LIBS} ${lib} CACHE INTERNAL "list of Enabled HPX modules" FORCE)

    set(MODULE_FORCE_LINKING_INCLUDES
      "${MODULE_FORCE_LINKING_INCLUDES}#include <hpx/${lib}/force_linking.hpp>\\n")

    set(MODULE_FORCE_LINKING_CALLS
      "${MODULE_FORCE_LINKING_CALLS}\\n        ${lib}::force_linking();")

    set(CONFIG_STRINGS_MODULES_INCLUDES
      "${CONFIG_STRINGS_MODULES_INCLUDES}#include <hpx/${lib}/config/config_strings.hpp>\\n")
    set(CONFIG_STRINGS_MODULES_ENTRIES
      "${CONFIG_STRINGS_MODULES_ENTRIES}\\n        { \\"${lib}\\", ${lib}::config_strings },")
  endif()
endforeach()

configure_file(
    "${PROJECT_SOURCE_DIR}/cmake/templates/modules.cpp.in"
    "${PROJECT_BINARY_DIR}/libs/modules.cpp"
    @ONLY)

configure_file(
  "${PROJECT_SOURCE_DIR}/cmake/templates/config_defines_strings_modules.hpp.in"
  "${PROJECT_BINARY_DIR}/hpx/config/config_defines_strings_modules.hpp"
  @ONLY)
configure_file(
  "${PROJECT_SOURCE_DIR}/cmake/templates/config_defines_strings_modules.hpp.in"
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/hpx/config/config_defines_strings_modules.hpp"
  @ONLY)
'''

f = open(os.path.join(cwd, 'CMakeLists.txt'), 'w')
f.write(libs_cmakelists)

# Adapting all_modules.rst
all_modules_rst = f'''..
    Copyright (c) 2018-2019 The STE||AR-Group

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _all_modules:

===========
All modules
===========

.. toctree::
   :maxdepth: 2

'''
for lib in libs:
    all_modules_rst += f'   /libs/{lib}/docs/index.rst\n'

f = open(os.path.join(cwd, 'all_modules.rst'), 'w')
f.write(all_modules_rst)

################################################################################

