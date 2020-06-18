#!/usr/bin/env python3
'''
Copyright (c) 2018 Thomas Heller

SPDX-License-Identifier: BSL-1.0
Distributed under the Boost Software License, Version 1.0. (See accompanying
file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

create_component_skeleton.py - A tool to generate a component skeleton
'''

import sys, os

if len(sys.argv) != 2:
    print('Usage: %s <component_name>' % sys.argv[0])
    print('Generates the skeleton for component_name in the current working directory')
    sys.exit(1)

component_name = sys.argv[1]
component_name_upper = component_name.upper()
header_str = '=' * len(component_name)

# CMake minimum version
cmake_version = '3.13'

cmake_header = f'''# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
'''

root_cmakelists_template = cmake_header + f'''
if(NOT HPX_WITH_DEFAULT_TARGETS)
  set(_exclude_from_all_flag EXCLUDE_FROM_ALL)
endif()

set({component_name}_headers)

set({component_name}_sources)

add_hpx_component({component_name}
  INTERNAL_FLAGS
  FOLDER "Core/Components"
  INSTALL_HEADERS
  HEADER_ROOT "${{CMAKE_CURRENT_SOURCE_DIR}}/include"
  HEADERS ${{{component_name}_headers}}
  SOURCE_ROOT "${{CMAKE_CURRENT_SOURCE_DIR}}/src"
  SOURCES ${{{component_name}_sources}}
  ${{_exclude_from_all_flag}}
)
'''

examples_cmakelists_template = cmake_header + f'''
if (HPX_WITH_TESTS AND HPX_WITH_TESTS_EXAMPLES)
  add_hpx_pseudo_target(tests.examples.components.{component_name})
  add_hpx_pseudo_dependencies(tests.examples.components tests.examples.components.{component_name})
endif()

'''

tests_cmakelists_template = cmake_header + f'''
include(HPX_Option)

if (HPX_WITH_TESTS_UNIT)
  add_hpx_pseudo_target(tests.unit.components.{component_name})
  add_hpx_pseudo_dependencies(tests.unit.components tests.unit.components.{component_name})
  add_subdirectory(unit)
endif()

if (HPX_WITH_TESTS_REGRESSIONS)
  add_hpx_pseudo_target(tests.regressions.components.{component_name})
  add_hpx_pseudo_dependencies(tests.regressions.components tests.regressions.components.{component_name})
  add_subdirectory(regressions)
endif()

if (HPX_WITH_TESTS_BENCHMARKS)
  add_hpx_pseudo_target(tests.performance.components.{component_name})
  add_hpx_pseudo_dependencies(tests.performance.components tests.performance.components.{component_name})
  add_subdirectory(performance)
endif()

if (HPX_WITH_TESTS_HEADERS)
  add_hpx_header_tests("components.{component_name}"
    HEADERS ${{{component_name}_headers}}
    HEADER_ROOT "${{CMAKE_CURRENT_SOURCE_DIR}}/include"
    COMPONENT_DEPENDENCIES {component_name})
endif()
'''

if component_name != '--recreate-index':
    def mkdir(path):
        if not os.path.exists(path):
            os.makedirs(path)

    mkdir(component_name)

    ################################################################################
    # Generate basic directory structure
    for subdir in ['examples', 'include', 'src', 'tests']:
        path = os.path.join(component_name, subdir)
        mkdir(path)
    # Generate include directory structure
    # Normalize path...
    include_path = ''.join(component_name)
    path = os.path.join(component_name, 'include', 'hpx', include_path)
    mkdir(path)
    path = os.path.join(component_name, 'tests', 'unit')
    mkdir(path)
    path = os.path.join(component_name, 'tests', 'regressions')
    mkdir(path)
    path = os.path.join(component_name, 'tests', 'performance')
    mkdir(path)
    ################################################################################

    ################################################################################
    # Generate CMakeLists.txt skeletons

    # Generate top level CMakeLists.txt
    f = open(os.path.join(component_name, 'CMakeLists.txt'), 'w')
    f.write(root_cmakelists_template)

    # Generate examples/CMakeLists.txt
    f = open(os.path.join(component_name, 'examples', 'CMakeLists.txt'), 'w')
    f.write(examples_cmakelists_template)

    # Generate tests/CMakeLists.txt
    f = open(os.path.join(component_name, 'tests', 'CMakeLists.txt'), 'w')
    f.write(tests_cmakelists_template)

    # Generate tests/unit/CMakeLists.txt
    f = open(os.path.join(component_name, 'tests', 'unit', 'CMakeLists.txt'), 'w')
    f.write(cmake_header)

    # Generate tests/regressions/CMakeLists.txt
    f = open(os.path.join(component_name, 'tests', 'regressions', 'CMakeLists.txt'), 'w')
    f.write(cmake_header)
    f.write('\n')

    # Generate tests/performance/CMakeLists.txt
    f = open(os.path.join(component_name, 'tests', 'performance', 'CMakeLists.txt'), 'w')
    f.write(cmake_header)
    ################################################################################

################################################################################

# Scan directory to get all components...
cwd = os.getcwd()
components = sorted([ component for component in os.listdir(cwd) if os.path.isdir(component) ])

# Adapting top level CMakeLists.txt
components_cmakelists = cmake_header + f'''
# This file is auto generated. Please do not edit manually.
'''

components_cmakelists += '''
include(HPX_Message)
include(HPX_AddPseudoDependencies)
include(HPX_AddPseudoTarget)

set(HPX_COMPONENTS
'''
for component in components:
    components_cmakelists += f'  {component}\n'
components_cmakelists += '  CACHE INTERNAL "list of HPX components" FORCE\n)\n\n'

components_cmakelists += '''
# add example pseudo targets needed for components
if(HPX_WITH_EXAMPLES)
  add_hpx_pseudo_target(examples.components)
  add_hpx_pseudo_dependencies(examples examples.components)
endif()

# add test pseudo targets needed for components
if(HPX_WITH_TESTS)
  if (HPX_WITH_TESTS_UNIT)
    add_hpx_pseudo_target(tests.unit.components)
    add_hpx_pseudo_dependencies(tests.unit tests.unit.components)
  endif()

  if (HPX_WITH_EXAMPLES AND HPX_WITH_TESTS_EXAMPLES)
    add_hpx_pseudo_target(tests.examples.components)
    add_hpx_pseudo_dependencies(tests.examples tests.examples.components)
  endif()

  if (HPX_WITH_TESTS_REGRESSIONS)
    add_hpx_pseudo_target(tests.regressions.components)
    add_hpx_pseudo_dependencies(tests.regressions tests.regressions.components)
  endif()

  if (HPX_WITH_TESTS_BENCHMARKS)
    add_hpx_pseudo_target(tests.performance.components)
    add_hpx_pseudo_dependencies(tests.performance tests.performance.components)
  endif()

  if (HPX_WITH_TESTS_HEADERS)
    add_hpx_pseudo_target(tests.headers.components)
    add_hpx_pseudo_dependencies(tests.headers tests.headers.components)
  endif()
endif()
'''

components_cmakelists += '''
hpx_info("Configuring components:")

add_hpx_pseudo_target(components)

foreach(component ${HPX_COMPONENTS})
  hpx_info("  ${component}")
  add_hpx_pseudo_target(components.${component})
  add_subdirectory(${component})
  add_hpx_pseudo_dependencies(components components.${component})
endforeach()

add_hpx_pseudo_dependencies(core components)
'''

f = open(os.path.join(cwd, 'CMakeLists.txt'), 'w')
f.write(components_cmakelists)

################################################################################
