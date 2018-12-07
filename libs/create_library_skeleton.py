#!/usr/bin/env python
'''
Copyright (c) 2018 Thomas Heller

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

if os.path.exists(lib_name):
    print('%s already exists, please delete it or choose another name' % lib_name)
    sys.exit(1)

# CMake minimum version
cmake_version = '3.3.2'

os.makedirs(lib_name)

################################################################################
# Generate basic directory structure
for subdir in ['cmake', 'docs', 'examples', 'include', 'src', 'tests']:
    path = os.path.join(lib_name, subdir)
    os.makedirs(path)
# Generate include directory structure
# Normalize path...
include_path = ''.join(lib_name)
path = os.path.join(lib_name, 'include', 'hpx', include_path)
os.makedirs(path)
path = os.path.join(lib_name, 'tests', 'unit')
os.makedirs(path)
path = os.path.join(lib_name, 'tests', 'regressions')
os.makedirs(path)
path = os.path.join(lib_name, 'tests', 'performance')
os.makedirs(path)
################################################################################

################################################################################
# Generate Readme skeleton
f = open(os.path.join(lib_name, 'Readme.md'), 'w')
f.write('<!-- Copyright (c) 2018 The STE||AR-Group                                         -->\n')
f.write('<!--                                                                              -->\n')
f.write('<!-- Distributed under the Boost Software License, Version 1.0. (See accompanying -->\n')
f.write('<!-- file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)        -->\n')
f.write('\n')
f.write('# %s\n' % lib_name)
f.write('\n')
f.write('This library is part of HPX.\n')
f.write('\n')
f.write('Extensive documentation can be found at\n')
f.write('https://stellar-group.github.io/hpx/docs/sphinx/latest/html/libs/%s/docs/index.html\n' % lib_name)
################################################################################

################################################################################
# Generate CMakeLists.txt skeletons
cmake_header =  '# Copyright (c) 2018 The STE||AR-Group\n'
cmake_header += '#\n'
cmake_header += '# Distributed under the Boost Software License, Version 1.0. (See accompanying\n'
cmake_header += '# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)\n'

# Generate top level CMakeLists.txt
f = open(os.path.join(lib_name, 'CMakeLists.txt'), 'w')
f.write(cmake_header)
f.write('# We require at least CMake V%s\n' % cmake_version)
f.write('cmake_minimum_required(VERSION %s FATAL_ERROR)\n' % cmake_version)
f.write('\n')
f.write('project(HPX.%s CXX)\n' % lib_name)
f.write('\n')
f.write('list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")\n')
f.write('\n')
f.write('option(HPX_%s_WITH_TESTS "Include tests for %s" On)\n' % (lib_name.upper(), lib_name))
f.write('\n')
f.write('message(STATUS "%s: Configuring")\n' % lib_name)
f.write('\n')
f.write('add_subdirectory(examples)\n')
f.write('add_subdirectory(src)\n')
f.write('add_subdirectory(tests)\n')
f.write('\n')
f.write('message(STATUS "%s: Configuring done")\n' % lib_name)

# Generate docs/index.rst
f = open(os.path.join(lib_name, 'docs', 'index.rst'), 'w')
f.write('..\n')
f.write('    Copyright (c) 2018 The STE||AR-Group\n')
f.write('\n')
f.write('    Distributed under the Boost Software License, Version 1.0. (See accompanying\n')
f.write('    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)\n')
f.write('\n')
f.write('.. _libs_%s:\n' % lib_name)
f.write('\n')
f.write('===========\n')
f.write('%s\n' % lib_name)
f.write('===========\n')
f.write('\n')

# Generate examples/CMakeLists.txt
f = open(os.path.join(lib_name, 'examples', 'CMakeLists.txt'), 'w')
f.write(cmake_header)
f.write('\n')

# Generate src/CMakeLists.txt
f = open(os.path.join(lib_name, 'src', 'CMakeLists.txt'), 'w')
f.write(cmake_header)
f.write('\n')

# Generate tests/CMakeLists.txt
f = open(os.path.join(lib_name, 'tests', 'CMakeLists.txt'), 'w')
f.write(cmake_header)
f.write('\n')
f.write('if (NOT HPX_WITH_TESTS AND HPX_TOP_LEVEL)\n')
f.write('  return()\n')
f.write('endif()\n')
f.write('if (NOT HPX_%s_WITH_TESTS)\n' % lib_name.upper().replace('.', '_'))
f.write('  message(STATUS "Tests for %s disabled")\n' % lib_name)
f.write('  return()\n')
f.write('endif()\n')
f.write('\n')
f.write('add_subdirectory(unit)\n')
f.write('add_subdirectory(regressions)\n')
f.write('add_subdirectory(performance)\n')

# Generate tests/unit/CMakeLists.txt
f = open(os.path.join(lib_name, 'tests', 'unit', 'CMakeLists.txt'), 'w')
f.write(cmake_header)
f.write('\n')

# Generate tests/regressions/CMakeLists.txt
f = open(os.path.join(lib_name, 'tests', 'regressions', 'CMakeLists.txt'), 'w')
f.write(cmake_header)
f.write('\n')

# Generate tests/performance/CMakeLists.txt
f = open(os.path.join(lib_name, 'tests', 'performance', 'CMakeLists.txt'), 'w')
f.write(cmake_header)
f.write('\n')
################################################################################

################################################################################

# Scan directory to get all libraries...
cwd = os.getcwd()
libs = sorted([ lib for lib in os.listdir(cwd) if os.path.isdir(lib) ])
# Adapting top level CMakeLists.txt
f = open(os.path.join(cwd, 'CMakeLists.txt'), 'w')
f.write(cmake_header)
f.write('\n')
for lib in libs:
    # Ignore subdirectories starting with _
    if not lib.startswith('_'):
        f.write('add_subdirectory(%s)\n' % lib)

# Adapting top level index.rst
f = open(os.path.join(cwd, 'index.rst'), 'w')
f.write('..\n')
f.write('    Copyright (c) 2018 The STE||AR-Group\n')
f.write('\n')
f.write('    Distributed under the Boost Software License, Version 1.0. (See accompanying\n')
f.write('    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)\n')
f.write('\n')
f.write('.. toctree::\n')
f.write('   :caption: Libraries\n')
f.write('   :maxdepth: 2\n')
f.write('\n')
for lib in libs:
    f.write('   /libs/%s/docs/index.rst\n' % lib)


################################################################################

