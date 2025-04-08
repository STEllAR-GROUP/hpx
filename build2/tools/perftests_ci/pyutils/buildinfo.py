# -*- coding: utf-8 -*-
'''
Copyright (c) 2020 ETH Zurich

SPDX-License-Identifier: BSL-1.0
Distributed under the Boost Software License, Version 1.0. (See accompanying
file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
'''

from pyutils import log


# CMake build type
build_type = ''
log.debug('Build type', build_type)

# CMake source dir
source_dir = '/Users/harith/Desktop/Open Source/hpx'
log.debug('Source dir', source_dir)

# CMake binary dir
binary_dir = '/Users/harith/Desktop/Open Source/hpx/build2'
log.debug('Binary dir', binary_dir)

# CMake install dir
install_dir = '/usr/local'
log.debug('Install dir', install_dir)

# Compiler
compiler = '/Library/Developer/CommandLineTools/usr/bin/c++ 16.0.0.16000026'
log.debug('Compiler', compiler)

# Target
envfile = '' or None
log.debug('Environment file', envfile)
