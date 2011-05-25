# Copyright (c) 2010-2011 Dylan Stark
# Copyright (c)      2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Set source directory for PXGL
set(pxgl_SOURCE_DIR ${hpx_SOURCE_DIR}/examples/pxgl)

# Build PXGL components
add_subdirectory(${pxgl_SOURCE_DIR}/src)

# Build PXGL examples
add_subdirectory(${pxgl_SOURCE_DIR}/examples)

