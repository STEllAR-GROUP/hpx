# Copyright (c) 2010-2011 Dylan Stark
# Copyright (c) 2011      Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# add pseudo-targets
add_custom_target(examples.cps.factorial)
add_custom_target(examples.cps.factorial_get)

# define build targets for this directory
add_hpx_executable(factorial SOURCES factorial.cpp)
add_hpx_executable(factorial_get SOURCES factorial_get.cpp)

# add pseudo-target dependencies
add_dependencies(examples.cps.factorial factorial)
add_dependencies(examples.cps.factorial_get factorial_get)

