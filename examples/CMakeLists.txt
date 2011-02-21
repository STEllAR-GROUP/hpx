# Copyright (c) 2007-2009 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# NOTE: please do not comment out targets here and commit. If you do not want
# to build all the examples, you can explicitly specify which examples you
# would like to build on the command line, e.g. `make examples.had_amr3d'.
# It is important that all of these examples remain linked into the CMake
# dependency tree for testing purposes.

set(subdirs 
    accumulator
    amr
    bhpx
    cosmo
    generic
    had_amr
    had_amr3d
    naming
    parcelset
    quickstart
    )

foreach(subdir ${subdirs})
  add_custom_target(examples.${subdir})
  add_subdirectory(${subdir})
  add_dependencies(examples examples.${subdir})
endforeach()

add_custom_target(chirag)
add_custom_target(matt)
add_custom_target(matt.all)

add_dependencies(chirag examples.bhpx)

add_dependencies(matt examples.had_amr3d)
add_dependencies(matt.all examples.cosmo)
add_dependencies(matt.all examples.amr)
add_dependencies(matt.all examples.had_amr)
add_dependencies(matt.all examples.had_amr3d)

