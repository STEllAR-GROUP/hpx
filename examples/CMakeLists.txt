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

set(hpx_EXAMPLE_SUBDIRS 
    accumulator
    amr
    #atomic
    basic_example
    bhpx
    cosmo
    cps
    generic
    had_amr
    had_amr3d
    heart_beat
    naming
    parcelset
    quickstart
    )

foreach(hpx_EXAMPLE_SUBDIR ${hpx_EXAMPLE_SUBDIRS})
  add_custom_target(examples.${hpx_EXAMPLE_SUBDIR})
  add_subdirectory(${hpx_EXAMPLE_SUBDIR})
  add_dependencies(examples examples.${hpx_EXAMPLE_SUBDIR})
endforeach()

