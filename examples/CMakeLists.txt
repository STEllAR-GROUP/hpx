# Copyright (c) 2007-2009 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(example_SUBDIRS 
    accumulator amr generic naming parcelset quickstart)

foreach(subdir ${example_SUBDIRS})
    add_subdirectory(${subdir})
endforeach()

