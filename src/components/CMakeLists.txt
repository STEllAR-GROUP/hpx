# Copyright (c) 2007-2010 Hartmut Kaiser
# Copyright (c)      2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(subdirs
    distributing_factory
    iostreams
    remote_object
)

foreach(subdir ${subdirs})
  add_subdirectory(${subdir})
endforeach()

