# Copyright (c) 2007-2009 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    # Removes AMR example when building on for Mac OS X
    set(example_SUBDIRS 
        accumulator naming parcelset generic
        quickstart
        #heart_beat
       )
else()
    set(example_SUBDIRS 
        accumulator generic had_amr naming parcelset
        #quickstart
        #heart_beat
       )
endif()

foreach(subdir ${example_SUBDIRS})
    add_subdirectory(${subdir})
endforeach()

