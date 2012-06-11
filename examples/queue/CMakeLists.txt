# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(types
    ints)

foreach(type ${types})
  source_group("Source Files\\Server" FILES queue_of_${type}/queue_of_${type}.cpp)
  source_group("Source Files" FILES queue_of_${type}_client.cpp)

  add_hpx_component(queue_of_${type}
                    SOURCES queue_of_${type}/queue_of_${type}.cpp
                    FOLDER "Examples/Queue")

  add_hpx_executable(queue_of_${type}_client
                     SOURCES queue_of_${type}_client.cpp
                     DEPENDENCIES queue_of_${type}_component
                     FOLDER "Examples/Queue")

  add_hpx_pseudo_target(examples.queue.queue_of_${type})

  add_hpx_pseudo_dependencies(examples.queue examples.queue.queue_of_${type})

  add_hpx_pseudo_dependencies(examples.queue.queue_of_${type}
                              queue_of_${type}_client_exe)
endforeach()

