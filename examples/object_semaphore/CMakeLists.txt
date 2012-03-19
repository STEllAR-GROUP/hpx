# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(types
    int)

foreach(type ${types})
  source_group("Source Files\\Server"
               FILES ${type}_object_semaphore/${type}_object_semaphore.cpp)
  source_group("Source Files" FILES ${type}_object_semaphore_client.cpp)

  add_hpx_component(${type}_object_semaphore
    SOURCES ${type}_object_semaphore/${type}_object_semaphore.cpp
    FOLDER "Examples/ObjectSemaphore")

  add_hpx_executable(${type}_object_semaphore_client
    SOURCES ${type}_object_semaphore_client.cpp
    DEPENDENCIES ${type}_object_semaphore_component
    FOLDER "Examples/ObjectSemaphore")

  add_hpx_pseudo_target(examples.object_semaphore.${type}_object_semaphore)

  add_hpx_pseudo_dependencies(examples.object_semaphore
    examples.object_semaphore.${type}_object_semaphore)

  add_hpx_pseudo_dependencies(examples.object_semaphore.${type}_object_semaphore
    ${type}_object_semaphore_client_exe)
endforeach()

