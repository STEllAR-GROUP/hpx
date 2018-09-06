# Copyright (c) 2015 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

add_hpx_executable(queue_client
  SOURCES queue_client.cpp
  FOLDER "Examples/Quickstart/Queue")

add_hpx_pseudo_target(examples.queue.queue_client)
add_hpx_pseudo_dependencies(examples.queue examples.queue.queue_client)
add_hpx_pseudo_dependencies(examples.queue.queue_client queue_client_exe)

if(HPX_WITH_TESTS AND HPX_WITH_TESTS_EXAMPLES)
  add_hpx_example_test("queue" queue_client THREADS_PER_LOCALITY 4)
  add_hpx_pseudo_target(tests.examples.queue.queue_client)
  add_hpx_pseudo_dependencies(tests.examples.queue
                              tests.examples.queue.queue_client)
  add_hpx_pseudo_dependencies(tests.examples.queue.queue_client
                              queue_client_exe)
endif()

