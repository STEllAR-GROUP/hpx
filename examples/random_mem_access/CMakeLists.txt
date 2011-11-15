# Copyright (c) 2011 Bryce Adelstein-Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

add_subdirectory(element)

set(sources ${subdir}_client.cpp)

source_group("Source Files" FILES random_mem_access_client.cpp)

add_hpx_executable(random_mem_access_client
  MODULE random_mem_access
  SOURCES random_mem_access_client.cpp
  DEPENDENCIES random_mem_access_element_component
               distributing_factory_component
               iostreams_component)

add_hpx_pseudo_dependencies(examples.random_mem_access
                            random_mem_access_client_exe)

