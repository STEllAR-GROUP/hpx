# Copyright (c) 2016 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(subdirs
   )

if(HPX_WITH_CUDA)
  set(subdirs ${subdirs}
      cuda
     )
endif()

foreach(subdir ${subdirs})
  add_hpx_pseudo_target(examples.compute.${subdir})
  add_subdirectory(${subdir})
  add_hpx_pseudo_dependencies(examples examples.compute.${subdir})
endforeach()

