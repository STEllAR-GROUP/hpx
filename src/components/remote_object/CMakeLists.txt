# Copyright (c) 2007-2012 Hartmut Kaiser
# Copyright (c)      2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

###############################################################################
set(root "${hpx_SOURCE_DIR}/hpx/components/remote_object")

if(HPX_DEFAULT_BUILD_TARGETS)
  add_hpx_component(remote_object
    FOLDER "Core/Components"
    HEADER_ROOT ${root}
    AUTOGLOB
    ESSENTIAL)
else()
  add_hpx_component(remote_object
    FOLDER "Core/Components"
    HEADER_ROOT ${root}
    AUTOGLOB)
endif()

add_hpx_pseudo_dependencies(components.remote_object
                            remote_object_component)

################################################################################
# add headers to partial_preprocessing
if(${BOOST_MINOR_VERSION} GREATER 50) 
  hpx_partial_preprocess_header(
      "hpx/components/remote_object/distributed_new.hpp"
      LIMIT HPX_LIMIT
  )
  hpx_partial_preprocess_header(
      "hpx/components/remote_object/ctor_fun.hpp"
      LIMIT HPX_LIMIT
  )
  hpx_partial_preprocess_header(
      "hpx/components/remote_object/new.hpp"
      LIMIT HPX_LIMIT
  )
endif()

