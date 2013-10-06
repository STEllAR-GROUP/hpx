# Copyright (c) 2007-2012 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(root "${hpx_SOURCE_DIR}/hpx/components/dataflow")

if(HPX_DEFAULT_BUILD_TARGETS)
  add_hpx_component(dataflow
    FOLDER "Core/Components"
    HEADER_ROOT ${root}
    AUTOGLOB
    ESSENTIAL)
else() 
  add_hpx_component(dataflow
    FOLDER "Core/Components"
    HEADER_ROOT ${root}
    AUTOGLOB)
endif()

add_hpx_pseudo_dependencies(components.dataflow
                            dataflow_component)

################################################################################
if(${BOOST_MINOR_VERSION} GREATER 50) 
  # add headers to preprocessing
  hpx_partial_preprocess_header(
      "hpx/components/dataflow/dataflow.hpp"
      LIMIT HPX_ACTION_ARGUMENT_LIMIT
  )
  hpx_partial_preprocess_header(
      "hpx/components/dataflow/server/detail/dataflow_impl.hpp"
      LIMIT HPX_ACTION_ARGUMENT_LIMIT
  )
  hpx_partial_preprocess_header(
      "hpx/components/dataflow/server/detail/apply_helper.hpp"
      LIMIT HPX_ACTION_ARGUMENT_LIMIT
  )
  hpx_partial_preprocess_header(
      "hpx/components/dataflow/server/dataflow.hpp"
      LIMIT HPX_ACTION_ARGUMENT_LIMIT
  )
endif()
