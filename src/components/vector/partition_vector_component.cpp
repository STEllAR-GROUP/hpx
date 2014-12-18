//  Copyright (c) 2014 Anuj R. Sharma
//  Copyright (c) 2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file src/components/vector/partition_vector_component.cpp

/// This file defines the necessary component boilerplate code which is
/// required for proper functioning of components in the context of HPX.

#include <hpx/include/components.hpp>

#include <hpx/components/vector/partition_vector_component.hpp>
#include <hpx/components/vector/vector.hpp>

HPX_DISTRIBUTED_METADATA(hpx::server::vector_config_data,
    hpx_server_vector_config_data);

HPX_REGISTER_COMPONENT_MODULE();

