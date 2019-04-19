// Copyright (c) 2019 Weile Wei
// Copyright (c) 2019 Maxwell Reeser
// Copyright (c) 2019 Hartmut Kaiser
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "hpx/lcos/distributed_object.hpp"
#include <hpx/hpx.hpp>
#include <hpx/runtime/components/component_factory.hpp>

///////////////////////////////////////////////////////////////////////////////
// Add factory registration functionality.

typedef hpx::components::component<hpx::lcos::meta_object_server>
    meta_object_type;

HPX_REGISTER_COMPONENT(meta_object_type, meta_object);

HPX_REGISTER_ACTION(meta_object_type::registration_action, register_mo_action);
HPX_REGISTER_ACTION(meta_object_type::get_server_list_action,
    get_server_list_mo_action);
