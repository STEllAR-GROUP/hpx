//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/serialization.hpp>

#include "server/simple_mobile_object.hpp"

HPX_REGISTER_COMPONENT_MODULE()

using hpx::test::server::simple_mobile_object;

typedef hpx::components::component<simple_mobile_object> mobile_component_type;

///////////////////////////////////////////////////////////////////////////////
// We use a special component registry for this component as it has to be
// disabled by default. All tests requiring this component to be active will
// enable it explicitly.
HPX_REGISTER_DISABLED_COMPONENT_FACTORY(
    hpx::components::component<simple_mobile_object>, simple_mobile_object)

///////////////////////////////////////////////////////////////////////////////
HPX_REGISTER_ACTION(
    simple_mobile_object::get_lva_action, simple_mobile_object_get_lva_action)

#endif
