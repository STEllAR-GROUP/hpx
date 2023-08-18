//  Copyright (c) 2007-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/include/components.hpp>
#include <hpx/include/serialization.hpp>

#include "server/cancelable_action.hpp"

///////////////////////////////////////////////////////////////////////////////
// Add factory registration functionality.
HPX_REGISTER_COMPONENT_MODULE()

///////////////////////////////////////////////////////////////////////////////
using cancelable_action_component_type =
    hpx::components::component<examples::server::cancelable_action>;

HPX_REGISTER_COMPONENT(cancelable_action_component_type, cancelable_action)

///////////////////////////////////////////////////////////////////////////////
// Serialization support for simple_accumulator actions.
HPX_REGISTER_ACTION(examples::server::cancelable_action::do_it_action,
    cancelable_action_do_it_action)
HPX_REGISTER_ACTION(examples::server::cancelable_action::cancel_it_action,
    cancelable_action_cancel_it_action)

#endif
