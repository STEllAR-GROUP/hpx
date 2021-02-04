//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//  Copyright (c) 2012-2021 Hartmut Kaiser
//  Copyright (c) 2016 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/actions/transfer_action.hpp>
#include <hpx/actions/transfer_continuation_action.hpp>
#include <hpx/actions_base/plain_action.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/naming.hpp>

#include <string>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace agas {

    // Register all performance counter types exposed by the component_namespace
    HPX_EXPORT void component_namespace_register_counter_types(
        error_code& ec = throws);

    // Create statistics counter for component namespace on this component
    HPX_EXPORT naming::gid_type component_namespace_statistics_counter(
        std::string const& name);

    HPX_DEFINE_PLAIN_ACTION(component_namespace_statistics_counter,
        component_namespace_statistics_counter_action);
}}    // namespace hpx::agas

HPX_ACTION_USES_MEDIUM_STACK(
    hpx::agas::component_namespace_statistics_counter_action)

HPX_REGISTER_ACTION_DECLARATION(
    hpx::agas::component_namespace_statistics_counter_action,
    component_namespace_statistics_counter_action)
