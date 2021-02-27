//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//  Copyright (c) 2012-2020 Hartmut Kaiser
//  Copyright (c) 2016 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/actions/transfer_action.hpp>
#include <hpx/actions_base/plain_action.hpp>
#include <hpx/async_distributed/transfer_continuation_action.hpp>

#include <string>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace agas {

    // Create statistics counter for symbol namespace on this locality
    HPX_EXPORT naming::gid_type symbol_namespace_statistics_counter(
        std::string const& name);

    HPX_DEFINE_PLAIN_ACTION(symbol_namespace_statistics_counter,
        symbol_namespace_statistics_counter_action);
}}    // namespace hpx::agas

HPX_ACTION_USES_MEDIUM_STACK(
    hpx::agas::symbol_namespace_statistics_counter_action)

HPX_REGISTER_ACTION_DECLARATION(
    hpx::agas::symbol_namespace_statistics_counter_action,
    symbol_namespace_statistics_counter_action)
