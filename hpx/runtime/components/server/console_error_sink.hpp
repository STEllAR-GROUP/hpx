//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/actions/transfer_action.hpp>
#include <hpx/actions/transfer_continuation_action.hpp>
#include <hpx/actions_base/plain_action.hpp>
#include <hpx/components_base/component_type.hpp>

#include <exception>
#include <string>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    // console logging happens here
    void console_error_sink(std::exception_ptr const&);

    HPX_DEFINE_PLAIN_ACTION(console_error_sink, console_error_sink_action);
}}}

HPX_ACTION_HAS_CRITICAL_PRIORITY(
    hpx::components::server::console_error_sink_action);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::components::server::console_error_sink_action,
    console_error_sink_action)

