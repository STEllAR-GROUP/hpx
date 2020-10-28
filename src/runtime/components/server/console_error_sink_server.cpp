//  Copyright (c) 2007-2016 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/actions_base/basic_action.hpp>
#include <hpx/actions/continuation.hpp>
#include <hpx/actions/transfer_action.hpp>
#include <hpx/actions/transfer_continuation_action.hpp>
#include <hpx/actions_base/basic_action.hpp>
#include <hpx/lcos/base_lco_with_value.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/runtime/components/server/console_error_sink.hpp>
#include <hpx/runtime/components/server/console_error_sink_singleton.hpp>
#include <hpx/runtime_configuration/ini.hpp>
#include <hpx/runtime_local/custom_exception_info.hpp>

#include <exception>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    // implementation of this console error sink
    void console_error_sink(std::exception_ptr const& e)
    {
        // dispatch this error to registered functions
        get_error_dispatcher()(hpx::diagnostic_information(e));
    }
}}}

///////////////////////////////////////////////////////////////////////////////
// This must be in global namespace
HPX_REGISTER_ACTION_ID(
    hpx::components::server::console_error_sink_action,
    console_error_sink_action,
    hpx::actions::console_error_sink_action_id)

