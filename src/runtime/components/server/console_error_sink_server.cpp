//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/util/ini.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/runtime/actions/basic_action.hpp>
// This is needed to get rid of an undefined reference to
// hpx::actions::detail::register_remote_action_invocation_count
#include <hpx/runtime/actions/transfer_action.hpp>
#include <hpx/runtime/components/server/console_error_sink.hpp>
#include <hpx/runtime/components/server/console_error_sink_singleton.hpp>
#include <hpx/util/serialize_exception.hpp>

#include <boost/lexical_cast.hpp>

#include <vector>
#include <iostream>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    // implementation of this console error sink
    void console_error_sink(boost::exception_ptr const& e)
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

