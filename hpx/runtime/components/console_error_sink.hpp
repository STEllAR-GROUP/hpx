//  Copyright (c) 2007-2010 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_CONSOLE_ERROR_SINK_JAN_23_2009_0621PM)
#define HPX_COMPONENTS_CONSOLE_ERROR_SINK_JAN_23_2009_0621PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/applier/apply.hpp>
#include <hpx/runtime/components/server/console_error_sink.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components
{
    // stub function allowing to apply the console_console_error_sink action
    void console_error_sink(naming::id_type const& dst, 
        naming::id_type const& src, boost::exception_ptr const& e)
    {
        // do logging only if applier is still valid
        if (NULL != applier::get_applier_ptr())
        {
            applier::apply<server::console_error_sink_action>(
                dst, naming::get_prefix_from_id(src), e);
        }
    }

}}

#endif
