//  Copyright (c) 2007-2011 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_CONSOLE_ERROR_SINK_JAN_23_2009_0621PM)
#define HPX_COMPONENTS_CONSOLE_ERROR_SINK_JAN_23_2009_0621PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/eager_future.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/applier/apply.hpp>
#include <hpx/runtime/components/server/console_error_sink.hpp>
#include <hpx/runtime/components/console_logging.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components
{
    // stub function allowing to apply the console_console_error_sink action
    void console_error_sink(naming::id_type const& dst, 
        naming::gid_type const& src, boost::exception_ptr const& e)
    {
        // do logging only if the system is still up. 
        if (system_is_running())
        {
            lcos::eager_future<server::console_error_sink_action> f
                (dst, naming::get_prefix_from_gid(src), e);
            f.get();
        }
    }
}}

#endif
