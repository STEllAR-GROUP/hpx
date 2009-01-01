//  Copyright (c) 2007-2009 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_CONSOLE_LOGGING_DEC_16_2008_0435PM)
#define HPX_COMPONENTS_CONSOLE_LOGGING_DEC_16_2008_0435PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/applier/apply.hpp>
#include <hpx/runtime/components/server/console_logging.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components
{
    // stub function allowing to apply the console_logging action
    void console_logging(naming::id_type const& prefix, 
        server::logging_destination dest, int level, std::string const& msg)
    {
        // do logging only if applier is still valid
        if (NULL != applier::get_applier_ptr())
            applier::apply<server::console_logging_action>(prefix, dest, level, msg);
    }

    // special initialization functions for console logging 
    namespace detail
    {
        void init_agas_console_log(util::section const& ini);
        void init_timing_console_log(util::section const& ini);
        void init_hpx_console_log(util::section const& ini);
    }

}}

#endif
