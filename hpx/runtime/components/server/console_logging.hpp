//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_CONSOLE_DEC_16_2008_0427PM)
#define HPX_COMPONENTS_CONSOLE_DEC_16_2008_0427PM

#include <string>

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/actions/plain_action.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server 
{
    // we know about three different console log destinations
    enum logging_destination
    {
        destination_hpx = 0,
        destination_timing = 1,
        destination_agas = 2,
    };

    ///////////////////////////////////////////////////////////////////////////
    // console logging happens here
    void console_logging(logging_destination dest, int level, std::string const& msg);

    typedef actions::plain_direct_action3<
        logging_destination, int, std::string const&, console_logging
    > console_logging_action;

}}}

#endif
