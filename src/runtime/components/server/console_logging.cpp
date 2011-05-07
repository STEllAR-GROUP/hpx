//  Copyright (c) 2007-2011 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <vector>
#include <iostream>

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/ini.hpp>
#include <hpx/runtime/components/plain_component_factory.hpp>
#include <hpx/runtime/components/server/console_logging.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/lexical_cast.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/export.hpp>

#include <boost/logging/format/named_write_fwd.hpp>
#include <boost/logging/format_fwd.hpp>

///////////////////////////////////////////////////////////////////////////////
// definitions related to console logging

///////////////////////////////////////////////////////////////////////////////
// This must be in global namespace
HPX_REGISTER_PLAIN_ACTION_EX(
    hpx::components::server::console_logging_action<>,
    console_logging_action);

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    // implementation of console based logging
    void console_logging(logging_destination dest, int level, 
        std::string const& msg)
    {
        switch (dest) {
        case destination_hpx:
            LHPX_CONSOLE_(level) << msg;
            break;

        case destination_timing:
            LTIM_CONSOLE_(level) << msg;
            break;

        case destination_agas:
            LAGAS_CONSOLE_(level) << msg;
            break;

        case destination_app:
            LAPP_CONSOLE_(level) << msg;
            break;

        default:
            break;
        }
    }
}}}

