//  Copyright (c) 2007-2009 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <vector>
#include <iostream>

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/ini.hpp>
#include <hpx/runtime/components/server/console_logging.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/lexical_cast.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/export.hpp>

#include <boost/logging/format/named_write_fwd.hpp>
#include <boost/logging/format_fwd.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    typedef boost::logging::named_logger<>::type logger_type;
    typedef boost::logging::level::holder filter_type;

    ///////////////////////////////////////////////////////////////////////////
    BOOST_DECLARE_LOG_FILTER(agas_console_level, filter_type)
    BOOST_DECLARE_LOG(agas_console_logger, logger_type)

    ///////////////////////////////////////////////////////////////////////////
    BOOST_DECLARE_LOG_FILTER(timing_console_level, filter_type)
    BOOST_DECLARE_LOG(timing_console_logger, logger_type)

    ///////////////////////////////////////////////////////////////////////////
    BOOST_DECLARE_LOG_FILTER(hpx_console_level, filter_type)
    BOOST_DECLARE_LOG(hpx_console_logger, logger_type)

}}

///////////////////////////////////////////////////////////////////////////////
// definitions related to console logging

#define LAGAS_CONSOLE_(lvl)                                                   \
    BOOST_LOG_USE_LOG(util::agas_console_logger(), read_msg().gather().out(), \
        util::agas_console_level()->is_enabled(lvl))                          \
/**/

#define LTIM_CONSOLE_(lvl)                                                    \
    BOOST_LOG_USE_LOG(util::timing_console_logger(), read_msg().gather().out(),\
        util::timing_console_level()->is_enabled(lvl))                        \
/**/

#define LHPX_CONSOLE_(lvl)                                                    \
    BOOST_LOG_USE_LOG(util::hpx_console_logger(), read_msg().gather().out(),  \
        util::hpx_console_level()->is_enabled(lvl))                           \
/**/

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

        default:
            break;
        }
    }

}}}

// This must be in global namespace
HPX_REGISTER_ACTION(hpx::components::server::console_logging_action);

