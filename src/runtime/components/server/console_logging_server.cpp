//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>

#include <hpx/util/ini.hpp>
#include <hpx/util/static.hpp>
#include <hpx/util/spinlock.hpp>
#include <hpx/util/tuple.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/runtime/components/server/console_logging.hpp>
#include <hpx/runtime/serialization/serialize_sequence.hpp>

#include <boost/fusion/include/at_c.hpp>
#include <boost/lexical_cast.hpp>

#include <hpx/util/logging/format/named_write_fwd.hpp>
#include <hpx/util/logging/format_fwd.hpp>

#include <vector>
#include <iostream>

///////////////////////////////////////////////////////////////////////////////
// definitions related to console logging

///////////////////////////////////////////////////////////////////////////////
// This must be in global namespace
HPX_REGISTER_ACTION_ID(
    hpx::components::server::console_logging_action<>,
    console_logging_action,
    hpx::actions::console_logging_action_id)

namespace hpx { namespace util { namespace detail
{
    struct log_lock_tag {};

    hpx::util::spinlock& get_log_lock()
    {
        hpx::util::static_<hpx::util::spinlock, log_lock_tag> lock;
        return lock.get();
    }
}}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    // implementation of console based logging
    void console_logging(messages_type const& msgs)
    {
        util::spinlock::scoped_lock l(util::detail::get_log_lock());

        using boost::fusion::at_c;

        for (message_type const& msg : msgs)
        {
            const logging_destination dest = at_c<0>(msg);
            const std::size_t level = at_c<1>(msg);
            std::string const& s = at_c<2>(msg);

            switch (dest) {
            case destination_hpx:
                LHPX_CONSOLE_(level) << s;
                break;

            case destination_timing:
                LTIM_CONSOLE_(level) << s;
                break;

            case destination_agas:
                LAGAS_CONSOLE_(level) << s;
                break;

            case destination_parcel:
                LPT_CONSOLE_(level) << s;
                break;

            case destination_app:
                LAPP_CONSOLE_(level) << s;
                break;

            case destination_debuglog:
                LDEB_CONSOLE_ << s;
                break;

            default:
                break;
            }
        }
    }
}}}

