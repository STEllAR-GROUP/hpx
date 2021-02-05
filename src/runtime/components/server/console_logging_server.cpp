//  Copyright (c) 2007-2016 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/actions/continuation.hpp>
#include <hpx/actions/transfer_action.hpp>
#include <hpx/actions/transfer_continuation_action.hpp>
#include <hpx/concurrency/spinlock.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/lcos/base_lco_with_value.hpp>
#include <hpx/modules/logging.hpp>
#include <hpx/runtime/components/server/console_logging.hpp>
#include <hpx/runtime_configuration/ini.hpp>
#include <hpx/type_support/static.hpp>

#include <cstddef>
#include <mutex>
#include <string>

///////////////////////////////////////////////////////////////////////////////
// definitions related to console logging

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
// This must be in global namespace
HPX_REGISTER_ACTION_ID(
    hpx::components::server::console_logging_action<>,
    console_logging_action,
    hpx::actions::console_logging_action_id)

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    // implementation of console based logging
    void console_logging(messages_type const& msgs)
    {
        std::lock_guard<util::spinlock> l(util::detail::get_log_lock());

        using hpx::get;

        for (message_type const& msg : msgs)
        {
            const logging_destination dest = get<0>(msg);
            const std::size_t level = get<1>(msg);
            std::string const& s = get<2>(msg);

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
                HPX_UNUSED(level);
                break;
            }
        }
    }
}}}

