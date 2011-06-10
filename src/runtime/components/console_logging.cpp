////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if HPX_AGAS_VERSION > 0x10

#include <hpx/runtime.hpp>
#include <hpx/runtime/components/console_logging.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components
{

void console_logging(
    server::logging_destination dest
  , int level
  , std::string const& msg
) {
    util::static_<pending_logs, pending_logs_tag> logs;

    // do logging only if applier is valid
    if (system_is_running())
    {
        if (logs.get().sending_logs_)
            logs.get().add_pending(msg);

        else
        {
            naming::gid_type raw_prefix;
            get_runtime().get_agas_client().get_console_prefix(raw_prefix);
            naming::id_type prefix(raw_prefix, naming::id_type::unmanaged);

            reset_on_exit exit(logs.get().sending_logs_, true);
            logs.get().handle_pending(prefix, dest, level);
            console_logging_locked(prefix, dest, level, msg);
        }
    }

    else
        logs.get().add_pending(msg);
}

}}

#endif

