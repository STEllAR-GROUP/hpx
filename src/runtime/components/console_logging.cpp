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

struct console_prefix_tag {};

naming::id_type get_console_prefix()
{
    naming::resolver_client& agas_client = get_runtime().get_agas_client();

    naming::gid_type console_prefix;
    for (int i = 0; i < HPX_MAX_NETWORK_RETRIES; ++i)
    {
        if (agas_client.get_console_prefix(console_prefix))
            break;

        boost::this_thread::sleep(boost::get_system_time() + 
            boost::posix_time::milliseconds(HPX_NETWORK_RETRIES_SLEEP));
    }

    if (!console_prefix) 
    {
        HPX_THROW_EXCEPTION(no_registered_console, 
            "get_console_prefix", "couldn't retrieve console locality");
    }

    return naming::id_type(console_prefix, naming::id_type::unmanaged);
}

struct console_prefix_thunk
{
    naming::id_type gid;

    console_prefix_thunk() : gid(get_console_prefix()) {}
};

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
            util::static_<console_prefix_thunk, console_prefix_tag> prefix;
            reset_on_exit exit(logs.get().sending_logs_, true);
            logs.get().handle_pending(prefix.get().gid, dest, level);
            console_logging_locked(prefix.get().gid, dest, level, msg);
        }
    }

    else
        logs.get().add_pending(msg);
}

}}

#endif

