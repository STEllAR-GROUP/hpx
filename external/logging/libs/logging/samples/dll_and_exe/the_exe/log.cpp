/**
 Boost Logging library

 Author: John Torjo, www.torjo.com

 Copyright (C) 2007 John Torjo (see www.torjo.com for email)

 Distributed under the Boost Software License, Version 1.0.
    (See accompanying file LICENSE_1_0.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)

 See http://www.boost.org for updates, documentation, and revision history.
 See http://www.torjo.com/log2/ for more details
*/

#include "log.h"

BOOST_DEFINE_LOG_FILTER(g_log_filter, boost::logging::filter::no_ts) 
BOOST_DEFINE_LOG(g_l, log_type )

using namespace boost::logging;

void init_logs() {
    // first, write to a clean file (that is, don't append to it)
    g_l()->writer().add_destination( destination::file("exe.txt", destination::file_settings().initial_overwrite(true) ));
    g_l()->writer().add_destination( destination::file("dllexe.txt", destination::file_settings().do_append(true) ));
    g_l()->writer().add_formatter( formatter::idx(), "[%] "  );
    g_l()->writer().add_formatter( formatter::append_newline() );
    g_l()->writer().add_destination( destination::cout() );
    g_l()->turn_cache_off();
}

