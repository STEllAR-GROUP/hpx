//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_HEARTBEAT_WIN_PERF_COUNTERS_2011_JUL_24_0259PM)
#define HPX_HEARTBEAT_WIN_PERF_COUNTERS_2011_JUL_24_0259PM

#include <hpx/hpx.hpp>

///////////////////////////////////////////////////////////////////////////////
// This is code specific to Windows
#if defined(HPX_WINDOWS)

///////////////////////////////////////////////////////////////////////////////
// Install the windows performance counters exposing  the HPX counters.
void install_windows_counters();

// Update the data to be exposed as the windows performance counter values.
void update_windows_counters(boost::uint64_t value);

// Uninstall the windows performance counters exposing  the HPX counters.
void uninstall_windows_counters();

#endif

#endif

