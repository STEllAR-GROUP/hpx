//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>

///////////////////////////////////////////////////////////////////////////////
// This is code specific to Windows
#if defined(HPX_WINDOWS)

#include <windows.h>
#include <perflib.h>
#include <winperf.h>

#include "win_perf_counters.hpp"
#include "hpx_counters.hpp"

#include <iostream>
#include <boost/lexical_cast.hpp>

///////////////////////////////////////////////////////////////////////////////
PPERF_COUNTERSET_INSTANCE queue_counter = 0;
PPERF_COUNTERSET_INSTANCE avg_queue_counter = 0;

///////////////////////////////////////////////////////////////////////////////
// Install the windows performance counters exposing  the HPX counters.
void install_windows_counters()
{
    // CounterInitialize() is created by ctrpp.exe and is present in ucsCounter.h
    // CounterInitialize starts the provider and initializes the counter sets.
    ULONG status = CounterInitialize(NULL, NULL, NULL, NULL);
    if (status != ERROR_SUCCESS) {
        std::cerr << "CounterInitialize failed with error code: "
                  << boost::lexical_cast<std::string>(status);
        return;
    }

    // Create the instances for multiple instance counter set.
    queue_counter = PerfCreateInstance(HPXHeartBeat, &QueueLengthGuid, L"Instance_1", 0);
    if (queue_counter == NULL) {
        std::cerr << "PerfCreateInstance for 'sum_queue_counter' failed "
                     "with error code: "
                  << boost::lexical_cast<std::string>(GetLastError());
        return;
    }

    avg_queue_counter = PerfCreateInstance
        (HPXHeartBeat, &QueueLengthGuid, L"Instance_2", 0);
    if (avg_queue_counter == NULL) {
        std::cerr << "PerfCreateInstance for 'avg_queue_counter' failed"
                     "with error code: "
                  << boost::lexical_cast<std::string>(GetLastError());
        return;
    }
}

// Update the data to be exposed as the windows performance counter values.
void update_windows_counters(boost::uint64_t value)
{
    // Set raw counter data for queue length.
    ULONG status = PerfSetULongCounterValue
        (HPXHeartBeat, queue_counter, 1, ULONG(value));
    if (status != ERROR_SUCCESS) {
        std::cerr << "PerfSetCounterRefValue for 'sum_queue_counter' failed "
                     "with error code: "
                  << boost::lexical_cast<std::string>(GetLastError());
        return;
    }

    // Set raw counter data for average queue length.
    status = PerfSetULongCounterValue(HPXHeartBeat, avg_queue_counter, 2, ULONG(value));
    if (status != ERROR_SUCCESS) {
        std::cerr << "PerfSetCounterRefValue for 'avg_queue_counter' failed "
                     "with error code: "
                  << boost::lexical_cast<std::string>(GetLastError());
        return;
    }
}

// Uninstall the windows performance counters exposing  the HPX counters.
void uninstall_windows_counters()
{
    // uninstall counter instances
    if (queue_counter) {
        PerfDeleteInstance(HPXHeartBeat, queue_counter);
        queue_counter = 0;
    }
    if (avg_queue_counter) {
        PerfDeleteInstance(HPXHeartBeat, avg_queue_counter);
        avg_queue_counter = 0;
    }

    // uninstall counter set, detach this provider
    CounterCleanup();
}

#endif
