// Copyright (c) 2012 Hartmut Kaiser
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if defined(HPX_WINDOWS)

#include <hpx/exception.hpp>
#include <boost/format.hpp>

#include <string>

#include <psapi.h>

namespace hpx { namespace performance_counters { namespace memory
{
    ///////////////////////////////////////////////////////////////////////////
    // returns virtual memory value
    boost::uint64_t read_psm_virtual(bool)
    {
        PROCESS_MEMORY_COUNTERS_EX pmc;
        std::memset(&pmc, '\0', sizeof(PROCESS_MEMORY_COUNTERS_EX));
        pmc.cb = sizeof(PROCESS_MEMORY_COUNTERS_EX);

        if (!GetProcessMemoryInfo(GetCurrentProcess(),
                reinterpret_cast<PPROCESS_MEMORY_COUNTERS>(&pmc),
                sizeof(PROCESS_MEMORY_COUNTERS_EX)))
        {
            HRESULT hr = GetLastError();
            LPVOID buffer = 0;
            if (!FormatMessage(
                FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                FORMAT_MESSAGE_IGNORE_INSERTS,
                NULL, hr,
                MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), // Default language
                (LPTSTR) &buffer, 0, NULL))
            {
                HPX_THROW_EXCEPTION(kernel_error,
                    "hpx::performance_counters::memory::read_psm_virtual",
                    boost::str(boost::format("format message failed with %x (while "
                        "retrieving message for %x)") % GetLastError() % hr));
                return boost::uint64_t(-1);
            }

            std::string msg(static_cast<char*>(buffer));
            LocalFree(buffer);
            HPX_THROW_EXCEPTION(kernel_error,
                "hpx::performance_counters::memory::read_psm_virtual", msg);
            return boost::uint64_t(-1);
        }

        return pmc.WorkingSetSize;
    }

    ///////////////////////////////////////////////////////////////////////////
    // returns resident memory value
    boost::uint64_t read_psm_resident(bool)
    {
        PROCESS_MEMORY_COUNTERS_EX pmc;
        std::memset(&pmc, '\0', sizeof(PROCESS_MEMORY_COUNTERS_EX));
        pmc.cb = sizeof(PROCESS_MEMORY_COUNTERS_EX);

        if (!GetProcessMemoryInfo(GetCurrentProcess(),
                reinterpret_cast<PPROCESS_MEMORY_COUNTERS>(&pmc),
                sizeof(PROCESS_MEMORY_COUNTERS_EX)))
        {
            HRESULT hr = GetLastError();
            LPVOID buffer = 0;
            if (!FormatMessage(
                FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                FORMAT_MESSAGE_IGNORE_INSERTS,
                NULL, hr,
                MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), // Default language
                (LPTSTR) &buffer, 0, NULL))
            {
                HPX_THROW_EXCEPTION(kernel_error,
                    "hpx::performance_counters::memory::read_psm_resident",
                    boost::str(boost::format("format message failed with %x (while "
                        "retrieving message for %x)") % GetLastError() % hr));
                return boost::uint64_t(-1);
            }

            std::string msg(static_cast<char*>(buffer));
            LocalFree(buffer);
            HPX_THROW_EXCEPTION(kernel_error,
                "hpx::performance_counters::memory::read_psm_resident", msg);
            return boost::uint64_t(-1);
        }

        return pmc.PrivateUsage;
    }
}}}

#endif
