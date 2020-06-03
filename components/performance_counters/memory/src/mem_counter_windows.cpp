// Copyright (c) 2012 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if defined(HPX_WINDOWS)

#include <hpx/modules/errors.hpp>
#include <hpx/modules/format.hpp>

#include <cstdint>
#include <cstring>
#include <string>

#include <windows.h> // this must go before psapi.h
#include <psapi.h>

namespace hpx { namespace performance_counters { namespace memory
{
    ///////////////////////////////////////////////////////////////////////////
    // returns virtual memory value
    std::uint64_t read_psm_virtual(bool)
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
                nullptr, hr,
                MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), // Default language
                (LPTSTR) &buffer, 0, nullptr))
            {
                HPX_THROW_EXCEPTION(kernel_error,
                    "hpx::performance_counters::memory::read_psm_virtual",
                    hpx::util::format("format message failed with {:x} (while "
                        "retrieving message for {:x})", GetLastError(), hr));
                return std::uint64_t(-1);
            }

            std::string msg(static_cast<char*>(buffer));
            LocalFree(buffer);
            HPX_THROW_EXCEPTION(kernel_error,
                "hpx::performance_counters::memory::read_psm_virtual", msg);
            return std::uint64_t(-1);
        }

        return pmc.WorkingSetSize;
    }

    ///////////////////////////////////////////////////////////////////////////
    // returns resident memory value
    std::uint64_t read_psm_resident(bool)
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
                nullptr, hr,
                MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), // Default language
                (LPTSTR) &buffer, 0, nullptr))
            {
                HPX_THROW_EXCEPTION(kernel_error,
                    "hpx::performance_counters::memory::read_psm_resident",
                    hpx::util::format("format message failed with {:x} (while "
                        "retrieving message for {:x})", GetLastError(), hr));
                return std::uint64_t(-1);
            }

            std::string msg(static_cast<char*>(buffer));
            LocalFree(buffer);
            HPX_THROW_EXCEPTION(kernel_error,
                "hpx::performance_counters::memory::read_psm_resident", msg);
            return std::uint64_t(-1);
        }

        return pmc.PrivateUsage;
    }

    // Returns total available memory
    std::uint64_t read_total_mem_avail(bool)
    {
        MEMORYSTATUSEX mem_status;
        std::memset(&mem_status, '\0', sizeof(MEMORYSTATUSEX));
        mem_status.dwLength = sizeof(MEMORYSTATUSEX);

        if (!GlobalMemoryStatusEx(&mem_status))
        {
            HRESULT hr = GetLastError();
            LPVOID buffer = 0;
            if (!FormatMessage(
                FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                FORMAT_MESSAGE_IGNORE_INSERTS,
                nullptr, hr,
                MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), // Default language
                (LPTSTR) &buffer, 0, nullptr))
            {
                HPX_THROW_EXCEPTION(kernel_error,
                    "hpx::performance_counters::memory::read_total_mem_avail",
                    hpx::util::format("format message failed with {:x} (while "
                        "retrieving message for {:x})", GetLastError(), hr));
                return std::uint64_t(-1);
            }

            std::string msg(static_cast<char*>(buffer));
            LocalFree(buffer);
            HPX_THROW_EXCEPTION(kernel_error,
                "hpx::performance_counters::memory::read_total_mem_avail", msg);
            return std::uint64_t(-1);
        }

        return mem_status.ullAvailPhys;
    }
}}}

#endif
