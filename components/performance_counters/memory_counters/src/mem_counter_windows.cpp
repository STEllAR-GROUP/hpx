// Copyright (c) 2012 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if defined(HPX_WINDOWS)

#include <hpx/modules/errors.hpp>

#include <cstdint>
#include <string>

#include <windows.h>    // this must go before psapi.h

#include <psapi.h>

namespace hpx::performance_counters::memory {

    ///////////////////////////////////////////////////////////////////////////
    // returns virtual memory value
    std::uint64_t read_psm_virtual(bool)
    {
        PROCESS_MEMORY_COUNTERS_EX pmc = {};
        pmc.cb = sizeof(PROCESS_MEMORY_COUNTERS_EX);

        if (!GetProcessMemoryInfo(GetCurrentProcess(),
                reinterpret_cast<PPROCESS_MEMORY_COUNTERS>(&pmc),
                sizeof(PROCESS_MEMORY_COUNTERS_EX)))
        {
            HRESULT const hr = GetLastError();
            LPVOID buffer = nullptr;
            if (!FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER |
                        FORMAT_MESSAGE_FROM_SYSTEM |
                        FORMAT_MESSAGE_IGNORE_INSERTS,
                    nullptr, hr,
                    MAKELANGID(
                        LANG_NEUTRAL, SUBLANG_DEFAULT),    // Default language
                    reinterpret_cast<LPTSTR>(&buffer), 0, nullptr))
            {
                HPX_THROW_EXCEPTION(hpx::error::kernel_error,
                    "hpx::performance_counters::memory::read_psm_virtual",
                    "format message failed with {:x} (while retrieving message "
                    "for {:x})",
                    GetLastError(), hr);
            }

            std::string const msg(static_cast<char*>(buffer));
            LocalFree(buffer);
            HPX_THROW_EXCEPTION(hpx::error::kernel_error,
                "hpx::performance_counters::memory::read_psm_virtual", msg);
        }

        return pmc.WorkingSetSize;
    }

    ///////////////////////////////////////////////////////////////////////////
    // returns resident memory value
    std::uint64_t read_psm_resident(bool)
    {
        PROCESS_MEMORY_COUNTERS_EX pmc = {};
        pmc.cb = sizeof(PROCESS_MEMORY_COUNTERS_EX);

        if (!GetProcessMemoryInfo(GetCurrentProcess(),
                reinterpret_cast<PPROCESS_MEMORY_COUNTERS>(&pmc),
                sizeof(PROCESS_MEMORY_COUNTERS_EX)))
        {
            HRESULT const hr = GetLastError();
            LPVOID buffer = nullptr;
            if (!FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER |
                        FORMAT_MESSAGE_FROM_SYSTEM |
                        FORMAT_MESSAGE_IGNORE_INSERTS,
                    nullptr, hr,
                    MAKELANGID(
                        LANG_NEUTRAL, SUBLANG_DEFAULT),    // Default language
                    reinterpret_cast<LPTSTR>(&buffer), 0, nullptr))
            {
                HPX_THROW_EXCEPTION(hpx::error::kernel_error,
                    "hpx::performance_counters::memory::read_psm_resident",
                    "format message failed with {:x} (while retrieving message "
                    "for {:x})",
                    GetLastError(), hr);
            }

            std::string const msg(static_cast<char*>(buffer));
            LocalFree(buffer);
            HPX_THROW_EXCEPTION(hpx::error::kernel_error,
                "hpx::performance_counters::memory::read_psm_resident", msg);
        }

        return pmc.PrivateUsage;
    }

    // Returns total available memory
    std::uint64_t read_total_mem_avail(bool)
    {
        MEMORYSTATUSEX mem_status = {};
        mem_status.dwLength = sizeof(MEMORYSTATUSEX);

        if (!GlobalMemoryStatusEx(&mem_status))
        {
            HRESULT const hr = GetLastError();
            LPVOID buffer = nullptr;
            if (!FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER |
                        FORMAT_MESSAGE_FROM_SYSTEM |
                        FORMAT_MESSAGE_IGNORE_INSERTS,
                    nullptr, hr,
                    MAKELANGID(
                        LANG_NEUTRAL, SUBLANG_DEFAULT),    // Default language
                    reinterpret_cast<LPTSTR>(&buffer), 0, nullptr))
            {
                HPX_THROW_EXCEPTION(hpx::error::kernel_error,
                    "hpx::performance_counters::memory::read_total_mem_avail",
                    "format message failed with {:x} (while "
                    "retrieving message for {:x})",
                    GetLastError(), hr);
            }

            std::string const msg(static_cast<char*>(buffer));
            LocalFree(buffer);
            HPX_THROW_EXCEPTION(hpx::error::kernel_error,
                "hpx::performance_counters::memory::read_total_mem_avail", msg);
        }

        return mem_status.ullAvailPhys;
    }
}    // namespace hpx::performance_counters::memory

#endif
