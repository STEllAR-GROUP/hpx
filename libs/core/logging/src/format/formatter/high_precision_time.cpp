// high_precision_time.hpp

// Boost Logging library
//
// Author: John Torjo, www.torjo.com
//
// Copyright (C) 2007 John Torjo (see www.torjo.com for email)
//
//  SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)
//
// See http://www.boost.org for updates, documentation, and revision history.
// See http://www.torjo.com/log2/ for more details

#include <hpx/logging/format/formatters.hpp>

#include <hpx/config.hpp>
#include <hpx/modules/format.hpp>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <memory>
#include <ostream>
#include <string>

#if !(defined(__linux) || defined(linux) || defined(__linux__) ||              \
    defined(__FreeBSD__) || defined(__APPLE__) || defined(HPX_MSVC))
#include <mutex>
#endif

namespace hpx { namespace util { namespace logging { namespace formatter {

    high_precision_time::~high_precision_time() = default;

    struct high_precision_time_impl : high_precision_time
    {
        /**
        constructs a high_precision_time object
    */
        explicit high_precision_time_impl(std::string const& format)
          : high_precision_time(format)
        {
        }

        void operator()(std::ostream& to) const override
        {
            auto const val = std::chrono::system_clock::now();
            std::time_t const tt = std::chrono::system_clock::to_time_t(val);

#if defined(__linux) || defined(linux) || defined(__linux__) ||                \
    defined(__FreeBSD__) || defined(__APPLE__)
            std::tm local_tm;
            localtime_r(&tt, &local_tm);
#elif defined(HPX_MSVC)
            std::tm local_tm;
            localtime_s(&local_tm, &tt);
#else
            // fall back to non-thread-safe version on other platforms
            std::tm local_tm;
            {
                static hpx::util::detail::spinlock mutex;
                std::lock_guard<hpx::util::detail::spinlock> ul(mutex);
                local_tm = *std::localtime(&tt);
            }
#endif

            std::chrono::nanoseconds const nanosecs =
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    val.time_since_epoch());
            std::chrono::microseconds const microsecs =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    val.time_since_epoch());
            std::chrono::milliseconds const millisecs =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    val.time_since_epoch());

            util::format_to(to, m_format, local_tm.tm_mday, local_tm.tm_mon + 1,
                local_tm.tm_year + 1900, local_tm.tm_year % 100,
                local_tm.tm_hour, local_tm.tm_min, local_tm.tm_sec,
                millisecs.count() % 1000, microsecs.count() % 1000,
                nanosecs.count() % 1000);
        }

        /** @brief configure through script

        the string = the time format
    */
        void configure(std::string const& str) override
        {
            m_format = str;
            replace_format("$dd", "{1:02d}");
            replace_format("$MM", "{2:02d}");
            replace_format("$yyyy", "{3:04d}");
            replace_format("$yy", "{4:02d}");
            replace_format("$hh", "{5:02d}");
            replace_format("$mm", "{6:02d}");
            replace_format("$ss", "{7:02d}");
            replace_format("$mili", "{8:03d}");
            replace_format("$micro", "{9:06d}");
            replace_format("$nano", "{10:09d}");
        }

    private:
        bool replace_format(char const* from, char const* to)
        {
            size_t start_pos = m_format.find(from);
            if (start_pos == std::string::npos)
                return false;
            m_format.replace(start_pos, std::strlen(from), to);
            return true;
        }

        std::string m_format;
    };

    std::unique_ptr<high_precision_time> high_precision_time::make(
        std::string const& format)
    {
        return std::unique_ptr<high_precision_time>(
            new high_precision_time_impl(format));
    }

}}}}    // namespace hpx::util::logging::formatter
