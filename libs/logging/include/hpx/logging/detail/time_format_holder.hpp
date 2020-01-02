// time_format_holder.hpp

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

#ifndef JT28092007_time_format_holder_HPP_DEFINED
#define JT28092007_time_format_holder_HPP_DEFINED

#include <hpx/format.hpp>

#include <cstddef>
#include <string>

namespace hpx { namespace util { namespace logging { namespace detail {

    /**
    This only holds the time format, and allows writing a certain time
*/
    struct time_format_holder
    {
    public:
        bool operator==(const time_format_holder& other) const
        {
            return m_format == other.m_format;
        }

        /**
        constructs a time format holder object
    */
        time_format_holder(const std::string& format)
        {
            set_format(format);
        }

        void set_format(const std::string& format)
        {
            m_format = format;
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

        std::string write_time(int day, int month, int year, int hour, int min,
            int sec, int millisec = 0, int microsec = 0, int nanosec = 0) const
        {
            return hpx::util::format(m_format, day, month, year, year % 100,
                hour, min, sec, millisec, microsec, nanosec);
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

        // the indexes of each escape sequence within the format string
        std::string m_format;
    };

}}}}    // namespace hpx::util::logging::detail

#endif
