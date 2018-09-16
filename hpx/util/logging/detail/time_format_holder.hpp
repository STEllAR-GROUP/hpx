// time_format_holder.hpp

// Boost Logging library
//
// Author: John Torjo, www.torjo.com
//
// Copyright (C) 2007 John Torjo (see www.torjo.com for email)
//
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)
//
// See http://www.boost.org for updates, documentation, and revision history.
// See http://www.torjo.com/log2/ for more details


#ifndef JT28092007_time_format_holder_HPP_DEFINED
#define JT28092007_time_format_holder_HPP_DEFINED

#include <hpx/util/assert.hpp>
#include <hpx/util/logging/detail/fwd.hpp>

#include <algorithm>
#include <cstdio>
#include <cstddef>
#include <sstream>
#include <string>
#include <vector>

namespace hpx { namespace util { namespace logging { namespace detail {

/**
    This only holds the time format, and allows writing a certain time
*/
struct time_format_holder {
private:
    struct index_info {
        typedef std::size_t uint;

        index_info(uint src_idx_, int *format_idx_, int advance_size_ = 2,
                   int size_ = 2)
            : src_idx(src_idx_), format_idx(format_idx_),
              advance_size(advance_size_), size(size_) {}
        uint src_idx;
        int * format_idx;
        int advance_size;
        int size;

        static bool by_index(const index_info & first, const index_info & second) {
            return first.src_idx < second.src_idx;
        }
    };

public:
    bool operator==(const time_format_holder & other) const {
        return m_format == other.m_format;
    }


    /**
        constructs a time format holder object
    */
    time_format_holder(const std::string & format) : m_day(-1),
        m_month(-1), m_yy(-1), m_yyyy(-1), m_hour(-1), m_min(-1),
        m_sec(-1),m_millisec(-1),m_microsec(-1),m_nanosec(-1) {
        set_format(format);
    }

    void set_format(const std::string & format) {
        // format too big
        HPX_ASSERT( format.size() < 64);
        m_format.clear();

        m_day = -1; m_month = -1; m_yy = -1; m_yyyy = -1; m_hour = -1;
        m_min = -1; m_sec = -1;m_millisec = -1;m_microsec = -1;m_nanosec = -1;

        typedef std::size_t uint;
        uint day_idx    = format.find("$dd");
        uint month_idx  = format.find("$MM");
        uint yy_idx     = format.find("$yy");
        uint yyyy_idx   = format.find("$yyyy");
        uint hour_idx   = format.find("$hh");
        uint min_idx    = format.find("$mm");
        uint sec_idx    = format.find("$ss");
        uint millisec_idx    = format.find("$mili");
        uint microsec_idx    = format.find("$micro");
        uint nanosec_idx    = format.find("$nano");

        typedef std::vector<index_info> array;
        array indexes;
        if ( day_idx != std::string::npos)
            indexes.push_back( index_info(day_idx, &m_day) );
        if ( month_idx != std::string::npos)
            indexes.push_back( index_info(month_idx, &m_month) );

        if ( yy_idx != std::string::npos || yyyy_idx != std::string::npos) {
            if ( yyyy_idx  != std::string::npos)
                indexes.push_back( index_info(yyyy_idx, &m_yyyy, 4) ); //-V112
            else
                indexes.push_back( index_info(yy_idx, &m_yy) );
        }

        if ( hour_idx != std::string::npos)
            indexes.push_back( index_info(hour_idx, &m_hour ) );
        if ( min_idx != std::string::npos)
            indexes.push_back( index_info(min_idx, &m_min) );
        if ( sec_idx != std::string::npos)
            indexes.push_back( index_info(sec_idx, &m_sec) );
        if ( millisec_idx != std::string::npos)
            indexes.push_back( index_info(millisec_idx, &m_millisec, 4, 3) );
            //-V112 //-V525
        if ( microsec_idx != std::string::npos)
            indexes.push_back( index_info(microsec_idx, &m_microsec, 5, 6) );
        if ( nanosec_idx != std::string::npos)
            indexes.push_back( index_info(nanosec_idx, &m_nanosec, 4, 9) ); //-V112

        std::sort( indexes.begin(), indexes.end(), index_info::by_index);

        // create the format string, that we can actually pass to sprintf
        uint prev_idx = 0;
        int idx = 0;
        for ( array::iterator begin = indexes.begin(), end = indexes.end();
        begin != end; ++begin) {
            m_format += format.substr( prev_idx, begin->src_idx - prev_idx);
            *begin->format_idx = idx;
            std::ostringstream cur_sprintf_format;
            cur_sprintf_format << "%0" << begin->size << "d";
            m_format += cur_sprintf_format.str();
            prev_idx = static_cast<hpx::util::logging::detail
                ::time_format_holder::index_info::uint>(begin->src_idx +
                    static_cast<hpx::util::logging::detail::time_format_holder
                    ::index_info::uint>(begin->advance_size) + 1ul);
            ++idx;
        }

        m_format += format.substr(prev_idx);
    }

    void write_time(char buffer[], int day, int month,
        int year, int hour, int min, int sec, int millisec, int microsec,
        int nanosec) const {
        int vals[11];
        vals[m_day + 1]      = day;
        vals[m_month + 1]    = month;
        vals[m_yy + 1]       = year % 100;
        vals[m_yyyy + 1]     = year;
        vals[m_hour + 1]     = hour;
        vals[m_min + 1]      = min;
        vals[m_sec + 1]      = sec;
        vals[m_millisec + 1]        = millisec;
        vals[m_microsec + 1]        = microsec;
        vals[m_nanosec + 1]         = nanosec;

        // ignore value at index 0
        // - it's there so that I don't have to test for an index being -1
        sprintf( buffer, m_format.c_str(), vals[1], vals[2], vals[3],
            vals[4], vals[5], vals[6], vals[7], vals[8], vals[9], vals[10] );
    }

    void write_time(char buffer[], int day, int month, int year,
        int hour, int min, int sec) const {
        int vals[8];
        vals[m_day + 1]      = day;
        vals[m_month + 1]    = month;
        vals[m_yy + 1]       = year % 100;
        vals[m_yyyy + 1]     = year;
        vals[m_hour + 1]     = hour;
        vals[m_min + 1]      = min;
        vals[m_sec + 1]      = sec;

        // ignore value at index 0
        // - it's there so that I don't have to test for an index being -1
        sprintf( buffer, m_format.c_str(), vals[1], vals[2], vals[3],
            vals[4], vals[5], vals[6], vals[7] );
    }

private:
    // the indexes of each escape sequence within the format string
    int m_day, m_month, m_yy, m_yyyy, m_hour, m_min, m_sec, m_millisec,
        m_microsec, m_nanosec;
    std::string m_format;
};

}}}}

#endif
