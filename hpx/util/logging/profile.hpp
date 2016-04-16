// profiler.hpp

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

/**
    @file hpx/util/logging/profile.hpp
    @brief Allows you to profile your application' logging

    That is, see how much CPU time is taken for logging
*/

#ifndef JT28092007_profiler_HPP_DEFINED
#define JT28092007_profiler_HPP_DEFINED

#if defined(HPX_MSVC) && (HPX_MSVC >= 1020)
# pragma once
#endif

#include <hpx/util/logging/detail/fwd.hpp>

#include <hpx/util/function.hpp>

#include <boost/date_time/microsec_time_clock.hpp>
#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <boost/date_time/posix_time/ptime.hpp>
#include <boost/function.hpp>
#include <sstream>
#include <string>
#include <vector>
#include <hpx/util/logging/format_fwd.hpp>
#include <hpx/util/logging/detail/find_format_writer.hpp>

namespace hpx { namespace util { namespace logging {

namespace writer {
    template<class msg_type, class base_type> struct on_dedicated_thread ;
}

/** @brief Allows profiling your application

If you want to profile your application (find out how much time is spent logging),
all you need to do is to surround your logger and filter class(es)
by profile::compute_for_logger and with profile::compute_for_filter,
like shown below:

Your code:
@code
namespace bl = hpx::util::logging;
typedef bl::logger_format_write< > logger_type;
typedef bl::filter::no_ts filter_type;
@endcode


Your code with profiling:
@code
// notice the extra include
#include <hpx/util/logging/profile.hpp>

namespace bl = hpx::util::logging;
typedef bl::logger_format_write< > raw_logger_type;
typedef bl::profile::compute_for_logger<raw_logger_type>::type logger_type;
typedef bl::filter::no_ts raw_filter_type;
typedef bl::profile::compute_for_filter<raw_filter_type>::type filter_type;
@endcode

In addition to the above, you'll need to set a place where to dump
the profile information (which will be dumped at end of the program).
This is just a functor that takes a <tt>const std::string&</tt> argument.
Thus, it can be any destination class. For instance:

@code
// where shall the profile results be outputted?
bl::profile::compute::inst().log_results( bl::destination::file("profile.txt") );
@endcode

\n
Results can look like this:

@code
gather time:      5.562500 seconds
write time:       5.265625 seconds
filter time:      0.31250 seconds
otherthread time: 0.0 seconds
@endcode


\n\n
For more info, see compute_for_logger and compute_for_filter classes.
*/
namespace profile {


/**
    @brief Computes profiling information, and can show it to the user

*/
struct compute {
    enum type {
        gather = 0,
        writer,
        filter,
        on_other_thread,

        last_ = on_other_thread
    };

    typedef ::hpx::util::function_nonser<void(const std::string&)> log_function;
    static void nothing(const std::string&) {}

    static compute & inst() {
        static compute c;
        return c;
    }

    void add_period(const ::boost::int64_t & mcs, type t) {
        // note : this will work even after this object has been destroyed
        // (since it's a global)
        m_cpu_times[t] += mcs;
    }

    void log_results(log_function l) {
        m_log = l;
    }
protected:
    compute() {
        m_log = &nothing;
    }
    ~compute() {
        dump_results();
    }
private:
    void dump_results() {
        std::ostringstream out;
        out << "gather time:      "
            << (m_cpu_times[gather] / 1000000) << "."
            << (m_cpu_times[gather] % 1000000) << " seconds " << std::endl;
        out << "write time:       "
            << (m_cpu_times[writer] / 1000000) << "."
            << (m_cpu_times[writer] % 1000000) << " seconds " << std::endl;
        out << "filter time:      "
            << (m_cpu_times[filter] / 1000000) << "."
            << (m_cpu_times[filter] % 1000000) << " seconds " << std::endl;
        out << "otherthread time: "
            << (m_cpu_times[on_other_thread] / 1000000) << "."
            << (m_cpu_times[on_other_thread] % 1000000) << " seconds " << std::endl;
        m_log( out.str() );
    }

private:
    compute(const compute&);
    void operator=(const compute&);

private:
    // the CPU time taken for each activity
    //
    // note: we don't use std::vector;
    // we want this to be available even after destruction
    // (if logs are used after they've been destroyed
    boost::int64_t m_cpu_times[ last_ + 1];

    // where should we dump the results?
    log_function m_log;
};



struct scoped_compute {
    compute& m_comp;
    compute::type m_type;

    ::boost::posix_time::ptime m_start, m_end;

    scoped_compute(compute& comp, compute::type type) : m_comp(comp), m_type(type) {
        m_start = ::boost::posix_time::microsec_clock::local_time();
    }
    ~scoped_compute() {
        m_end = ::boost::posix_time::microsec_clock::local_time();
        m_comp.add_period( (m_end - m_start).total_microseconds() , m_type);
    }

};



template<class gather_msg> struct compute_gather : gather_msg {
    compute_gather() : m_lk( get_compute(), compute::gather) {}
    compute_gather(const compute_gather& other) : m_lk( get_compute(), compute::gather),
        gather_msg(other) {}

    compute& get_compute() const { return compute::inst(); }
private:
    scoped_compute m_lk;
};

template<class writer_msg> struct compute_write : writer_msg {
    compute& get_compute() const { return compute::inst(); }

    template<class msg_type> void operator()(msg_type& msg) const {
        scoped_compute lk( get_compute(), compute::writer );
        writer_msg::operator()(msg);
    }

    // just in case you do write on another thread
    virtual void write_array() const {
        scoped_compute lk( get_compute(), compute::on_other_thread );
        write_array_impl( this);
    }
private:
    template<class msg_type, class base_type> void write_array_impl(const
        ::hpx::util::logging::writer::on_dedicated_thread<msg_type,base_type> *) const {
        // call base class's implementation
        writer_msg::write_array();
    }

    // does not derive from on_dedicated_thread
    void write_array_impl(const void *) const {}
};

/** @brief Profiles a filter. Don't use directly, use compute_for_filter instead.

*/
template<class filter_msg> struct compute_filter : filter_msg {
    HPX_LOGGING_FORWARD_CONSTRUCTOR(compute_filter, filter_msg)

    // is_enabled - for any up to 5 params - const function
    compute& get_compute() const { return compute::inst(); }

    bool is_enabled() const {
        scoped_compute lk( get_compute(), compute::filter );
        return filter_msg::is_enabled();
    }
    template<class p1> bool is_enabled(const p1 & v1) const {
        scoped_compute lk( get_compute(), compute::filter );
        return filter_msg::is_enabled(v1);
    }
    template<class p1, class p2> bool is_enabled(const p1 & v1, const p2 &v2) const {
        scoped_compute lk( get_compute(), compute::filter );
        return filter_msg::is_enabled(v1, v2);
    }
    template<class p1, class p2, class p3> bool is_enabled(const p1 & v1, const p2 &v2,
        const p3 & v3) const {
        scoped_compute lk( get_compute(), compute::filter );
        return filter_msg::is_enabled(v1, v2, v3);
    }
    template<class p1, class p2, class p3, class p4> bool is_enabled(const p1 & v1,
        const p2 &v2, const p3 & v3, const p4 & v4) const {
        scoped_compute lk( get_compute(), compute::filter );
        return filter_msg::is_enabled(v1, v2, v3, v4);
    }
    template<class p1, class p2, class p3, class p4, class p5>
    bool is_enabled(const p1 & v1, const p2 &v2, const p3 & v3,
        const p4 & v4, const p5 & v5) const {
        scoped_compute lk( get_compute(), compute::filter );
        return filter_msg::is_enabled(v1, v2, v3, v4, v5);
    }

};




/** @brief given the logger type, gets the write_msg part,
without needing to know the logger's definition (a typedef is enough)

*/
template<class> struct logger_to_write {};
template<class gather_msg, class write_msg>
struct logger_to_write< logger<gather_msg,write_msg> > {
    // ... the easy part
    typedef write_msg write_type;
};

// specialize for logger_format_write
template<class format_base, class destination_base,
class thread_safety, class gather, class lock_resource>
        struct logger_to_write< logger_format_write<format_base,
            destination_base, thread_safety, gather, lock_resource> > {

    typedef typename detail::format_find_writer<format_base,
        destination_base, lock_resource, thread_safety>::type write_type;
};



/** @brief Allows you to compute profiling for your logger class

@code
#include <hpx/util/logging/profile.hpp>
@endcode

To do profiling for a logger, just surround it with compute_for_logger. Example:

<b>Old code</b>

@code
#include <hpx/util/logging/format_fwd.hpp>

namespace bl = hpx::util::logging ;
typedef bl::logger_format_write< > logger_type;

HPX_DECLARE_LOG(g_l, logger_type)
...
HPX_DEFINE_LOG(g_l, logger_type)

@endcode


<b>New code</b>

@code
#include <hpx/util/logging/format_fwd.hpp>
#include <hpx/util/logging/profile.hpp>

namespace bl = hpx::util::logging ;
typedef bl::logger_format_write< > raw_log_type;
typedef bl::profile::compute_for_logger<raw_log_type>::type logger_type;

HPX_DECLARE_LOG(g_l, logger_type)
...
HPX_DEFINE_LOG(g_l, logger_type)

@endcode

@sa compute_for_filter

*/
template<class logger_type> struct compute_for_logger {
    typedef logger<
        compute_gather< typename logger_to_gather<logger_type>::gather_type > ,
        compute_write< typename logger_to_write<logger_type>::write_type > > type;

};




/** @brief Allows you to compute profiling for your filter class

@code
#include <hpx/util/logging/profile.hpp>
@endcode

In case you want to profile your filter, there's just one requirement:
- your function must be called @c is_enabled() and be const






To do profiling for a filter, just surround it with compute_for_filter. Example:

<b>Old code</b>

@code
#include <hpx/util/logging/format_fwd.hpp>

namespace bl = hpx::util::logging ;
typedef bl::filter::no_ts filter;

HPX_DECLARE_LOG_FILTER(g_l_filter, filter)
...
HPX_DEFINE_LOG_FILTER(g_l_filter, filter)

@endcode


<b>New code</b>

@code
#include <hpx/util/logging/format_fwd.hpp>
#include <hpx/util/logging/profile.hpp>

namespace bl = hpx::util::logging ;
typedef bl::filter::no_ts raw_filter;
typedef compute_for_filter<raw_filter>::type filter;

HPX_DECLARE_LOG_FILTER(g_l_filter, filter)
...
HPX_DEFINE_LOG_FILTER(g_l_filter, filter)

@endcode







@sa compute_for_logger

*/
template<class filter_type> struct compute_for_filter {
    typedef compute_filter<filter_type> type;
};



}

}}}

#endif

