// format.hpp

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

// this is fixed!
#ifndef JT28092007_format_HPP_DEFINED
#define JT28092007_format_HPP_DEFINED

#include <hpx/util/assert.hpp>
#include <hpx/util/logging/detail/fwd.hpp>
#include <hpx/util/logging/detail/manipulator.hpp>
#include <hpx/util/logging/format/array.hpp>
#include <hpx/util/logging/format/op_equal.hpp>
#include <hpx/util/logging/format_fwd.hpp>

#include <algorithm>
#include <memory>
#include <set>
#include <string>
#include <type_traits>
#include <vector>

namespace hpx { namespace util { namespace logging {

/**
@file hpx/util/logging/format.hpp

Include this file when you're using @ref manipulator "formatters and destinations",
and you want to define the logger classes, in a source file
(using HPX_DEFINE_LOG)

*/

    ///////////////////////////////////////////////////////////////////////////
    // Format and write
    //

    /**
        @brief The @c %format_and_write classes know how to call
        the formatter and destination @c objects.

        Usually you'll be happy with the
        format_and_write::simple class - which simply calls @c
        operator() on the formatters , and @c operator() on the destinations.

        Note that usually the formatter and destination class just have an @c operator(),
        which when called, formats the message
        or writes it to a destination. In case your formatters/destinations are
        more complex than that (for instance, more than
        a member function needs to be called),
        you'll have to implement your own %format_and_write class.
    */
    namespace format_and_write {


    /**
        @brief Formats the message, and writes it to destinations
        - calls @c operator() on the formatters , and @c operator() on the destinations.
        Ignores @c clear_format() commands.

        @param msg_type The message to pass to the formatter. This is the
        type that is passed to the formatter objects and to the destination objects.
        Thus, it needs to be convertible to the argument to be sent to the
        formatter objects and to the argument to be sent to the destination objects.
        Usually, it's the argument you pass on to your destination classes.

        If you derive from @c destination::base, this type can be
        @c destination::base::raw_param (see below).

        Example:

        @code
        typedef destination::base<const std::string &> dest_base;
        // in this case : msg_type = std::string = dest_base::raw_param
        struct write_to_cout : dest_base {
            void operator()(param msg) const {
                std::cout << msg ;
            }
        };


        typedef destination::base<const std::string &> dest_base;
        // in this case : msg_type = cache_string = dest_base::raw_param
        struct write_to_file : dest_base, ... {
            void operator()(param msg) const {
                context() << msg ;
            }
        };

        @endcode
    */
    struct simple {
        simple ( msg_type & msg) : m_msg(msg) {}

        template<class formatter_ptr> void format(const formatter_ptr & fmt) {
            (*fmt)(m_msg);
        }
        template<class destination_ptr> void write(const destination_ptr & dest) {
            (*dest)(m_msg);
        }
        void clear_format() {}
    protected:
        msg_type &m_msg;
    };

    } // namespace format_and_write



    ///////////////////////////////////////////////////////////////////////////
    // Message routing
    //

    /**
    @brief Specifies the route : how formatting and writing to destinations take place.

    Classes in this namespace specify when formatters and destinations are to be called.

    @sa msg_route::simple

    */
    namespace msg_route {

    /**
        @brief Recomended base class for message routers that
        need access to the underlying formatter and/or destination array.
    */
    template<class formatter_array, class destination_array>
    struct formatter_and_destination_array_holder {
    protected:
        formatter_and_destination_array_holder (const formatter_array & formats_,
            const destination_array & destinations_)
            : m_formats(formats_), m_destinations(destinations_) {}

        const formatter_array & formats() const             { return m_formats; }
        const destination_array & destinations() const      { return m_destinations; }

    private:
        const formatter_array & m_formats;
        const destination_array & m_destinations;
    };

/**
@brief Represents a simple router - first calls all formatters
- in the order they were added, then all destinations - in the order they were added

Example:

@code
typedef logger< format_write > logger_type;
HPX_DEFINE_LOG_FILTER(g_log_filter, filter::no_ts )
HPX_DEFINE_LOG(g_l, logger_type)
#define L_ HPX_LOG_USE_LOG_IF_FILTER(g_l(), g_log_filter()->is_enabled() )

// add formatters : [idx] [time] message [enter]
g_l()->writer().add_formatter( write_idx() );
g_l()->writer().add_formatter( write_time() );
g_l()->writer().add_formatter( append_newline() );

// write to cout and file
g_l()->writer().add_destination( write_to_cout() );
g_l()->writer().add_destination( write_to_file("out.txt") );

// usage
int i = 1;
L_ << "testing " << i << i+1 << i+2;
@endcode

In the above case:
- First, the formatters are called: @c write_idx() is called, then @c write_time(),
then @c append_newline().
- Then, the destinations are called: @c write_to_cout(), and then @c write_to_file().



@param format_base The base class for all formatter classes from your application.
See manipulator.

@param destination_base The base class for all destination classes from your application.
See manipulator.

    */\
    struct simple {
        typedef typename formatter::base::ptr_type formatter_ptr;
        typedef typename destination::base::ptr_type destination_ptr;

        typedef std::vector<formatter_ptr> f_array;
        typedef std::vector<destination_ptr> d_array;
        struct write_info {
            f_array formats;
            d_array destinations;
        };

        template<class formatter_array, class destination_array>
        simple(const formatter_array&, const destination_array&) {}

        void append_formatter(formatter_ptr fmt) {
            m_to_write.formats.push_back(fmt);
        }
        void del_formatter(formatter_ptr fmt) {
            typename f_array::iterator del = std::remove(m_to_write.formats.begin(),
                m_to_write.formats.end(), fmt);
            m_to_write.formats.erase(del, m_to_write.formats.end());
        }

        void append_destination(destination_ptr dest) {
            m_to_write.destinations.push_back(dest);
        }

        void del_destination(destination_ptr dest) {
            typename d_array::iterator del =
                std::remove(m_to_write.destinations.begin(),
                    m_to_write.destinations.end(), dest);
            m_to_write.destinations.erase(del, m_to_write.destinations.end());
        }

        template<class format_and_write> void write(msg_type & msg) const {
            format_and_write m(msg);

            for ( typename f_array::const_iterator b_f = m_to_write.formats.begin(),
                e_f = m_to_write.formats.end(); b_f != e_f; ++b_f)
                m.format(*b_f);

            for ( typename d_array::const_iterator b_d = m_to_write.destinations.begin(),
                e_d = m_to_write.destinations.end(); b_d != e_d; ++b_d)
                m.write(*b_d);
        }

    private:
        write_info m_to_write;
    };


    } // namespace msg_route
}}}

#include <hpx/util/logging/detail/manipulator.hpp>
#include <hpx/util/logging/detail/format_write_detail.hpp>

#include <hpx/util/logging/format/formatter/defaults.hpp>
#include <hpx/util/logging/format/destination/defaults.hpp>

#endif
