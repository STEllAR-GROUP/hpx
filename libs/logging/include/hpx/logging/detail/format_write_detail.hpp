// format_write_detail.hpp

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

// this is fixed!
#ifndef JT28092007_format_write_detail_HPP_DEFINED
#define JT28092007_format_write_detail_HPP_DEFINED

#include <hpx/logging/detail/fwd.hpp>

#include <memory>
#include <type_traits>

namespace hpx { namespace util { namespace logging {

    namespace format_and_write {
        struct simple;
    }

    /**
@brief Classes that write the message, once it's been @ref gather "gathered".

The most important class is writer::format_write

*/
    namespace writer {

        /**
@brief Allows custom formatting of the message before %logging it,
and writing it to several destinations.

Once the message has been "gathered", it's time to write it.
The current class defines the following concepts:
- formatter - allows formatting the message before writing it
- destination - is a place where the message is to be written to (like, the console,
a file, a socket, etc.)

You can add several formatters and destinations. Note that each formatter class and
each destination class is a @c %manipulator.
Make sure you know what a manipulator is before using formatters and destinations.



\n\n
@section object_router The router object

Once you've added the formatters and destinations,
the @ref msg_route "router" comes into play. The @ref msg_route "router"
specifies how formatters and destinations are called.
By default, all formatters are called first, in the order they were added,
and then all destinations are called, in the order they were added.
You can easily access the router() instance.

@code
typedef logger< format_write > logger_type;
HPX_DECLARE_LOG(g_l, logger_type)
HPX_DECLARE_LOG_FILTER(g_log_filter, filter::no_ts )
#define L_ HPX_LOG_USE_LOG_IF_FILTER(g_l(), g_log_filter()->is_enabled() )

// add formatters : [idx] [time] message [enter]
g_l()->writer().add_formatter( formatter::idx() );
g_l()->writer().add_formatter( formatter::time("$hh:$mm.$ss ") );

// write to cout and file
g_l()->writer().add_destination( destination::cout() );
g_l()->writer().add_destination( destination::file("out.txt") );

// usage
int i = 1;
L_ << "testing " << i << i+1 << i+2;
@endcode

In the above case, @c formatter::idx() is called, then @c formatter::time().
Now, the destinations are called: @c destination::cout(), and then
@c destination::file().



\n\n
@section apply_format_and_write_object The apply_format_and_write object

Once the formatters and destinations are added, and you know the route, you have an
extra object - the format_and_write - which
contains the logic for calling the formatters and destinations.
The format_and_write class knows how to call the formatters and destinations @em objects.
Usually you'll be happy with the
format_and_write::simple class - which simply calls @c operator() on the formatters,
and @c operator() on the destinations.
Otherwise, take a look at format_and_write namespace.

An object of this type (apply_format_and_write) is created for each new logged message.


\n\n
@note This class is not thread-safe. If you want thread-safety,
check out the other writer classes: on_dedicated_thread and ts_write



\n\n
@param format_base The base class for all formatter classes from your application.
See manipulator.

@param destination_base The base class for all destination classes from your application.
See manipulator.

@param apply_format_and_write [optional] The class that knows how to call
the formatters and destinations. See @ref apply_format_and_write_object

@param router_type [optional] The class that knows when to call the formatters,
and when to call the destinations. See @ref object_router.



\n\n
@remarks Normally the router could own the formatters and destination objects.
However, then, it would need to own the objects,
which would mean needing to come up with a smart pointer strategy.
This would complicate the router logic.
Also, iterating over formatters/destinations would be slower,
if we were to keep smart pointers within the router itself.



@bug adding a spaced generic formatter and deleting the formatter - it won't happen

*/
        struct format_write
        {
            using formatter_base = formatter::base;
            using destination_base = destination::base;
            using router_type = msg_route::simple;
            using formatter_array = array::ptr_holder<formatter_base>;
            using destination_array = array::ptr_holder<destination_base>;

            typedef typename formatter_base::ptr_type formatter_ptr;
            typedef typename destination_base::ptr_type destination_ptr;

            typedef ::hpx::util::logging::format_and_write::simple
                apply_format_and_write_type;

            format_write()
              : m_router(m_formatters, m_destinations)
            {
            }

        private:
            // non-generic
            template <class Formatter>
            void add_formatter_impl(Formatter fmt, const std::false_type&)
            {
                formatter_ptr p = m_formatters.append(fmt);
                m_router.append_formatter(p);
            }

            // non-generic
            template <class Formatter>
            void del_formatter_impl(Formatter fmt, const std::false_type&)
            {
                formatter_ptr p = m_formatters.get_ptr(fmt);
                m_router.del_formatter(p);
                m_formatters.del(fmt);
            }

            // non-generic
            template <class Destination>
            void add_destination_impl(Destination dest, const std::false_type&)
            {
                destination_ptr p = m_destinations.append(dest);
                m_router.append_destination(p);
            }

            // non-generic
            template <class Destination>
            void del_destination_impl(Destination dest, const std::false_type&)
            {
                destination_ptr p = m_destinations.get_ptr(dest);
                m_router.del_destination(p);
                m_destinations.del(dest);
            }

            // generic manipulator
            template <class Formatter>
            void add_formatter_impl(Formatter fmt, const std::true_type&)
            {
                typedef hpx::util::logging::manipulator::detail ::
                    generic_holder<Formatter, formatter_base>
                        holder;
                add_formatter_impl(holder(fmt), std::false_type());
            }

            // generic manipulator
            template <class Formatter>
            void del_formatter_impl(Formatter fmt, const std::true_type&)
            {
                typedef hpx::util::logging::manipulator::detail ::
                    generic_holder<Formatter, formatter_base>
                        holder;
                del_formatter_impl(holder(fmt), std::false_type());
            }

            // generic manipulator
            template <class Destination>
            void add_destination_impl(Destination dest, const std::true_type&)
            {
                typedef hpx::util::logging::manipulator::detail ::
                    generic_holder<Destination, destination_base>
                        holder;
                add_destination_impl(holder(dest), std::false_type());
            }

            // generic manipulator
            template <class Destination>
            void del_destination_impl(Destination dest, const std::true_type&)
            {
                typedef hpx::util::logging::manipulator::detail ::
                    generic_holder<Destination, destination_base>
                        holder;
                del_destination_impl(holder(dest), std::false_type());
            }

        public:
            /**
        @brief Adds a formatter

        @param fmt The formatter
    */
            template <class Formatter>
            void add_formatter(Formatter fmt)
            {
                typedef hpx::util::logging::manipulator::is_generic is_generic;
                add_formatter_impl<Formatter>(
                    fmt, std::is_base_of<is_generic, Formatter>());
            }

            /**
        @brief Adds a formatter. Also, the second argument is the @ref
        hpx::util::logging::formatter::spacer_t "spacer" string

        @param fmt The formatter
        @param format_str The @ref hpx::util::logging::formatter::spacer_t
        "spacer" string
    */
            template <class Formatter>
            void add_formatter(Formatter fmt, const char* format_str)
            {
                add_formatter(spacer(fmt, format_str));
            }

            /**
        @brief Deletes a formatter

        @param fmt The formatter to delete
    */
            template <class Formatter>
            void del_formatter(Formatter fmt)
            {
                typedef hpx::util::logging::manipulator::is_generic is_generic;
                del_formatter_impl<Formatter>(
                    fmt, std::is_base_of<is_generic, Formatter>());
            }

            /**
        @brief Adds a destination
    */
            template <class Destination>
            void add_destination(Destination dest)
            {
                typedef hpx::util::logging::manipulator::is_generic is_generic;
                add_destination_impl<Destination>(
                    dest, std::is_base_of<is_generic, Destination>());
            }

            /**
        @brief Deletes a destination
    */
            template <class Destination>
            void del_destination(Destination dest)
            {
                typedef hpx::util::logging::manipulator::is_generic is_generic;
                del_destination_impl<Destination>(
                    dest, std::is_base_of<is_generic, Destination>());
            }

            /**
    returns the object that actually routes the message
    */
            router_type& router()
            {
                return m_router;
            }

            /**
    returns the object that actually routes the message
    */
            const router_type& router() const
            {
                return m_router;
            }

            /**
        does the actual write
    */
            void operator()(msg_type& msg) const
            {
                router().template write<apply_format_and_write_type>(msg);
            }

        private:
            formatter_array m_formatters;
            destination_array m_destinations;
            router_type m_router;
        };

    }    // namespace writer

}}}    // namespace hpx::util::logging

#endif
