// named_write.hpp

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

#ifndef JT28092007_named_writer_HPP_DEFINED
#define JT28092007_named_writer_HPP_DEFINED

#include <hpx/assertion.hpp>
#include <hpx/logging/format.hpp>
#include <cstddef>
#include <string>

// all destinations
#include <hpx/logging/format/destination/file.hpp>
#include <hpx/logging/format/destination/named.hpp>

// all formats
#include <hpx/logging/format/formatter/high_precision_time.hpp>
#include <hpx/logging/format/formatter/named_spacer.hpp>
#include <hpx/logging/format/formatter/thread_id.hpp>

namespace hpx { namespace util { namespace logging { namespace detail {
    typedef formatter::high_precision_time formatter_time_type;
}}}}    // namespace hpx::util::logging::detail

namespace hpx { namespace util { namespace logging { namespace writer {

    /**
@brief Composed of a named formatter and a named destinations.
Thus, you can specify the formatting and destinations as strings

@code
#include <hpx/logging/format/named_write.hpp>
@endcode


Contains a very easy interface for using @ref manipulator "formatters and destinations":
- at construction, specify 2 params: the %formatter string and the destinations string

Setting the @ref manipulator "formatters and destinations" to
write to is extremely simple:

@code
// Set the formatters (first param) and destinatins (second step) in one step
g_l()->writer().write("%time%($hh:$mm.$ss.$mili) [%idx%] |\n",
"cout file(out.txt) debug");

// set the formatter(s)
g_l()->writer().format("%time%($hh:$mm.$ss.$mili) [%idx%] |\n");

// set the destination(s)
g_l()->writer().destination("cout file(out.txt) debug");
@endcode


@section format_string_syntax The syntax of the format string

- The format string specifies how the message is to be logged
- Every formatter is escaped using <tt>%</tt><em>fmt</em><tt>%</tt>
  - Available formatters:
    - <tt>"%idx%"</tt> - writes the index of the message (formatter::idx)
    - <tt>"%time%"</tt> - writes the time (formatter::high_precision_time)
    - <tt>"%thread_id%"</tt> - writes the thread id (formatter::thread_id)
    - if you want to write @c "%", double it, like this: @c "%%"
- @c "|" is used to specify the original message. What is before it,
is prepended to the message, what is after, is appended to the message
- If a formatter is configurable, append @em (params) to it
  - For now, only @c "%time%" is configurable. For instance,
  @c "%time%($hh:$mm.$ss.$mili)" writes time like @c "21:14.24.674"

Example:
@code
"%time%($hh:$mm.$ss.$mili) [%idx%] |\n"
@endcode

The output can look like:

@code
21:03.17.243 [1] this is so cool
21:03.17.243 [2] first error
21:03.17.243 [3] hello, world
@endcode


@section dest_string_syntax The syntax of the destinations string

- The syntax of the destination string specifies where the message is to be logged
  - Every destination is specified by name
  - Separate destinations by space (' ')
- Available destinations
  - <tt>"cout"</tt> - writes to std::cout (destination::cout)
  - <tt>"cerr"</tt> - writes to std::cerr (destination::cerr)
  - <tt>"debug"</tt> - writes to the debug window: OutputDebugString in Windows,
  console on Linux (destination::dbg_window)
  - <tt>"file"</tt> - writes to a file (destination::file)
- If a destination is configurable, append @em (params) to it
  - Right now, @c "file" is configurable
    - Append <tt>(</tt><em>filename</em><tt>)</tt> to them to specify the file name.
    Example: @c "file(out.txt)" will write to the out.txt file

Examples:
- <tt>"file(out.txt) cout"</tt> - will write to a file called out.txt and to cout
- <tt>"cout debug"</tt> - will write to cout and debug window (see above)

@note
If you want to output to 2 files, don't use "file(one.txt) file(two.txt)".
This will just configure "file" twice, ending up with writing only to "two.txt" file.

@param format_write_ the underlying format writer


*/
    struct named_write
    {
        named_write()
        {
            m_writer.add_formatter(m_format_before);
            m_writer.add_destination(m_destination);

            init();
        }

        /** @brief sets the format string: what should be before,
    and what after the original message, separated by "|"

    Example: \n
    "[%idx%] |\n" - this writes "[%idx%] " before the message,
    and "\n" after the message

    If "|" is not present, the whole message is prepended to the message
    */
        void format(std::string const& format_str)
        {
            m_format_str = format_str;

            std::size_t idx = format_str.find('|');
            std::string before, after;
            if (idx != std::string::npos)
            {
                before = format_str.substr(0, idx);
                after = format_str.substr(idx + 1);
            }
            else
                before = format_str;

            format(before, after);
        };

        /** @brief sets the format strings (what should be before,
    and what after the original message)
    */
        void format(std::string const& format_before_str,
            std::string const& format_after_str)
        {
            m_format_before_str = format_before_str;

            set_and_configure(
                m_format_before, format_before_str, parse_formatter());
        };

        /** @brief sets the destinations string - where should logged messages
         * be outputted
    */
        void destination(std::string const& destination_str)
        {
            m_destination_str = destination_str;
            set_and_configure(
                m_destination, destination_str, parse_destination());
        }

        /** @brief Specifies the formats and destinations in one step
    */
        void write(
            std::string const& format_str, std::string const& destination_str)
        {
            format(format_str);
            destination(destination_str);
        }

        std::string const& format() const
        {
            return m_format_str;
        }
        std::string const& destination() const
        {
            return m_destination_str;
        }

        void operator()(message& msg) const
        {
            m_writer(msg);
        }

        /** @brief Replaces a destination from the named destination.

    You can use this, for instance, when you want to share a
    destination between multiple named writers.
    */
        template <class destination>
        void replace_destination(std::string const& name, destination d)
        {
            m_destination.del(name);
            m_destination.add(name, d);
        }

        /** @brief Replaces a formatter from the named formatter.

    You can use this, for instance, when you want to share
    a formatter between multiple named writers.
    */
        template <class formatter>
        void replace_formatter(std::string const& name, formatter d)
        {
            if (m_format_before_str.find(name) != std::string::npos)
            {
                m_format_before.del(name);
            }
            m_format_before.add(name, d);
        }

        template <class formatter>
        void add_formatter(formatter fmt)
        {
            m_writer.add_formatter(fmt);
        }

        template <class destination>
        void add_destination(std::string const& name, destination d)
        {
            m_destination.add(name, d);
        }

    private:
        struct parse_destination
        {
            bool has_manipulator_name() const
            {
                return !m_manipulator.empty();
            }
            std::string get_manipulator_name() const
            {
                HPX_ASSERT(has_manipulator_name());
                if (m_manipulator[0] == '-' || m_manipulator[0] == '+')
                    // + or - -> turning on or off a destination
                    return m_manipulator.substr(1);
                else
                    return m_manipulator;
            }
            void clear()
            {
                m_manipulator.clear();
            }

            void add(char c)
            {
                // destination always follows ' '
                if (c == ' ')
                    clear();
                else
                    m_manipulator += c;
            }

        private:
            std::string m_manipulator;
        };

        struct parse_formatter
        {
            // formatter starts and ends with %
            bool has_manipulator_name() const
            {
                if (m_manipulator.empty())
                    return false;
                if (m_manipulator.size() > 1)
                    if (m_manipulator[0] == '%' &&
                        (*m_manipulator.rbegin() == '%'))
                        return true;

                return false;
            }

            std::string get_manipulator_name() const
            {
                HPX_ASSERT(has_manipulator_name());
                // ignore starting and ending %
                return m_manipulator.substr(1, m_manipulator.size() - 2);
            }
            void clear()
            {
                m_manipulator.clear();
            }

            void add(char c)
            {
                if (has_manipulator_name())
                    // was a manipulator until now
                    clear();

                if (c == '%')
                {
                    m_manipulator += c;
                    if (!has_manipulator_name())
                        // it could be the start of a formatter
                        m_manipulator = '%';
                }
                else if (m_manipulator.empty())
                // NOLINTNEXTLINE(bugprone-branch-clone)
                {
                    ;    // ignore this char - not from a manipulator
                }
                else if (m_manipulator[0] == '%')
                {
                    m_manipulator += c;
                }
                else
                {
                    // manipulator should always start with %
                    HPX_ASSERT(false);
                }
            }

        private:
            std::string m_manipulator;
        };

        template <class manipulator, class parser_type>
        void set_and_configure(
            manipulator& manip, std::string const& name, parser_type parser)
        {
            // need to parse string
            bool parsing_params = false;
            std::string params;
            std::string stripped_str;
            for (std::string::const_iterator b = name.begin(), e = name.end();
                 b != e; ++b)
            {
                if ((*b == '(') && !parsing_params)
                {
                    if (parser.has_manipulator_name())
                    {
                        parsing_params = true;
                        params.clear();
                    }
                    else
                    {
                        stripped_str += *b;
                        parser.add(*b);
                    }
                }
                else if ((*b == ')') && parsing_params)
                {
                    HPX_ASSERT(parser.has_manipulator_name());
                    manip.configure_inner(
                        parser.get_manipulator_name(), params);
                    parser.clear();
                    parsing_params = false;
                }
                else
                {
                    if (parsing_params)
                        params += *b;
                    else
                    {
                        stripped_str += *b;
                        parser.add(*b);
                    }
                }
            }
            manip.string(stripped_str);
        }

    private:
        void init()
        {
            m_format_before.add("idx", formatter::idx())
                .add("time",
                    ::hpx::util::logging::detail::formatter_time_type(
                        "$hh:$mm:$ss"))
                .add("thread_id", formatter::thread_id());

            m_destination.add("file", destination::file(""))
                .add("cout", destination::cout())
                .add("cerr", destination::cerr())
                .add("debug", destination::dbg_window());
        }

    private:
        formatter::named_spacer_t m_format_before;
        destination::named m_destination;
        format_write m_writer;

        std::string m_format_str;
        std::string m_format_before_str;
        std::string m_destination_str;
    };

}}}}    // namespace hpx::util::logging::writer

#endif
