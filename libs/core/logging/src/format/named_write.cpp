// named_write.cpp

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

#include <hpx/logging/format/named_write.hpp>

#include <hpx/local/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/logging/format/destinations.hpp>
#include <hpx/logging/format/formatters.hpp>

#include <cstddef>
#include <memory>
#include <sstream>
#include <string>

namespace hpx { namespace util { namespace logging { namespace detail {

    static std::string unescape(std::string escaped)
    {
        typedef std::size_t size_type;
        size_type idx_start = 0;
        while (true)
        {
            size_type found = escaped.find("%%", idx_start);
            if (found != std::string::npos)
            {
                escaped.erase(
                    escaped.begin() + static_cast<std::ptrdiff_t>(found));
                ++idx_start;
            }
            else
                break;
        }
        return escaped;
    }

    void named_formatters::compute_write_steps()
    {
        typedef std::size_t size_type;

        write_steps.clear();
        std::string remaining = format_string;
        size_type start_search_idx = 0;
        while (!remaining.empty())
        {
            size_type idx = remaining.find_first_of("%|", start_search_idx);
            switch (idx != std::string::npos ? remaining[idx] : '\0')
            {
            case '|':
            {
                // up to here, this is a spacer string
                start_search_idx = 0;
                std::string spacer = detail::unescape(remaining.substr(0, idx));
                remaining.erase(0, idx + 1);

                formatter::manipulator* fmt = (formatter::manipulator*) -1;
                write_steps.push_back(write_step(spacer, fmt));
                break;
            }
            case '%':
            {
                // see if just escaped
                if ((idx < remaining.size() - 1) && remaining[idx + 1] == '%')
                {
                    // we found an escaped char
                    start_search_idx = idx + 2;
                    continue;
                }

                // up to here, this is a spacer string
                start_search_idx = 0;
                std::string spacer = detail::unescape(remaining.substr(0, idx));
                remaining.erase(0, idx + 1);
                // find end of formatter name
                idx = remaining.find('%');
                formatter::manipulator* fmt = nullptr;
                if (idx != std::string::npos)
                {
                    std::string name = remaining.substr(0, idx);
                    remaining.erase(0, idx + 1);
                    auto iter = find_named(formatters, name);
                    if (iter != formatters.end())
                        fmt = iter->value.get();
                }
                // note: fmt could be null, in case
                write_steps.push_back(write_step(spacer, fmt));
                break;
            }
            case '\0':
            {
                // last part
                write_steps.push_back(
                    write_step(detail::unescape(remaining), nullptr));
                remaining.clear();
                break;
            }
            }
        }
    }

    void named_destinations::compute_write_steps()
    {
        write_steps.clear();

        std::istringstream in(format_string);
        std::string word;
        while (in >> word)
        {
            if (word[0] == '+')
                word.erase(word.begin());
            else if (word[0] == '-')
                // ignore this word
                continue;

            auto iter = find_named(destinations, word);
            if (iter != destinations.cend())
                write_steps.push_back(iter->value.get());
        }
    }

    namespace {
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

        template <typename Named, typename ParserType>
        void configure(
            Named& named, std::string const& format, ParserType parser)
        {
            // need to parse string
            bool parsing_params = false;
            std::string params;
            std::string stripped_str;
            for (char c : format)
            {
                if ((c == '(') && !parsing_params)
                {
                    if (parser.has_manipulator_name())
                    {
                        parsing_params = true;
                        params.clear();
                    }
                    else
                    {
                        stripped_str += c;
                        parser.add(c);
                    }
                }
                else if (c == ')' && parsing_params)
                {
                    HPX_ASSERT(parser.has_manipulator_name());
                    named.configure(parser.get_manipulator_name(), params);
                    parser.clear();
                    parsing_params = false;
                }
                else
                {
                    if (parsing_params)
                        params += c;
                    else
                    {
                        stripped_str += c;
                        parser.add(c);
                    }
                }
            }
            named.string(stripped_str);
        }
    }    // namespace

}}}}    // namespace hpx::util::logging::detail

namespace hpx { namespace util { namespace logging { namespace writer {

    named_write::named_write()
    {
        set_formatter<formatter::idx>("idx");
        set_formatter<formatter::high_precision_time>("time", "$hh:$mm:$ss");
        set_formatter<formatter::thread_id>("thread_id");

        set_destination<destination::file>("file", "");
        set_destination<destination::cout>("cout");
        set_destination<destination::cerr>("cerr");
        set_destination<destination::dbg_window>("debug");
    }

    void named_write::configure_formatter(std::string const& format)
    {
        detail::configure(m_format, format, detail::parse_formatter{});
    }

    void named_write::configure_destination(std::string const& format)
    {
        detail::configure(m_destination, format, detail::parse_destination{});
    }

}}}}    // namespace hpx::util::logging::writer
