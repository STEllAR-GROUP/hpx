// Copyright Vladimir Prus 2004.
//  SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/program_options/config.hpp>

#if defined(HPX_PROGRAM_OPTIONS_HAVE_BOOST_PROGRAM_OPTIONS_COMPATIBILITY)
// hpxinspect:nodeprecatedinclude:boost/program_options/detail/parsers.hpp

#include <boost/program_options/detail/parsers.hpp>

namespace hpx { namespace program_options {

    using boost::program_options::collect_unrecognized;
    using boost::program_options::parse_command_line;

}}    // namespace hpx::program_options

#else

#include <hpx/program_options/detail/convert.hpp>
#include <hpx/program_options/parsers.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

namespace hpx { namespace program_options {

    template <class Char>
    basic_command_line_parser<Char>::basic_command_line_parser(
        const std::vector<std::basic_string<Char>>& xargs)
      : detail::cmdline(to_internal(xargs))
    {
    }

    template <class Char>
    basic_command_line_parser<Char>::basic_command_line_parser(
        int argc, const Char* const argv[])
      : detail::cmdline(to_internal(
            std::vector<std::basic_string<Char>>(argv + 1, argv + argc)))
      , m_desc()
    {
    }

    template <class Char>
    basic_command_line_parser<Char>& basic_command_line_parser<Char>::options(
        const options_description& desc)
    {
        detail::cmdline::set_options_description(desc);
        m_desc = &desc;
        return *this;
    }

    template <class Char>
    basic_command_line_parser<Char>&
    basic_command_line_parser<Char>::positional(
        const positional_options_description& desc)
    {
        detail::cmdline::set_positional_options(desc);
        return *this;
    }

    template <class Char>
    basic_command_line_parser<Char>& basic_command_line_parser<Char>::style(
        int xstyle)
    {
        detail::cmdline::style(xstyle);
        return *this;
    }

    template <class Char>
    basic_command_line_parser<Char>&
    basic_command_line_parser<Char>::extra_parser(ext_parser ext)
    {
        detail::cmdline::set_additional_parser(ext);
        return *this;
    }

    template <class Char>
    basic_command_line_parser<Char>&
    basic_command_line_parser<Char>::allow_unregistered()
    {
        detail::cmdline::allow_unregistered();
        return *this;
    }

    template <class Char>
    basic_command_line_parser<Char>&
    basic_command_line_parser<Char>::extra_style_parser(style_parser s)
    {
        detail::cmdline::extra_style_parser(s);
        return *this;
    }

    template <class Char>
    basic_parsed_options<Char> basic_command_line_parser<Char>::run()
    {
        // save the canonical prefixes which were used by this cmdline parser
        //    eventually inside the parsed results
        //    This will be handy to format recognizable options
        //    for diagnostic messages if everything blows up much later on
        parsed_options result(
            m_desc, detail::cmdline::get_canonical_option_prefix());
        result.options = detail::cmdline::run();

        // Presence of parsed_options -> wparsed_options conversion
        // does the trick.
        return basic_parsed_options<Char>(result);
    }

    template <class Char>
    basic_parsed_options<Char> parse_command_line(int argc,
        const Char* const argv[], const options_description& desc, int style,
        std::function<std::pair<std::string, std::string>(const std::string&)>
            ext)
    {
        return basic_command_line_parser<Char>(argc, argv)
            .options(desc)
            .style(style)
            .extra_parser(ext)
            .run();
    }

    template <class Char>
    std::vector<std::basic_string<Char>> collect_unrecognized(
        const std::vector<basic_option<Char>>& options,
        enum collect_unrecognized_mode mode)
    {
        std::vector<std::basic_string<Char>> result;
        for (std::size_t i = 0; i < options.size(); ++i)
        {
            if (options[i].unregistered ||
                (mode == include_positional && options[i].position_key != -1))
            {
                std::copy(options[i].original_tokens.begin(),
                    options[i].original_tokens.end(),
                    std::back_inserter(result));
            }
        }
        return result;
    }

}}    // namespace hpx::program_options

#endif
