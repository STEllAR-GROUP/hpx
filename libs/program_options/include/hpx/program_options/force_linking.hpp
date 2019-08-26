//  Copyright (c) 2019 The STE||AR GROUP
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PROGRAM_OPTIONS_FORCE_LINKING_HPP)
#define HPX_PROGRAM_OPTIONS_FORCE_LINKING_HPP

#include <hpx/program_options.hpp>

#include <functional>
#include <iosfwd>
#include <string>
#include <vector>

namespace hpx { namespace program_options
{
    using parse_environment1_type = basic_parsed_options<char> (*)(
        options_description const&, char const*);
    using parse_environment2_type = basic_parsed_options<char> (*)(
        options_description const&, std::function<std::string(std::string)> const&);
    using parse_environment3_type = basic_parsed_options<char> (*)(
        options_description const&, std::string const&);

    using parse_config_file_char1_type = basic_parsed_options<char> (*)(
        char const*, options_description const&, bool);
    using parse_config_file_char2_type = basic_parsed_options<char> (*)(
        std::basic_istream<char, struct std::char_traits<char>>&,
        options_description const&, bool);

    using parse_config_file_wchar1_type = basic_parsed_options<wchar_t> (*)(
        std::basic_istream<wchar_t, struct std::char_traits<wchar_t>>&,
        options_description const&, bool);

    using split_unix_type = std::vector<std::string> (*)(std::string const&,
        std::string const&, std::string const&, std::string const&);

    struct force_linking_helper
    {
        parse_environment1_type parse_environment1;
        parse_environment2_type parse_environment2;
        parse_environment3_type parse_environment3;

        parse_config_file_char1_type parse_config_file_char1;
        parse_config_file_char2_type parse_config_file_char2;

        parse_config_file_wchar1_type parse_config_file_wchar1;

        split_unix_type split_unix;
    };

    force_linking_helper& force_linking();
}}

#endif
