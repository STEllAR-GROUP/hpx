//  Copyright Vladimir Prus 2004.
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/program_options/config.hpp>

#include <string>
#include <vector>

namespace hpx::program_options {

    /** Option found in input source.
        Contains a key and a value. The key, in turn, can be a string (name of
        an option), or an integer (position in input source) \-- in case no name
        is specified. The latter is only possible for command line.
        The template parameter specifies the type of char used for storing the
        option's value.
    */
    template <typename Char>
    class basic_option
    {
    public:
        basic_option() = default;

        basic_option(std::string const& xstring_key,
            std::vector<std::string> const& xvalue)
          : string_key(xstring_key)
          , value(xvalue)
        {
        }

        /** String key of this option. Intentionally independent of the template
            parameter. */
        std::string string_key;
        /** Position key of this option. All options without an explicit name are
            sequentially numbered starting from 0. If an option has explicit name,
            'position_key' is equal to -1. It is possible that both
            position_key and string_key is specified, in case name is implicitly
            added.
         */
        int position_key = -1;

        /** Option's value */
        std::vector<std::basic_string<Char>> value;
        /** The original unchanged tokens this option was
            created from. */
        std::vector<std::basic_string<Char>> original_tokens;
        /** True if option was not recognized. In that case,
            'string_key' and 'value' are results of purely
            syntactic parsing of source. The original tokens can be
            recovered from the "original_tokens" member.
        */
        bool unregistered = false;
        /** True if string_key has to be handled
            case insensitive.
        */
        bool case_insensitive = false;
    };

    using option = basic_option<char>;
    using woption = basic_option<wchar_t>;
}    // namespace hpx::program_options
