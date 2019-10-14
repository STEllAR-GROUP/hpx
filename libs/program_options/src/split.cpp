// Copyright Sascha Ochsenknecht 2009.
//  SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/program_options/config.hpp>

#if !defined(HPX_PROGRAM_OPTIONS_HAVE_BOOST_PROGRAM_OPTIONS_COMPATIBILITY)
#include <hpx/program_options/parsers.hpp>

#include <boost/tokenizer.hpp>

#include <string>
#include <vector>

namespace hpx { namespace program_options { namespace detail {

    template <class Char>
    std::vector<std::basic_string<Char>> split_unix(
        const std::basic_string<Char>& cmdline,
        const std::basic_string<Char>& seperator,
        const std::basic_string<Char>& quote,
        const std::basic_string<Char>& escape)
    {
        using tokenizerT = boost::tokenizer<boost::escaped_list_separator<Char>,
            typename std::basic_string<Char>::const_iterator,
            std::basic_string<Char>>;

        tokenizerT tok(cmdline.begin(), cmdline.end(),
            boost::escaped_list_separator<Char>(escape, seperator, quote));

        std::vector<std::basic_string<Char>> result;
        for (typename tokenizerT::iterator cur_token(tok.begin()),
             end_token(tok.end());
             cur_token != end_token; ++cur_token)
        {
            if (!cur_token->empty())
                result.push_back(*cur_token);
        }
        return result;
    }

}}}    // namespace hpx::program_options::detail

namespace hpx { namespace program_options {

    // Take a command line string and splits in into tokens, according
    // to the given collection of separators chars.
    HPX_EXPORT std::vector<std::string> split_unix(const std::string& cmdline,
        const std::string& seperator, const std::string& quote,
        const std::string& escape)
    {
        return detail::split_unix<char>(cmdline, seperator, quote, escape);
    }

    HPX_EXPORT std::vector<std::wstring> split_unix(const std::wstring& cmdline,
        const std::wstring& seperator, const std::wstring& quote,
        const std::wstring& escape)
    {
        return detail::split_unix<wchar_t>(cmdline, seperator, quote, escape);
    }

}}    // namespace hpx::program_options

#endif
