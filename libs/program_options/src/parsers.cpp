// Copyright Vladimir Prus 2002-2004.
//  SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/program_options/config.hpp>

#if !defined(HPX_PROGRAM_OPTIONS_HAVE_BOOST_PROGRAM_OPTIONS_COMPATIBILITY)
#include <hpx/program_options/detail/cmdline.hpp>
#include <hpx/program_options/detail/config_file.hpp>
#include <hpx/program_options/detail/convert.hpp>
#include <hpx/program_options/environment_iterator.hpp>
#include <hpx/program_options/options_description.hpp>
#include <hpx/program_options/parsers.hpp>
#include <hpx/program_options/positional_options.hpp>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>

#ifdef _WIN32
#include <stdlib.h>
#else
#include <unistd.h>
#endif

// The 'environ' should be declared in some cases. E.g. Linux man page says:
// (This variable must be declared in the user program, but is declared in
// the header file unistd.h in case the header files came from libc4 or libc5,
// and in case they came from glibc and _GNU_SOURCE was defined.)
// To be safe, declare it here.

// It appears that on Mac OS X the 'environ' variable is not
// available to dynamically linked libraries.
// See: http://article.gmane.org/gmane.comp.lib.boost.devel/103843
// See: http://lists.gnu.org/archive/html/bug-guile/2004-01/msg00013.html
#if defined(__APPLE__) && defined(__DYNAMIC__)
// The proper include for this is crt_externs.h, however it's not
// available on iOS. The right replacement is not known. See
// https://svn.boost.org/trac/boost/ticket/5053
extern "C" {
extern char*** _NSGetEnviron(void);
}
#define environ (*_NSGetEnviron())
#else
#if defined(__MWERKS__)
#include <crtl.h>
#else
#if !defined(_WIN32) || defined(__COMO_VERSION__)
extern char** environ;
#endif
#endif
#endif

using namespace std;

namespace hpx { namespace program_options {

    namespace {

        woption woption_from_option(const option& opt)
        {
            woption result;
            result.string_key = opt.string_key;
            result.position_key = opt.position_key;
            result.unregistered = opt.unregistered;

            std::transform(opt.value.begin(), opt.value.end(),
                back_inserter(result.value),
                std::bind(from_utf8, std::placeholders::_1));

            std::transform(opt.original_tokens.begin(),
                opt.original_tokens.end(),
                back_inserter(result.original_tokens),
                std::bind(from_utf8, std::placeholders::_1));
            return result;
        }
    }    // namespace

    basic_parsed_options<wchar_t>::basic_parsed_options(
        const parsed_options& po)
      : description(po.description)
      , utf8_encoded_options(po)
      , m_options_prefix(po.m_options_prefix)
    {
        for (const auto& option : po.options)
            options.push_back(woption_from_option(option));
    }

    template <class Char>
    basic_parsed_options<Char> parse_config_file(std::basic_istream<Char>& is,
        const options_description& desc, bool allow_unregistered)
    {
        set<string> allowed_options;

        const vector<shared_ptr<option_description>>& options = desc.options();
        for (const auto& option : options)
        {
            const option_description& d = *option;

            if (d.long_name().empty())
                throw error("abbreviated option names are not permitted in "
                            "options configuration files");

            allowed_options.insert(d.long_name());
        }

        // Parser return char strings
        parsed_options result(&desc);
        copy(detail::basic_config_file_iterator<Char>(
                 is, allowed_options, allow_unregistered),
            detail::basic_config_file_iterator<Char>(),
            back_inserter(result.options));
        // Convert char strings into desired type.
        return basic_parsed_options<Char>(result);
    }

    template HPX_EXPORT basic_parsed_options<char> parse_config_file(
        std::basic_istream<char>& is, const options_description& desc,
        bool allow_unregistered);

    template HPX_EXPORT basic_parsed_options<wchar_t> parse_config_file(
        std::basic_istream<wchar_t>& is, const options_description& desc,
        bool allow_unregistered);

    template <class Char>
    basic_parsed_options<Char> parse_config_file(const char* filename,
        const options_description& desc, bool allow_unregistered)
    {
        // Parser return char strings
        std::basic_ifstream<Char> strm(filename);
        if (!strm)
        {
            throw reading_file(filename);
        }

        basic_parsed_options<Char> result =
            parse_config_file(strm, desc, allow_unregistered);

        if (strm.bad())
        {
            throw reading_file(filename);
        }

        return result;
    }

    template HPX_EXPORT basic_parsed_options<char> parse_config_file(
        const char* filename, const options_description& desc,
        bool allow_unregistered);

    template HPX_EXPORT basic_parsed_options<wchar_t> parse_config_file(
        const char* filename, const options_description& desc,
        bool allow_unregistered);

    HPX_EXPORT parsed_options parse_environment(const options_description& desc,
        const std::function<std::string(std::string)>& name_mapper)
    {
        parsed_options result(&desc);

        for (environment_iterator i(environ), e; i != e; ++i)
        {
            string option_name = name_mapper(i->first);

            if (!option_name.empty())
            {
                option n;
                n.string_key = option_name;
                n.value.push_back(i->second);
                result.options.push_back(n);
            }
        }

        return result;
    }

    namespace detail {

        class prefix_name_mapper
        {
        public:
            prefix_name_mapper(const std::string& prefix)
              : prefix(prefix)
            {
            }

            std::string operator()(const std::string& s)
            {
                string result;
                if (s.find(prefix) == 0)
                {
                    for (string::size_type n = prefix.size(); n < s.size(); ++n)
                    {
                        // Intel-Win-7.1 does not understand
                        // push_back on string.
                        result += static_cast<char>(tolower(s[n]));
                    }
                }
                return result;
            }

        private:
            std::string prefix;
        };
    }    // namespace detail

    HPX_EXPORT parsed_options parse_environment(
        const options_description& desc, const std::string& prefix)
    {
        return parse_environment(desc, detail::prefix_name_mapper(prefix));
    }

    HPX_EXPORT parsed_options parse_environment(
        const options_description& desc, const char* prefix)
    {
        return parse_environment(desc, string(prefix));
    }

}}    // namespace hpx::program_options

#endif
