//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/error_code.hpp>
#include <hpx/util/regex_from_pattern.hpp>

#include <string>

namespace hpx { namespace util
{
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        inline std::string
        regex_from_character_set(std::string::const_iterator& it,
            std::string::const_iterator end, error_code& ec)
        {
            std::string::const_iterator start = it;
            std::string result(1, *it);  // copy '['
            if (*++it == '!') {
                result.append(1, '^');   // negated character set
            }
            else if (*it == ']') {
                HPX_THROWS_IF(ec, bad_parameter, "regex_from_character_set",
                    "Invalid pattern (empty character set) at: " +
                        std::string(start, end));
                return "";
            }
            else {
                result.append(1, *it);   // append this character
            }

            // copy while in character set
            while (++it != end) {
                result.append(1, *it);
                if (*it == ']')
                    break;
            }

            if (it == end || *it != ']') {
                HPX_THROWS_IF(ec, bad_parameter, "regex_from_character_set",
                    "Invalid pattern (missing closing ']') at: " +
                        std::string(start, end));
                return "";
            }

            return result;
        }
    }

    std::string regex_from_pattern(std::string const& pattern, error_code& ec)
    {
        std::string result;
        std::string::const_iterator end = pattern.end();
        for (std::string::const_iterator it = pattern.begin(); it != end; ++it)
        {
            char c = *it;
            switch (c) {
            case '*':
                result.append(".*");
                break;

            case '?':
                result.append(1, '.');
                break;

            case '[':
                {
                    std::string r =
                        detail::regex_from_character_set(it, end, ec);
                    if (ec) return "";
                    result.append(r);
                }
                break;

            case '\\':
                if (++it == end) {
                    HPX_THROWS_IF(ec, bad_parameter,
                        "regex_from_pattern",
                        "Invalid escape sequence at: " + pattern);
                    return "";
                }
                result.append(1, *it);
                break;

            // escape regex special characters
            case '+': HPX_FALLTHROUGH;
            case '.': HPX_FALLTHROUGH;
            case '(': HPX_FALLTHROUGH;
            case ')': HPX_FALLTHROUGH;
            case '{': HPX_FALLTHROUGH;
            case '}': HPX_FALLTHROUGH;
            case '^': HPX_FALLTHROUGH;
            case '$':
                result.append("\\");
                HPX_FALLTHROUGH;

            default:
                result.append(1, c);
                break;
            }
        }
        return result;
    }
}}
