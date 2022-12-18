//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <cctype>
#include <string>

namespace hpx::string_util {

    namespace detail {

        template <typename Char, typename Traits, typename Allocator>
        struct is_any_of_pred
        {
            using string_type = std::basic_string<Char, Traits, Allocator>;

            bool operator()(int c) const noexcept
            {
                return chars.find(c) != string_type::npos;
            }

            string_type chars;
        };
    }    // namespace detail

    template <typename Char, typename Traits, typename Allocator>
    detail::is_any_of_pred<Char, Traits, Allocator> is_any_of(
        std::basic_string<Char, Traits, Allocator> const& chars)
    {
        return detail::is_any_of_pred<Char, Traits, Allocator>{chars};
    }

    template <typename Char, typename Traits, typename Allocator>
    detail::is_any_of_pred<Char, Traits, Allocator> is_any_of(
        std::basic_string<Char, Traits, Allocator>&& chars)
    {
        return detail::is_any_of_pred<Char, Traits, Allocator>{HPX_MOVE(chars)};
    }

    inline auto is_any_of(char const* chars)
    {
        return detail::is_any_of_pred<char, std::char_traits<char>,
            std::allocator<char>>{std::string{chars}};
    }

    struct is_space
    {
        bool operator()(int c) const noexcept
        {
            return std::isspace(c);
        }
    };
}    // namespace hpx::string_util
