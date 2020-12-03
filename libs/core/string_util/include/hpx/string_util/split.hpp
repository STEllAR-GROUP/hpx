//  Copyright (c) 2020 ETH Zurich
//  Copyright (c) 2002-2006 Pavol Droba
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <string>
#include <utility>

namespace hpx { namespace string_util {
    namespace detail {
        template <typename It, typename CharT, typename Traits,
            typename Allocator>
        std::basic_string<CharT, Traits, Allocator> substr(
            std::basic_string<CharT, Traits, Allocator> const& s,
            It const& first, It const& last)
        {
            std::size_t const pos = std::distance(std::begin(s), first);
            std::size_t const count = std::distance(first, last);
            return s.substr(pos, count);
        }
    }    // namespace detail

    enum class token_compress_mode
    {
        off,
        on
    };

    template <typename Container, typename Predicate, typename CharT,
        typename Traits, typename Allocator>
    void split(Container& container,
        std::basic_string<CharT, Traits, Allocator> const& str,
        Predicate&& pred,
        token_compress_mode compress_mode = token_compress_mode::off)
    {
        container.clear();

        auto token_begin = std::begin(str);
        auto token_end = std::end(str);

        do
        {
            token_end = std::find_if(token_begin, std::end(str), pred);

            container.push_back(detail::substr(str, token_begin, token_end));

            if (token_end != std::end(str))
            {
                token_begin = token_end + 1;
            }

            if (compress_mode == token_compress_mode::on)
            {
                // Skip contiguous separators
                while (token_begin != std::end(str) && pred(int(*token_begin)))
                {
                    ++token_begin;
                }
            }
        } while (token_end != std::end(str));
    }

    template <typename Container, typename Predicate>
    void split(Container& container, char const* str, Predicate&& pred,
        token_compress_mode compress_mode = token_compress_mode::off)
    {
        split(container, std::string{str}, std::forward<Predicate>(pred),
            compress_mode);
    }
}}    // namespace hpx::string_util
