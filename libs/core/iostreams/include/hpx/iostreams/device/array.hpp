//  Copyright (c) 2026 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// (C) Copyright 2008 CodeRage, LLC (turkanis at coderage dot com)
// (C) Copyright 2004-2007 Jonathan Turkanis

// See http://www.boost.org/libs/iostreams for documentation.

#pragma once

#include <hpx/config.hpp>
#include <hpx/iostreams/categories.hpp>

#include <cstddef>
#include <span>
#include <type_traits>

namespace hpx::iostreams {

    namespace detail {

        HPX_CXX_CORE_EXPORT template <typename Mode, typename Ch>
        class array_adapter : private std::span<Ch>
        {
        public:
            using char_type = Ch;
            using pair_type = std::span<Ch>;

            struct category
              : Mode
              , device_tag
              , direct_tag
            {
            };

            constexpr array_adapter(char_type* begin, char_type* end) noexcept
              : pair_type(begin, end)
            {
            }

            constexpr array_adapter(
                char_type* begin, std::size_t length) noexcept
              : pair_type(begin, length)
            {
            }

            constexpr array_adapter(
                char_type const* begin, char_type const* end) noexcept
                requires(!std::is_convertible_v<Mode, output>)
              : pair_type(
                    const_cast<char_type*>(begin), const_cast<char_type*>(end))
            {
            }

            constexpr array_adapter(
                char_type const* begin, std::size_t length) noexcept
                requires(!std::is_convertible_v<Mode, output>)
              : pair_type(const_cast<char_type*>(begin), length)
            {
            }

            template <int N>
            constexpr explicit array_adapter(char_type (&ar)[N]) noexcept
              : pair_type(ar)
            {
            }

            constexpr pair_type input_sequence() noexcept
                requires(std::is_convertible_v<Mode, input>)
            {
                return *this;
            }

            constexpr pair_type output_sequence() noexcept
                requires(std::is_convertible_v<Mode, output>)
            {
                return *this;
            }
        };
    }    // namespace detail

    HPX_CXX_CORE_EXPORT template <typename Ch>
    using array_source = detail::array_adapter<input_seekable, Ch>;

    HPX_CXX_CORE_EXPORT template <typename Ch>
    using array_sink = detail::array_adapter<output_seekable, Ch>;

    HPX_CXX_CORE_EXPORT template <typename Ch>
    using array = detail::array_adapter<seekable, Ch>;
}    // namespace hpx::iostreams
