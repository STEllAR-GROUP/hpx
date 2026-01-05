//  Copyright (c) 2026 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// (C) Copyright 2008 CodeRage, LLC (turkanis at coderage dot com)
// (C) Copyright 2003-2007 Jonathan Turkanis

// See http://www.boost.org/libs/iostreams for documentation.

#pragma once

#include <hpx/config.hpp>
#include <hpx/iostreams/config/defines.hpp>
#include <hpx/iostreams/categories.hpp>
#include <hpx/iostreams/positioning.hpp>

#include <iosfwd>
#include <type_traits>

namespace hpx::iostreams {

    //--------------Definitions of helper templates for device concepts-----------//
    template <typename Mode, typename Ch = char>
    struct device
    {
        using char_type = Ch;

        struct category
          : Mode
          , device_tag
          , closable_tag
          , localizable_tag
        {
        };

        static constexpr void close() noexcept
        {
            static_assert(!std::is_convertible_v<Mode, detail::two_sequence>);
        }

        static constexpr void close(std::ios_base::openmode) noexcept
        {
            static_assert(std::is_convertible_v<Mode, detail::two_sequence>);
        }

        template <typename Locale>
        static constexpr void imbue(Locale const&) noexcept
        {
        }
    };

    using source = device<input>;
    using sink = device<output>;

#if defined(HPX_IOSTREAMS_HAVE_WIDE_STREAMS)
    template <typename Mode, typename Ch = wchar_t>
    struct wdevice : device<Mode, Ch>
    {
    };

    using wsource = wdevice<input>;
    using wsink = wdevice<output>;
#endif

    //--------------Definitions of helper templates for simple filter concepts----//
    template <typename Mode, typename Ch = char>
    struct filter
    {
        using char_type = Ch;

        struct category
          : Mode
          , filter_tag
          , closable_tag
          , localizable_tag
        {
        };

        template <typename Device>
        static constexpr void close(Device&) noexcept
        {
            static_assert(!std::is_convertible_v<Mode, detail::two_sequence>);
            static_assert(!std::is_convertible_v<Mode, dual_use>);
        }

        template <typename Device>
        static constexpr void close(Device&, std::ios_base::openmode) noexcept
        {
            static_assert(std::is_convertible_v<Mode, detail::two_sequence> ||
                std::is_convertible_v<Mode, dual_use>);
        }

        template <typename Locale>
        static constexpr void imbue(Locale const&) noexcept
        {
        }
    };

    using input_filter = filter<input>;
    using output_filter = filter<output>;
    using seekable_filter = filter<seekable>;
    using dual_use_filter = filter<dual_use>;

#if defined(HPX_IOSTREAMS_HAVE_WIDE_STREAMS)
    template <typename Mode, typename Ch = wchar_t>
    struct wfilter : filter<Mode, Ch>
    {
    };

    using input_wfilter = wfilter<input>;
    using output_wfilter = wfilter<output>;
    using seekable_wfilter = wfilter<seekable>;
    using dual_use_wfilter = wfilter<dual_use>;
#endif

    //------Definitions of helper templates for multi-character filter concepts----//
    template <typename Mode, typename Ch = char>
    struct multichar_filter : filter<Mode, Ch>
    {
        struct category
          : filter<Mode, Ch>::category
          , multichar_tag
        {
        };
    };

    using multichar_input_filter = multichar_filter<input>;
    using multichar_output_filter = multichar_filter<output>;
    using multichar_dual_use_filter = multichar_filter<dual_use>;

#if defined(HPX_IOSTREAMS_HAVE_WIDE_STREAMS)
    template <typename Mode, typename Ch = wchar_t>
    struct multichar_wfilter : multichar_filter<Mode, Ch>
    {
    };

    using multichar_input_wfilter = multichar_wfilter<input>;
    using multichar_output_wfilter = multichar_wfilter<output>;
    using multichar_dual_use_wfilter = multichar_wfilter<dual_use>;
#endif
}    // namespace hpx::iostreams
