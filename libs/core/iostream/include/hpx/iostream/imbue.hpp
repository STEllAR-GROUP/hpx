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
#include <hpx/iostream/detail/dispatch.hpp>
#include <hpx/iostream/operations_fwd.hpp>
#include <hpx/modules/type_support.hpp>

#include <type_traits>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx::iostream {

    namespace detail {

        // Implementation templates for simulated tag dispatch.
        HPX_CXX_CORE_EXPORT template <typename T>
        struct imbue_impl;
    }    // namespace detail

    HPX_CXX_CORE_EXPORT template <typename T, typename Locale>
    void imbue(T& t, Locale const& loc)
    {
        detail::imbue_impl<T>::imbue(util::unwrap_ref(t), loc);
    }

    namespace detail {

        //------------------Definition of imbue_impl----------------------------------//
        HPX_CXX_CORE_EXPORT template <typename T>
        struct imbue_impl
          : std::conditional_t<is_custom_v<T>, operations<T>,
                imbue_impl<
                    dispatch_t<T, streambuf_tag, localizable_tag, any_tag>>>
        {
        };

        template <>
        struct imbue_impl<any_tag>
        {
            template <typename T, typename Locale>
            static void imbue(T&, Locale const&)
            {
            }
        };

        template <>
        struct imbue_impl<streambuf_tag>
        {
            template <typename T, typename Locale>
            static void imbue(T& t, Locale const& loc)
            {
                t.pubimbue(loc);
            }
        };

        template <>
        struct imbue_impl<localizable_tag>
        {
            template <typename T, typename Locale>
            static void imbue(T& t, Locale const& loc)
            {
                t.imbue(loc);
            }
        };
    }    // namespace detail
}    // namespace hpx::iostream

#include <hpx/config/warnings_suffix.hpp>
