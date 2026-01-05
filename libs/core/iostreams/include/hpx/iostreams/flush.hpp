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
#include <hpx/iostreams/detail/dispatch.hpp>
#include <hpx/iostreams/operations_fwd.hpp>
#include <hpx/iostreams/traits.hpp>
#include <hpx/modules/type_support.hpp>

#include <type_traits>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx::iostreams {

    namespace detail {

        HPX_CXX_CORE_EXPORT template <typename T>
        struct flush_device_impl;

        HPX_CXX_CORE_EXPORT template <typename T>
        struct flush_filter_impl;
    }    // namespace detail

    HPX_CXX_CORE_EXPORT template <typename T>
    bool flush(T& t)
    {
        return detail::flush_device_impl<T>::flush(util::unwrap_ref(t));
    }

    HPX_CXX_CORE_EXPORT template <typename T, typename Sink>
    bool flush(T& t, Sink& snk)
    {
        return detail::flush_filter_impl<T>::flush(util::unwrap_ref(t), snk);
    }

    namespace detail {

        //------------------Definition of flush_device_impl---------------------------//
        HPX_CXX_CORE_EXPORT template <typename T>
        struct flush_device_impl
          : std::conditional_t<is_custom_v<T>, operations<T>,
                flush_device_impl<dispatch_t<T, ostream_tag, streambuf_tag,
                    flushable_tag, any_tag>>>
        {
        };

        template <>
        struct flush_device_impl<ostream_tag>
        {
            template <typename T>
            static bool flush(T& t)
            {
                return t.rdbuf()->pubsync() == 0;
            }
        };

        template <>
        struct flush_device_impl<streambuf_tag>
        {
            template <typename T>
            static bool flush(T& t)
            {
                return t.pubsync() == 0;
            }
        };

        template <>
        struct flush_device_impl<flushable_tag>
        {
            template <typename T>
            static bool flush(T& t)
            {
                return t.flush();
            }
        };

        template <>
        struct flush_device_impl<any_tag>
        {
            template <typename T>
            static bool flush(T&)
            {
                return true;
            }
        };

        //------------------Definition of flush_filter_impl---------------------------//
        HPX_CXX_CORE_EXPORT template <typename T>
        struct flush_filter_impl
          : std::conditional_t<is_custom_v<T>, operations<T>,
                flush_filter_impl<dispatch_t<T, flushable_tag, any_tag>>>
        {
        };

        template <>
        struct flush_filter_impl<flushable_tag>
        {
            template <typename T, typename Sink>
            static bool flush(T& t, Sink& snk)
            {
                return t.flush(snk);
            }
        };

        template <>
        struct flush_filter_impl<any_tag>
        {
            template <typename T, typename Sink>
            static bool flush(T&, Sink&)
            {
                return false;
            }
        };
    }    // End namespace detail.
}    // namespace hpx::iostreams

#include <hpx/config/warnings_suffix.hpp>
