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
#include <hpx/iostream/categories.hpp>
#include <hpx/iostream/detail/dispatch.hpp>
#include <hpx/iostream/operations_fwd.hpp>
#include <hpx/iostream/positioning.hpp>
#include <hpx/modules/type_support.hpp>

#include <iosfwd>
#include <type_traits>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx::iostream {

    namespace detail {

        HPX_CXX_CORE_EXPORT template <typename T>
        struct seek_device_impl;

        HPX_CXX_CORE_EXPORT template <typename T>
        struct seek_filter_impl;
    }    // namespace detail

    HPX_CXX_CORE_EXPORT template <typename T>
    std::streampos seek(T& t, stream_offset off, std::ios_base::seekdir way,
        std::ios_base::openmode which = std::ios_base::in | std::ios_base::out)
    {
        return detail::seek_device_impl<T>::seek(
            util::unwrap_ref(t), off, way, which);
    }

    HPX_CXX_CORE_EXPORT template <typename T, typename Device>
    std::streampos seek(T& t, Device& dev, stream_offset off,
        std::ios_base::seekdir way,
        std::ios_base::openmode which = std::ios_base::in | std::ios_base::out)
    {
        return detail::seek_filter_impl<T>::seek(
            util::unwrap_ref(t), dev, off, way, which);
    }

    namespace detail {

        //------------------Definition of seek_device_impl----------------------------//
        HPX_CXX_CORE_EXPORT template <typename T>
        struct seek_device_impl
          : std::conditional_t<is_custom_v<T>, operations<T>,
                seek_device_impl<dispatch_t<T, iostream_tag, istream_tag,
                    ostream_tag, streambuf_tag, two_head, any_tag>>>
        {
        };

        HPX_CXX_CORE_EXPORT struct seek_impl_basic_ios
        {
            template <typename T>
            static std::streampos seek(T& t, stream_offset off,
                std::ios_base::seekdir way, std::ios_base::openmode which)
            {
                return t.rdbuf()->pubseekoff(off, way, which);
            }
        };

        template <>
        struct seek_device_impl<iostream_tag> : seek_impl_basic_ios
        {
        };

        template <>
        struct seek_device_impl<istream_tag> : seek_impl_basic_ios
        {
        };

        template <>
        struct seek_device_impl<ostream_tag> : seek_impl_basic_ios
        {
        };

        template <>
        struct seek_device_impl<streambuf_tag>
        {
            template <typename T>
            static std::streampos seek(T& t, stream_offset off,
                std::ios_base::seekdir way, std::ios_base::openmode which)
            {
                return t.pubseekoff(off, way, which);
            }
        };

        template <>
        struct seek_device_impl<two_head>
        {
            template <typename T>
            static std::streampos seek(T& t, stream_offset off,
                std::ios_base::seekdir way, std::ios_base::openmode which)
            {
                return t.seek(off, way, which);
            }
        };

        template <>
        struct seek_device_impl<any_tag>
        {
            template <typename T>
            static std::streampos seek(T& t, stream_offset off,
                std::ios_base::seekdir way, std::ios_base::openmode)
            {
                return t.seek(off, way);
            }
        };

        //------------------Definition of seek_filter_impl----------------------------//
        HPX_CXX_CORE_EXPORT template <typename T>
        struct seek_filter_impl
          : std::conditional_t<is_custom_v<T>, operations<T>,
                seek_filter_impl<dispatch_t<T, two_head, any_tag>>>
        {
        };

        template <>
        struct seek_filter_impl<two_head>
        {
            template <typename T, typename Device>
            static std::streampos seek(T& t, Device& d, stream_offset off,
                std::ios_base::seekdir way, std::ios_base::openmode which)
            {
                return t.seek(d, off, way, which);
            }
        };

        template <>
        struct seek_filter_impl<any_tag>
        {
            template <typename T, typename Device>
            static std::streampos seek(T& t, Device& d, stream_offset off,
                std::ios_base::seekdir way, std::ios_base::openmode)
            {
                return t.seek(d, off, way);
            }
        };
    }    // namespace detail
}    // namespace hpx::iostream

#include <hpx/config/warnings_suffix.hpp>
