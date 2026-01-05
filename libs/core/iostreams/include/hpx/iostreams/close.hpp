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
#include <hpx/iostreams/categories.hpp>
#include <hpx/iostreams/detail/adapter/non_blocking_adapter.hpp>
#include <hpx/iostreams/flush.hpp>
#include <hpx/iostreams/operations_fwd.hpp>
#include <hpx/iostreams/traits.hpp>
#include <hpx/modules/type_support.hpp>

#include <iosfwd>
#include <type_traits>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx::iostreams {

    HPX_CXX_CORE_EXPORT template <typename T>
    void close(T& t);

    HPX_CXX_CORE_EXPORT template <typename T>
    void close(T& t, std::ios_base::openmode which);

    HPX_CXX_CORE_EXPORT template <typename T, typename Sink>
    void close(T& t, Sink& snk, std::ios_base::openmode which);

    namespace detail {

        HPX_CXX_CORE_EXPORT template <typename T>
        void close_all(T& t)
        {
            try
            {
                iostreams::close(t, std::ios_base::in);
            }
            catch (...)
            {
                try
                {
                    iostreams::close(t, std::ios_base::out);
                }
                catch (...)
                {
                }
                throw;
            }
            iostreams::close(t, std::ios_base::out);
        }

        HPX_CXX_CORE_EXPORT template <typename T, typename Sink>
        void close_all(T& t, Sink& snk)
        {
            try
            {
                iostreams::close(t, snk, std::ios_base::in);
            }
            catch (...)
            {
                try
                {
                    iostreams::close(t, snk, std::ios_base::out);
                }
                catch (...)
                {
                }
                throw;
            }
            iostreams::close(t, snk, std::ios_base::out);
        }
    }    // namespace detail.

    namespace detail {

        HPX_CXX_CORE_EXPORT template <typename T>
        struct close_impl;
    }    // namespace detail.

    HPX_CXX_CORE_EXPORT template <typename T>
    void close(T& t)
    {
        detail::close_all(t);
    }

    HPX_CXX_CORE_EXPORT template <typename T>
    void close(T& t, std::ios_base::openmode which)
    {
        if (which == (std::ios_base::in | std::ios_base::out))
        {
            detail::close_all(t);
        }
        else
        {
            detail::close_impl<T>::close(util::unwrap_ref(t), which);
        }
    }

    HPX_CXX_CORE_EXPORT template <typename T, typename Sink>
    void close(T& t, Sink& snk, std::ios_base::openmode which)
    {
        if (which == (std::ios_base::in | std::ios_base::out))
        {
            detail::close_all(t, snk);
        }
        else
        {
            detail::close_impl<T>::close(util::unwrap_ref(t), snk, which);
        }
    }

    namespace detail {

        //------------------Definition of close_impl----------------------------------//
        HPX_CXX_CORE_EXPORT struct close_stream
        {
        };

        HPX_CXX_CORE_EXPORT struct close_filtering_stream
        {
        };

        HPX_CXX_CORE_EXPORT template <typename T>
        struct close_tag
        {
            using category = category_of_t<T>;
            using unwrapped = util::unwrap_reference_t<T>;

            using type = util::select_t<
                std::negation<std::is_convertible<category, closable_tag>>,
                any_tag,
                std::disjunction<is_iostreams_stream<unwrapped>,
                    is_iostreams_stream_buffer<unwrapped>>,
                close_stream,
                std::disjunction<is_filtering_stream<unwrapped>,
                    is_filtering_streambuf<unwrapped>>,
                close_filtering_stream,
                std::disjunction<std::is_convertible<category, two_sequence>,
                    std::is_convertible<category, dual_use>>,
                two_sequence, util::else_t, closable_tag>;
        };

        HPX_CXX_CORE_EXPORT template <typename T>
        struct close_impl
          : std::conditional_t<is_custom_v<T>, operations<T>,
                close_impl<typename close_tag<T>::type>>
        {
        };

        template <>
        struct close_impl<any_tag>
        {
            template <typename T>
            static void close(T& t, std::ios_base::openmode const which)
            {
                if (which == std::ios_base::out)
                    iostreams::flush(t);
            }

            template <typename T, typename Sink>
            static void close(
                T& t, Sink& snk, std::ios_base::openmode const which)
            {
                if (which == std::ios_base::out)
                {
                    non_blocking_adapter<Sink> nb(snk);
                    iostreams::flush(t, nb);
                }
            }
        };

        template <>
        struct close_impl<close_stream>
        {
            template <typename T>
            static void close(T& t)
            {
                t.close();
            }

            template <typename T>
            static void close(T& t, std::ios_base::openmode const which)
            {
                if (which == std::ios_base::out)
                    t.close();
            }
        };

        template <>
        struct close_impl<close_filtering_stream>
        {
            template <typename T>
            static void close(T& t, std::ios_base::openmode const which)
            {
                using category = category_of<T>::type;
                constexpr bool in = std::is_convertible_v<category, input> &&
                    !std::is_convertible_v<category, output>;

                if (in == (which == std::ios_base::in) && t.is_complete())
                    t.pop();
            }
        };

        template <>
        struct close_impl<closable_tag>
        {
            template <typename T>
            static void close(T& t, std::ios_base::openmode const which)
            {
                using category = typename category_of<T>::type;
                constexpr bool in = std::is_convertible_v<category, input> &&
                    !std::is_convertible_v<category, output>;

                if (in == (which == std::ios_base::in))
                    t.close();
            }

            template <typename T, typename Sink>
            static void close(
                T& t, Sink& snk, std::ios_base::openmode const which)
            {
                using category = category_of<T>::type;
                constexpr bool in = std::is_convertible_v<category, input> &&
                    !std::is_convertible_v<category, output>;

                if (in == (which == std::ios_base::in))
                {
                    non_blocking_adapter<Sink> nb(snk);
                    t.close(nb);
                }
            }
        };

        template <>
        struct close_impl<two_sequence>
        {
            template <typename T>
            static void close(T& t, std::ios_base::openmode which)
            {
                t.close(which);
            }

            template <typename T, typename Sink>
            static void close(T& t, Sink& snk, std::ios_base::openmode which)
            {
                non_blocking_adapter<Sink> nb(snk);
                t.close(nb, which);
            }
        };
    }    // namespace detail
}    // namespace hpx::iostreams

#include <hpx/config/warnings_suffix.hpp>
