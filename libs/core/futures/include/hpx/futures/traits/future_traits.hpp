//  Copyright (c) 2007-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/local/config.hpp>
#include <hpx/futures/traits/is_future.hpp>

#include <type_traits>

namespace hpx { namespace lcos {
    template <typename R>
    class future;
    template <typename R>
    class shared_future;
}}    // namespace hpx::lcos

namespace hpx { namespace traits {
    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        template <typename Future, typename Enable = void>
        struct future_traits_customization_point
        {
        };
    }    // namespace detail

    template <typename T>
    struct future_traits : detail::future_traits_customization_point<T>
    {
    };

    template <typename Future>
    struct future_traits<Future const> : future_traits<Future>
    {
    };

    template <typename Future>
    struct future_traits<Future&> : future_traits<Future>
    {
    };

    template <typename Future>
    struct future_traits<Future const&> : future_traits<Future>
    {
    };

    template <typename R>
    struct future_traits<lcos::future<R>>
    {
        using type = R;
        using result_type = R;
    };

    template <typename R>
    struct future_traits<lcos::shared_future<R>>
    {
        using type = R;
        using result_type = R const&;
    };

    template <>
    struct future_traits<lcos::shared_future<void>>
    {
        using type = void;
        using result_type = void;
    };

    template <typename Future>
    using future_traits_t = typename future_traits<Future>::type;

    ///////////////////////////////////////////////////////////////////////////
    template <typename Future, typename Enable = void>
    struct is_future_void : std::false_type
    {
    };

    template <typename Future>
    struct is_future_void<Future, std::enable_if_t<is_future_v<Future>>>
      : std::is_void<future_traits_t<Future>>
    {
    };
}}    // namespace hpx::traits
