//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <functional>
#include <type_traits>

namespace hpx { namespace lcos {
    template <typename R>
    class future;
    template <typename R>
    class shared_future;
}}    // namespace hpx::lcos

namespace hpx { namespace traits {
    namespace detail {
        template <typename Future, typename Enable = void>
        struct is_unique_future : std::false_type
        {
        };

        template <typename R>
        struct is_unique_future<lcos::future<R>> : std::true_type
        {
        };

        template <typename Future, typename Enable = void>
        struct is_future_customization_point : std::false_type
        {
        };
    }    // namespace detail

    template <typename Future>
    struct is_future : detail::is_future_customization_point<Future>
    {
    };

    template <typename R>
    struct is_future<lcos::future<R>> : std::true_type
    {
    };

    template <typename R>
    struct is_future<lcos::shared_future<R>> : std::true_type
    {
    };

    template <typename Future>
    struct is_ref_wrapped_future : std::false_type
    {
    };

    template <typename Future>
    struct is_ref_wrapped_future<std::reference_wrapper<Future>>
      : is_future<Future>
    {
    };
}}    // namespace hpx::traits
