//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c) 2013 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/functional/invoke_result.hpp>
#include <hpx/futures/traits/future_traits.hpp>
#include <hpx/futures/traits/is_future.hpp>
#include <hpx/type_support/always_void.hpp>
#include <hpx/type_support/identity.hpp>
#include <hpx/type_support/lazy_conditional.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace traits {
    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        struct no_executor
        {
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        template <typename Future, typename F>
        struct continuation_not_callable
        {
            static auto error(Future future, F& f)
            {
                f(std::move(future));
            }

            using type =
                decltype(error(std::declval<Future>(), std::declval<F&>()));
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Future, typename F, typename Enable = void>
        struct future_then_result
        {
            typedef typename continuation_not_callable<Future, F>::type type;
        };

        template <typename Future, typename F>
        struct future_then_result<Future, F,
            typename hpx::util::always_void<
                hpx::util::invoke_result_t<F&, Future>>::type>
        {
            using cont_result = hpx::util::invoke_result_t<F&, Future>;

            // perform unwrapping of future<future<R>>
            using result_type = util::lazy_conditional_t<
                hpx::traits::detail::is_unique_future<cont_result>::value,
                hpx::traits::future_traits<cont_result>,
                hpx::util::identity<cont_result>>;

            using type = hpx::lcos::future<result_type>;
        };

    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    template <typename Future, typename F>
    struct future_then_result : detail::future_then_result<Future, F>
    {
    };

    template <typename Future, typename F>
    using future_then_result_t = typename future_then_result<Future, F>::type;

}}    // namespace hpx::traits
