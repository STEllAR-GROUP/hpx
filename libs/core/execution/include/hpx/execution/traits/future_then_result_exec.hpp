//  Copyright (c) 2007-2022 Hartmut Kaiser
//  Copyright (c) 2013 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/execution/traits/executor_traits.hpp>
#include <hpx/execution_base/traits/is_executor.hpp>
#include <hpx/functional/invoke_result.hpp>
#include <hpx/futures/traits/future_then_result.hpp>
#include <hpx/futures/traits/future_traits.hpp>
#include <hpx/futures/traits/is_future.hpp>
#include <hpx/type_support/identity.hpp>
#include <hpx/type_support/lazy_conditional.hpp>

#include <type_traits>
#include <utility>

namespace hpx::traits {

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {

        ///////////////////////////////////////////////////////////////////////
        template <typename Executor, typename Future, typename F,
            typename Enable = void>
        struct future_then_executor_result
        {
            using type = typename continuation_not_callable<Future, F>::type;
        };

        template <typename Executor, typename Future, typename F>
        struct future_then_executor_result<Executor, Future, F,
            std::void_t<hpx::util::invoke_result_t<F&, Future>>>
        {
            using func_result_type = hpx::util::invoke_result_t<F&, Future>;

            using cont_result =
                traits::executor_future_t<Executor, func_result_type, Future>;

            // perform unwrapping of future<future<R>>
            using result_type = util::lazy_conditional_t<
                hpx::traits::detail::is_unique_future_v<cont_result>,
                hpx::traits::future_traits<cont_result>,
                hpx::type_identity<cont_result>>;

            using type = hpx::future<result_type>;
        };
    }    // namespace detail

    template <typename Executor, typename Future, typename F>
    struct future_then_executor_result
      : detail::future_then_executor_result<std::decay_t<Executor>, Future, F>
    {
    };

    template <typename Executor, typename Future, typename F>
    using future_then_executor_result_t =
        typename future_then_executor_result<Executor, Future, F>::type;
}    // namespace hpx::traits
