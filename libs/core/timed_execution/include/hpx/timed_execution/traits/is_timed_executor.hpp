//  Copyright (c) 2014-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/execution_base/traits/is_executor.hpp>

#include <type_traits>

namespace hpx::parallel::execution {

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {

        /// \cond NOINTERNAL
        template <typename T>
        struct is_timed_executor : std::false_type
        {
        };
        /// \endcond
    }    // namespace detail

    // Executor type traits:

    // Condition: T meets the syntactic requirements for OneWayExecutor
    // Precondition: T is a complete type
    template <typename T>
    struct is_timed_executor : detail::is_timed_executor<std::decay_t<T>>
    {
    };

    template <typename T>
    using is_timed_executor_t = typename is_timed_executor<T>::type;

    template <typename T>
    inline constexpr bool is_timed_executor_v = is_timed_executor<T>::value;
}    // namespace hpx::parallel::execution

namespace hpx::traits {

    // new executor framework
    template <typename Executor, typename Enable = void>
    struct is_timed_executor : parallel::execution::is_timed_executor<Executor>
    {
    };
}    // namespace hpx::traits
// namespace hpx::traits
