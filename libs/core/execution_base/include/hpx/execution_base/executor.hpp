//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config/constexpr.hpp>
#include <hpx/functional/tag_fallback_invoke.hpp>
#include <hpx/functional/traits/is_invocable.hpp>

#include <exception>
#include <type_traits>
#include <utility>

namespace hpx { namespace execution { namespace experimental {
    HPX_INLINE_CONSTEXPR_VARIABLE struct execute_t
      : hpx::functional::tag_fallback<execute_t>
    {
        // TODO: Constrain to is_executor/is_sender
        template <typename Executor, typename F>
        friend constexpr HPX_FORCEINLINE auto tag_fallback_invoke(execute_t,
            Executor&& executor,
            F&& f) noexcept(noexcept(std::forward<Executor>(executor)
                                         .execute(std::forward<F>(f)))) ->
            typename std::enable_if<hpx::traits::is_invocable<F>::value,
                decltype(std::forward<Executor>(executor).execute(
                    std::forward<F>(f)))>::type
        {
            return std::forward<Executor>(executor).execute(std::forward<F>(f));
        }

        // TODO: tag_invoke(execute_t) fallback
        // TODO: submit(...) fallback
    } execute;

    HPX_INLINE_CONSTEXPR_VARIABLE struct bulk_execute_t
      : hpx::functional::tag_fallback<bulk_execute_t>
    {
        // TODO: Constrain to is_executor/is_sender
        template <typename Executor, typename F, typename N>
        friend constexpr HPX_FORCEINLINE auto tag_fallback_invoke(
            bulk_execute_t, Executor&& executor, F&& f,
            N n) noexcept(noexcept(std::forward<Executor>(executor)
                                       .bulk_execute(std::forward<F>(f)),
            n)) ->
            typename std::enable_if<hpx::traits::is_invocable<F, N>::value &&
                    std::is_convertible<N, std::size_t>::value,
                decltype(std::forward<Executor>(executor).bulk_execute(
                    std::forward<F>(f), n))>::type
        {
            return std::forward<Executor>(executor).bulk_execute(
                std::forward<F>(f), n);
        }

        // TODO: Fall back to one execute for each i in n.
    } bulk_execute;
}}}    // namespace hpx::execution::experimental
