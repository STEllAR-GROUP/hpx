//  Copyright (c) 2022 Srinivas Yadav
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/execution/traits/is_execution_policy.hpp>
#include <hpx/functional/detail/tag_fallback_invoke.hpp>
#include <hpx/functional/invoke.hpp>
#include <hpx/parallel/algorithms/detail/advance_to_sentinel.hpp>
#include <hpx/parallel/util/compare_projected.hpp>
#include <hpx/parallel/util/loop.hpp>
#include <hpx/parallel/util/projection_identity.hpp>
#include <hpx/parallel/util/result_types.hpp>

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { inline namespace v1 { namespace detail {

    template <typename ExPolicy>
    struct sequential_mismatch_t final
      : hpx::functional::detail::tag_fallback<sequential_mismatch_t<ExPolicy>>
    {
    private:
        template <typename Iter1, typename Sent, typename Iter2, typename F>
        friend inline constexpr auto tag_fallback_invoke(sequential_mismatch_t,
            Iter1 first1, Sent last1, Iter2 first2, F&& f)
        {
            while (first1 != last1 && HPX_INVOKE(f, *first1, *first2))
            {
                ++first1, ++first2;
            }
            return std::make_pair(first1, first2);
        }

        template <typename ZipIterator, typename Token, typename F>
        friend inline constexpr void tag_fallback_invoke(sequential_mismatch_t,
            std::size_t base_idx, ZipIterator it, std::size_t part_count,
            Token& tok, F&& f)
        {
            util::loop_idx_n<ExPolicy>(base_idx, it, part_count, tok,
                [&f, &tok](auto t, std::size_t i) mutable -> void {
                    if (!hpx::util::invoke(f, hpx::get<0>(t), hpx::get<1>(t)))
                    {
                        tok.cancel(i);
                    }
                });
        }
    };

#if !defined(HPX_COMPUTE_DEVICE_CODE)
    template <typename ExPolicy>
    inline constexpr sequential_mismatch_t<ExPolicy> sequential_mismatch =
        sequential_mismatch_t<ExPolicy>{};
#else
    template <typename ExPolicy, typename Iter1, typename Sent, typename Iter2,
        typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE auto sequential_mismatch(
        Iter1 first1, Sent last1, Iter2 first2, F&& f)
    {
        return sequential_mismatch_t<ExPolicy>{}(
            first1, last1, first2, HPX_FORWARD(F, f));
    }

    template <typename ExPolicy, typename ZipIterator, typename Token,
        typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE void sequential_mismatch(
        std::size_t base_idx, ZipIterator it, std::size_t part_count,
        Token& tok, F&& f)
    {
        return sequential_mismatch_t<ExPolicy>{}(
            base_idx, it, part_count, tok, HPX_FORWARD(F, f));
    }
#endif

    template <typename ExPolicy>
    struct sequential_mismatch_binary_t final
      : hpx::functional::detail::tag_fallback<
            sequential_mismatch_binary_t<ExPolicy>>
    {
    private:
        // Our own version of the C++14 equal (_binary).
        template <typename Iter1, typename Sent1, typename Iter2,
            typename Sent2, typename F, typename Proj1, typename Proj2>
        friend inline constexpr util::in_in_result<Iter1, Iter2>
        tag_fallback_invoke(sequential_mismatch_binary_t, Iter1 first1,
            Sent1 last1, Iter2 first2, Sent2 last2, F&& f, Proj1&& proj1,
            Proj2&& proj2)
        {
            while (first1 != last1 && first2 != last2 &&
                HPX_INVOKE(
                    f, HPX_INVOKE(proj1, *first1), HPX_INVOKE(proj2, *first2)))
            {
                (void) ++first1, ++first2;
            }
            return {first1, first2};
        }

        template <typename ZipIterator, typename Token, typename F,
            typename Proj1, typename Proj2>
        friend inline constexpr void tag_fallback_invoke(
            sequential_mismatch_binary_t, std::size_t base_idx, ZipIterator it,
            std::size_t part_count, Token& tok, F&& f, Proj1&& proj1,
            Proj2&& proj2)
        {
            util::loop_idx_n<ExPolicy>(base_idx, it, part_count, tok,
                [&f, &proj1, &proj2, &tok](
                    auto t, std::size_t i) mutable -> void {
                    if (!hpx::util::invoke(f,
                            hpx::util::invoke(proj1, hpx::get<0>(t)),
                            hpx::util::invoke(proj2, hpx::get<1>(t))))
                    {
                        tok.cancel(i);
                    }
                });
        }
    };

#if !defined(HPX_COMPUTE_DEVICE_CODE)
    template <typename ExPolicy>
    inline constexpr sequential_mismatch_binary_t<ExPolicy>
        sequential_mismatch_binary = sequential_mismatch_binary_t<ExPolicy>{};
#else
    template <typename ExPolicy, typename Iter1, typename Sent1, typename Iter2,
        typename Sent2, typename F, typename Proj1, typename Proj2>
    HPX_HOST_DEVICE HPX_FORCEINLINE util::in_in_result<Iter1, Iter2>
    sequential_mismatch_binary(Iter1 first1, Sent1 last1, Iter2 first2,
        Sent2 last2, F&& f, Proj1&& proj1, Proj2&& proj2)
    {
        return sequential_mismatch_binary_t<ExPolicy>{}(first1, last1, first2,
            last2, HPX_FORWARD(F, f), HPX_FORWARD(Proj1, proj1),
            HPX_FORWARD(Proj2, proj2));
    }

    template <typename ExPolicy, typename ZipIterator, typename Token,
        typename F, typename Proj1, typename Proj2>
    HPX_HOST_DEVICE HPX_FORCEINLINE void sequential_mismatch_binary(
        std::size_t base_idx, ZipIterator it, std::size_t part_count,
        Token& tok, F&& f, Proj1&& proj1, Proj2&& proj2)
    {
        return sequential_mismatch_binary_t<ExPolicy>{}(base_idx, it,
            part_count, tok, HPX_FORWARD(F, f), HPX_FORWARD(Proj1, proj1),
            HPX_FORWARD(Proj2, proj2));
    }
#endif

}}}}    // namespace hpx::parallel::v1::detail
