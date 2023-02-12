//  Copyright (c) 2022 Srinivas Yadav
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_DATAPAR)
#include <hpx/execution/traits/is_execution_policy.hpp>
#include <hpx/execution/traits/vector_pack_find.hpp>
#include <hpx/executors/datapar/execution_policy.hpp>
#include <hpx/functional/tag_invoke.hpp>
#include <hpx/parallel/algorithms/detail/advance_to_sentinel.hpp>
#include <hpx/parallel/algorithms/detail/distance.hpp>
#include <hpx/parallel/algorithms/detail/mismatch.hpp>
#include <hpx/parallel/datapar/handle_local_exceptions.hpp>
#include <hpx/parallel/datapar/loop.hpp>
#include <hpx/parallel/util/cancellation_token.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/result_types.hpp>
#include <hpx/parallel/util/zip_iterator.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { namespace detail {

    ///////////////////////////////////////////////////////////////////////////
    template <typename ExPolicy>
    struct datapar_mismatch
    {
        template <typename ZipIterator, typename Token, typename F>
        HPX_HOST_DEVICE HPX_FORCEINLINE static void call(std::size_t base_idx,
            ZipIterator it, std::size_t part_count, Token& tok, F&& f)
        {
            util::loop_idx_n<ExPolicy>(base_idx, it, part_count, tok,
                [&f, &tok](auto t, std::size_t i) mutable -> void {
                    auto msk = !hpx::invoke(f, hpx::get<0>(t), hpx::get<1>(t));
                    int offset = hpx::parallel::traits::find_first_of(msk);
                    if (offset != -1)
                    {
                        tok.cancel(i + offset);
                    }
                });
        }

        template <typename Iter1, typename Sent, typename Iter2, typename F>
        HPX_HOST_DEVICE HPX_FORCEINLINE static auto call(
            Iter1 first1, Sent last1, Iter2 first2, F&& f)
        {
            auto count = hpx::parallel::detail::distance(first1, last1);
            util::cancellation_token<std::size_t> tok(count);
            call(0, hpx::util::zip_iterator(first1, first2), count, tok,
                HPX_FORWARD(F, f));
            std::size_t mismatched = tok.get_data();

            if (mismatched != count)
                std::advance(first1, mismatched);
            else
                first1 = detail::advance_to_sentinel(first1, last1);

            std::advance(first2, mismatched);
            return std::make_pair(first1, first2);
        }
    };

    template <typename ExPolicy, typename ZipIterator, typename Token,
        typename F,
        HPX_CONCEPT_REQUIRES_(hpx::is_vectorpack_execution_policy_v<ExPolicy>)>
    HPX_HOST_DEVICE HPX_FORCEINLINE void tag_invoke(
        sequential_mismatch_t<ExPolicy>, std::size_t base_idx, ZipIterator it,
        std::size_t part_count, Token& tok, F&& f)
    {
        if constexpr (hpx::parallel::util::detail::iterator_datapar_compatible<
                          ZipIterator>::value)
        {
            return datapar_mismatch<ExPolicy>::call(
                base_idx, it, part_count, tok, HPX_FORWARD(F, f));
        }
        else
        {
            using base_policy_type =
                decltype((hpx::execution::experimental::to_non_simd(
                    std::declval<ExPolicy>())));
            return sequential_mismatch<base_policy_type>(
                base_idx, it, part_count, tok, HPX_FORWARD(F, f));
        }
    }

    template <typename ExPolicy, typename Iter1, typename Sent, typename Iter2,
        typename F,
        HPX_CONCEPT_REQUIRES_(hpx::is_vectorpack_execution_policy_v<ExPolicy>)>
    HPX_HOST_DEVICE HPX_FORCEINLINE auto tag_invoke(
        sequential_mismatch_t<ExPolicy>, Iter1 first1, Sent last1, Iter2 first2,
        F&& f)
    {
        if constexpr (hpx::parallel::util::detail::iterator_datapar_compatible<
                          Iter1>::value &&
            hpx::parallel::util::detail::iterator_datapar_compatible<
                Iter2>::value)
        {
            return datapar_mismatch<ExPolicy>::call(
                first1, last1, first2, HPX_FORWARD(F, f));
        }
        else
        {
            using base_policy_type =
                decltype((hpx::execution::experimental::to_non_simd(
                    std::declval<ExPolicy>())));
            return sequential_mismatch<base_policy_type>(
                first1, last1, first2, HPX_FORWARD(F, f));
        }
    }

    /////////////////////////////////////////////////////////////////////////
    template <typename ExPolicy>
    struct datapar_mismatch_binary
    {
        template <typename ZipIterator, typename Token, typename F,
            typename Proj1, typename Proj2>
        HPX_HOST_DEVICE HPX_FORCEINLINE static void call1(std::size_t base_idx,
            ZipIterator it, std::size_t part_count, Token& tok, F&& f,
            Proj1&& proj1, Proj2&& proj2)
        {
            util::loop_idx_n<ExPolicy>(base_idx, it, part_count, tok,
                [&f, &proj1, &proj2, &tok](
                    auto t, std::size_t i) mutable -> void {
                    auto msk =
                        !hpx::invoke(f, hpx::invoke(proj1, hpx::get<0>(t)),
                            hpx::invoke(proj2, hpx::get<1>(t)));
                    int offset = hpx::parallel::traits::find_first_of(msk);
                    if (offset != -1)
                    {
                        tok.cancel(i + offset);
                    }
                });
        }

        template <typename Iter1, typename Sent1, typename Iter2,
            typename Sent2, typename F, typename Proj1, typename Proj2>
        HPX_HOST_DEVICE HPX_FORCEINLINE static util::in_in_result<Iter1, Iter2>
        call2(Iter1 first1, Sent1 last1, Iter2 first2, Sent2 last2, F&& f,
            Proj1&& proj1, Proj2&& proj2)
        {
            auto count = hpx::parallel::detail::distance(first1, last1);

            util::cancellation_token<std::size_t> tok(count);
            call1(0, hpx::util::zip_iterator(first1, first2), count, tok,
                HPX_FORWARD(F, f), HPX_FORWARD(Proj1, proj1),
                HPX_FORWARD(Proj2, proj2));
            std::size_t mismatched = tok.get_data();

            if (mismatched != count)
            {
                std::advance(first1, mismatched);
                std::advance(first2, mismatched);
            }
            else
            {
                first1 = detail::advance_to_sentinel(first1, last1);
                first2 = detail::advance_to_sentinel(first2, last2);
            }
            return {first1, first2};
        }
    };

    template <typename ExPolicy, typename ZipIterator, typename Token,
        typename F, typename Proj1, typename Proj2,
        HPX_CONCEPT_REQUIRES_(hpx::is_vectorpack_execution_policy_v<ExPolicy>)>
    HPX_HOST_DEVICE HPX_FORCEINLINE void tag_invoke(
        sequential_mismatch_binary_t<ExPolicy>, std::size_t base_idx,
        ZipIterator it, std::size_t part_count, Token& tok, F&& f,
        Proj1&& proj1, Proj2&& proj2)
    {
        if constexpr (hpx::parallel::util::detail::iterator_datapar_compatible<
                          ZipIterator>::value)
        {
            return datapar_mismatch_binary<ExPolicy>::call1(base_idx, it,
                part_count, tok, HPX_FORWARD(F, f), HPX_FORWARD(Proj1, proj1),
                HPX_FORWARD(Proj2, proj2));
        }
        else
        {
            using base_policy_type =
                decltype((hpx::execution::experimental::to_non_simd(
                    std::declval<ExPolicy>())));
            return sequential_mismatch_binary<base_policy_type>(base_idx, it,
                part_count, tok, HPX_FORWARD(F, f), HPX_FORWARD(Proj1, proj1),
                HPX_FORWARD(Proj2, proj2));
        }
    }

    template <typename ExPolicy, typename Iter1, typename Sent1, typename Iter2,
        typename Sent2, typename F, typename Proj1, typename Proj2,
        HPX_CONCEPT_REQUIRES_(hpx::is_vectorpack_execution_policy_v<ExPolicy>)>
    HPX_HOST_DEVICE HPX_FORCEINLINE util::in_in_result<Iter1, Iter2> tag_invoke(
        sequential_mismatch_binary_t<ExPolicy>, Iter1 first1, Sent1 last1,
        Iter2 first2, Sent2 last2, F&& f, Proj1&& proj1, Proj2&& proj2)
    {
        if constexpr (hpx::parallel::util::detail::iterator_datapar_compatible<
                          Iter1>::value &&
            hpx::parallel::util::detail::iterator_datapar_compatible<
                Iter2>::value)
        {
            return datapar_mismatch_binary<ExPolicy>::call2(first1, last1,
                first2, last2, HPX_FORWARD(F, f), HPX_FORWARD(Proj1, proj1),
                HPX_FORWARD(Proj2, proj2));
        }
        else
        {
            using base_policy_type =
                decltype((hpx::execution::experimental::to_non_simd(
                    std::declval<ExPolicy>())));
            return sequential_mismatch_binary<base_policy_type>(first1, last1,
                first2, last2, HPX_FORWARD(F, f), HPX_FORWARD(Proj1, proj1),
                HPX_FORWARD(Proj2, proj2));
        }
    }
}}}    // namespace hpx::parallel::detail
#endif
