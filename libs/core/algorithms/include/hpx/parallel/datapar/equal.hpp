//  Copyright (c) 2022 Srinivas Yadav
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_DATAPAR)
#include <hpx/execution/traits/is_execution_policy.hpp>
#include <hpx/execution/traits/vector_pack_all_any_none.hpp>
#include <hpx/executors/datapar/execution_policy.hpp>
#include <hpx/functional/tag_invoke.hpp>
#include <hpx/parallel/algorithms/detail/distance.hpp>
#include <hpx/parallel/algorithms/detail/equal.hpp>
#include <hpx/parallel/datapar/handle_local_exceptions.hpp>
#include <hpx/parallel/datapar/loop.hpp>
#include <hpx/parallel/util/cancellation_token.hpp>
#include <hpx/parallel/util/result_types.hpp>
#include <hpx/parallel/util/zip_iterator.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { namespace detail {

    ///////////////////////////////////////////////////////////////////////////
    template <typename ExPolicy>
    struct datapar_equal
    {
        template <typename ZipIterator, typename Token, typename F>
        HPX_HOST_DEVICE HPX_FORCEINLINE static void call(
            ZipIterator it, std::size_t part_count, Token& tok, F&& f)
        {
            util::loop_n<ExPolicy>(it, part_count, tok,
                [&f, &tok](auto const& curr) mutable -> void {
                    auto t = *curr;
                    if (!hpx::parallel::traits::all_of(
                            HPX_INVOKE(f, hpx::get<0>(t), hpx::get<1>(t))))
                    {
                        tok.cancel();
                    }
                });
        }

        template <typename InIter1, typename InIter2, typename F>
        HPX_HOST_DEVICE HPX_FORCEINLINE static bool call(
            InIter1 first1, InIter1 last1, InIter2 first2, F&& f)
        {
            auto count = hpx::parallel::detail::distance(first1, last1);
            util::cancellation_token<> tok;
            call(hpx::util::zip_iterator(first1, first2), count, tok,
                HPX_FORWARD(F, f));
            return !tok.was_cancelled();
        }
    };

    template <typename ExPolicy, typename ZipIterator, typename Token,
        typename F,
        HPX_CONCEPT_REQUIRES_(hpx::is_vectorpack_execution_policy_v<ExPolicy>)>
    HPX_HOST_DEVICE HPX_FORCEINLINE void tag_invoke(
        sequential_equal_t<ExPolicy>, ZipIterator it, std::size_t part_count,
        Token& tok, F&& f)
    {
        if constexpr (hpx::parallel::util::detail::iterator_datapar_compatible<
                          ZipIterator>::value)
        {
            return datapar_equal<ExPolicy>::call(
                it, part_count, tok, HPX_FORWARD(F, f));
        }
        else
        {
            using base_policy_type =
                decltype((hpx::execution::experimental::to_non_simd(
                    std::declval<ExPolicy>())));
            return sequential_equal<base_policy_type>(
                it, part_count, tok, HPX_FORWARD(F, f));
        }
    }

    template <typename ExPolicy, typename InIter1, typename InIter2, typename F,
        HPX_CONCEPT_REQUIRES_(hpx::is_vectorpack_execution_policy_v<ExPolicy>)>
    HPX_HOST_DEVICE HPX_FORCEINLINE bool tag_invoke(
        sequential_equal_t<ExPolicy>, InIter1 first1, InIter1 last1,
        InIter2 first2, F&& f)
    {
        if constexpr (hpx::parallel::util::detail::iterator_datapar_compatible<
                          InIter1>::value &&
            hpx::parallel::util::detail::iterator_datapar_compatible<
                InIter2>::value)
        {
            return datapar_equal<ExPolicy>::call(
                first1, last1, first2, HPX_FORWARD(F, f));
        }
        else
        {
            using base_policy_type =
                decltype((hpx::execution::experimental::to_non_simd(
                    std::declval<ExPolicy>())));
            return sequential_equal<base_policy_type>(
                first1, last1, first2, HPX_FORWARD(F, f));
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename ExPolicy>
    struct datapar_equal_binary
    {
        template <typename ZipIterator, typename Token, typename F,
            typename Proj1, typename Proj2>
        HPX_HOST_DEVICE HPX_FORCEINLINE static void call(ZipIterator it,
            std::size_t part_count, Token& tok, F&& f, Proj1&& proj1,
            Proj2&& proj2)
        {
            util::loop_n<ExPolicy>(it, part_count, tok,
                [&f, &proj1, &proj2, &tok](auto const& curr) mutable -> void {
                    auto t = *curr;
                    if (!hpx::parallel::traits::all_of(
                            hpx::invoke(f, hpx::invoke(proj1, hpx::get<0>(t)),
                                hpx::invoke(proj2, hpx::get<1>(t)))))
                    {
                        tok.cancel();
                    }
                });
        }

        template <typename InIter1, typename Sent1, typename InIter2,
            typename F, typename Proj1, typename Proj2>
        HPX_HOST_DEVICE HPX_FORCEINLINE static bool call(InIter1 first1,
            Sent1 last1, InIter2 first2, F&& f, Proj1&& proj1, Proj2&& proj2)
        {
            auto count = hpx::parallel::detail::distance(first1, last1);
            util::cancellation_token<> tok;
            call(hpx::util::zip_iterator(first1, first2), count, tok,
                HPX_FORWARD(F, f), HPX_FORWARD(Proj1, proj1),
                HPX_FORWARD(Proj2, proj2));
            return !tok.was_cancelled();
        }
    };

    template <typename ExPolicy, typename ZipIterator, typename Token,
        typename F, typename Proj1, typename Proj2,
        HPX_CONCEPT_REQUIRES_(hpx::is_vectorpack_execution_policy_v<ExPolicy>)>
    HPX_HOST_DEVICE HPX_FORCEINLINE void tag_invoke(
        sequential_equal_binary_t<ExPolicy>, ZipIterator it,
        std::size_t part_count, Token& tok, F&& f, Proj1&& proj1, Proj2&& proj2)
    {
        if constexpr (hpx::parallel::util::detail::iterator_datapar_compatible<
                          ZipIterator>::value)
        {
            return datapar_equal_binary<ExPolicy>::call(it, part_count, tok,
                HPX_FORWARD(F, f), HPX_FORWARD(Proj1, proj1),
                HPX_FORWARD(Proj2, proj2));
        }
        else
        {
            using base_policy_type =
                decltype((hpx::execution::experimental::to_non_simd(
                    std::declval<ExPolicy>())));
            return sequential_equal_binary<base_policy_type>(it, part_count,
                tok, HPX_FORWARD(F, f), HPX_FORWARD(Proj1, proj1),
                HPX_FORWARD(Proj2, proj2));
        }
    }

    template <typename ExPolicy, typename InIter1, typename Sent1,
        typename InIter2, typename Sent2, typename F, typename Proj1,
        typename Proj2,
        HPX_CONCEPT_REQUIRES_(hpx::is_vectorpack_execution_policy_v<ExPolicy>)>
    HPX_HOST_DEVICE HPX_FORCEINLINE bool tag_invoke(
        sequential_equal_binary_t<ExPolicy>, InIter1 first1, Sent1 last1,
        InIter2 first2, Sent2 last2, F&& f, Proj1&& proj1, Proj2&& proj2)
    {
        if constexpr (hpx::parallel::util::detail::iterator_datapar_compatible<
                          InIter1>::value &&
            hpx::parallel::util::detail::iterator_datapar_compatible<
                InIter2>::value)
        {
            return datapar_equal_binary<ExPolicy>::call(first1, last1, first2,
                HPX_FORWARD(F, f), HPX_FORWARD(Proj1, proj1),
                HPX_FORWARD(Proj2, proj2));
        }
        else
        {
            using base_policy_type =
                decltype((hpx::execution::experimental::to_non_simd(
                    std::declval<ExPolicy>())));
            return sequential_equal_binary<base_policy_type>(first1, last1,
                first2, last2, HPX_FORWARD(F, f), HPX_FORWARD(Proj1, proj1),
                HPX_FORWARD(Proj2, proj2));
        }
    }
}}}    // namespace hpx::parallel::detail
#endif
