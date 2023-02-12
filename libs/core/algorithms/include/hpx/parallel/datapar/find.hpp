//  Copyright (c) 2021 Srinivas Yadav
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_DATAPAR)
#include <hpx/execution/traits/is_execution_policy.hpp>
#include <hpx/execution/traits/vector_pack_all_any_none.hpp>
#include <hpx/execution/traits/vector_pack_find.hpp>
#include <hpx/executors/datapar/execution_policy.hpp>
#include <hpx/functional/tag_invoke.hpp>
#include <hpx/parallel/algorithms/detail/find.hpp>
#include <hpx/parallel/datapar/handle_local_exceptions.hpp>
#include <hpx/parallel/datapar/iterator_helpers.hpp>
#include <hpx/parallel/datapar/loop.hpp>
#include <hpx/parallel/datapar/zip_iterator.hpp>
#include <hpx/parallel/util/cancellation_token.hpp>
#include <hpx/parallel/util/result_types.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { namespace detail {
    ///////////////////////////////////////////////////////////////////////////
    template <typename ExPolicy>
    struct datapar_find
    {
        template <typename Iterator, typename Sentinel, typename T,
            typename Proj>
        static inline Iterator call(
            Iterator first, Sentinel last, T const& val, Proj proj)
        {
            return util::loop_pred<std::decay_t<ExPolicy>>(
                first, last, [&val, &proj](auto const& curr) {
                    auto msk = HPX_INVOKE(proj, *curr) == val;
                    return hpx::parallel::traits::find_first_of(msk);
                });
        }

        template <typename FwdIter, typename Token, typename T, typename Proj>
        static inline constexpr void call(std::size_t base_idx,
            FwdIter part_begin, std::size_t part_count, Token& tok,
            T const& val, Proj&& proj)
        {
            util::loop_idx_n<std::decay_t<ExPolicy>>(base_idx, part_begin,
                part_count, tok,
                [&val, &proj, &tok](auto& v, std::size_t i) -> void {
                    auto msk = HPX_INVOKE(proj, v) == val;
                    int offset = hpx::parallel::traits::find_first_of(msk);
                    if (offset != -1)
                        tok.cancel(i + offset);
                });
        }
    };

    template <typename ExPolicy, typename Iterator, typename Sentinel,
        typename T, typename Proj = hpx::identity,
        HPX_CONCEPT_REQUIRES_(hpx::is_vectorpack_execution_policy_v<ExPolicy>)>
    HPX_HOST_DEVICE HPX_FORCEINLINE Iterator tag_invoke(
        sequential_find_t<ExPolicy>, Iterator first, Sentinel last,
        T const& val, Proj proj = Proj())
    {
        if constexpr (hpx::parallel::util::detail::iterator_datapar_compatible<
                          Iterator>::value)
        {
            return datapar_find<ExPolicy>::call(first, last, val, proj);
        }
        else
        {
            using base_policy_type =
                decltype((hpx::execution::experimental::to_non_simd(
                    std::declval<ExPolicy>())));
            return sequential_find<base_policy_type>(first, last, val, proj);
        }
    }

    template <typename ExPolicy, typename FwdIter, typename Token, typename T,
        typename Proj,
        HPX_CONCEPT_REQUIRES_(hpx::is_vectorpack_execution_policy_v<ExPolicy>)>
    HPX_HOST_DEVICE HPX_FORCEINLINE void tag_invoke(sequential_find_t<ExPolicy>,
        std::size_t base_idx, FwdIter part_begin, std::size_t part_count,
        Token& tok, T const& val, Proj&& proj)
    {
        return datapar_find<ExPolicy>::call(base_idx, part_begin, part_count,
            tok, val, HPX_FORWARD(Proj, proj));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename ExPolicy>
    struct datapar_find_if
    {
        template <typename Iterator, typename Sentinel, typename Pred,
            typename Proj>
        static inline Iterator call(
            Iterator first, Sentinel last, Pred pred, Proj proj)
        {
            return util::loop_pred<std::decay_t<ExPolicy>>(
                first, last, [&pred, &proj](auto const& curr) {
                    auto msk = HPX_INVOKE(pred, HPX_INVOKE(proj, *curr));
                    return hpx::parallel::traits::find_first_of(msk);
                });
        }

        template <typename FwdIter, typename Token, typename F, typename Proj>
        static inline constexpr void call(FwdIter part_begin,
            std::size_t part_count, Token& tok, F&& op, Proj&& proj)
        {
            util::loop_n<std::decay_t<ExPolicy>>(part_begin, part_count, tok,
                [&op, &tok, &proj](auto const& curr) {
                    auto msk = HPX_INVOKE(op, HPX_INVOKE(proj, *curr));
                    if (hpx::parallel::traits::any_of(msk))
                    {
                        tok.cancel();
                    }
                });
        }

        template <typename FwdIter, typename Token, typename F, typename Proj>
        static inline constexpr void call(std::size_t base_idx,
            FwdIter part_begin, std::size_t part_count, Token& tok, F&& f,
            Proj&& proj)
        {
            util::loop_idx_n<std::decay_t<ExPolicy>>(base_idx, part_begin,
                part_count, tok,
                [&f, &proj, &tok](auto& v, std::size_t i) -> void {
                    auto msk = HPX_INVOKE(f, HPX_INVOKE(proj, v));
                    int offset = hpx::parallel::traits::find_first_of(msk);
                    if (offset != -1)
                        tok.cancel(i + offset);
                });
        }
    };

    template <typename ExPolicy, typename Iterator, typename Sentinel,
        typename Pred, typename Proj = hpx::identity,
        HPX_CONCEPT_REQUIRES_(hpx::is_vectorpack_execution_policy_v<ExPolicy>)>
    HPX_HOST_DEVICE HPX_FORCEINLINE Iterator tag_invoke(
        sequential_find_if_t<ExPolicy>, Iterator first, Sentinel last,
        Pred pred, Proj proj = Proj())
    {
        if constexpr (hpx::parallel::util::detail::iterator_datapar_compatible<
                          Iterator>::value)
        {
            return datapar_find_if<ExPolicy>::call(first, last, pred, proj);
        }
        else
        {
            using base_policy_type =
                decltype((hpx::execution::experimental::to_non_simd(
                    std::declval<ExPolicy>())));
            return sequential_find_if<base_policy_type>(
                first, last, pred, proj);
        }
    }

    template <typename ExPolicy, typename FwdIter, typename Token, typename F,
        typename Proj,
        HPX_CONCEPT_REQUIRES_(hpx::is_vectorpack_execution_policy_v<ExPolicy>)>
    HPX_HOST_DEVICE HPX_FORCEINLINE void tag_invoke(
        sequential_find_if_t<ExPolicy>, FwdIter part_begin,
        std::size_t part_count, Token& tok, F&& op, Proj&& proj)
    {
        return datapar_find_if<ExPolicy>::call(part_begin, part_count, tok,
            HPX_FORWARD(F, op), HPX_FORWARD(Proj, proj));
    }

    template <typename ExPolicy, typename FwdIter, typename Token, typename F,
        typename Proj,
        HPX_CONCEPT_REQUIRES_(hpx::is_vectorpack_execution_policy_v<ExPolicy>)>
    HPX_HOST_DEVICE HPX_FORCEINLINE void tag_invoke(
        sequential_find_if_t<ExPolicy>, std::size_t base_idx,
        FwdIter part_begin, std::size_t part_count, Token& tok, F&& op,
        Proj&& proj)
    {
        return datapar_find_if<ExPolicy>::call(base_idx, part_begin, part_count,
            tok, HPX_FORWARD(F, op), HPX_FORWARD(Proj, proj));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename ExPolicy>
    struct datapar_find_if_not
    {
        template <typename Iterator, typename Sentinel, typename Pred,
            typename Proj>
        static inline Iterator call(
            Iterator first, Sentinel last, Pred pred, Proj proj)
        {
            return util::loop_pred<std::decay_t<ExPolicy>>(
                first, last, [&pred, &proj](auto const& curr) mutable {
                    auto msk = !HPX_INVOKE(pred, HPX_INVOKE(proj, *curr));
                    return hpx::parallel::traits::find_first_of(msk);
                });
        }

        template <typename FwdIter, typename Token, typename F, typename Proj>
        static inline constexpr void call(FwdIter part_begin,
            std::size_t part_count, Token& tok, F&& op, Proj&& proj)
        {
            util::loop_n<std::decay_t<ExPolicy>>(part_begin, part_count, tok,
                [&op, &tok, &proj](auto const& curr) {
                    auto msk = !HPX_INVOKE(op, HPX_INVOKE(proj, *curr));
                    if (hpx::parallel::traits::any_of(msk))
                    {
                        tok.cancel();
                    }
                });
        }

        template <typename FwdIter, typename Token, typename F, typename Proj>
        static inline constexpr void call(std::size_t base_idx,
            FwdIter part_begin, std::size_t part_count, Token& tok, F&& f,
            Proj&& proj)
        {
            util::loop_idx_n<std::decay_t<ExPolicy>>(base_idx, part_begin,
                part_count, tok,
                [&f, &proj, &tok](auto& v, std::size_t i) -> void {
                    auto msk = !HPX_INVOKE(f, HPX_INVOKE(proj, v));
                    int offset = hpx::parallel::traits::find_first_of(msk);
                    if (offset != -1)
                        tok.cancel(i + offset);
                });
        }
    };

    template <typename ExPolicy, typename Iterator, typename Sentinel,
        typename Pred, typename Proj = hpx::identity,
        HPX_CONCEPT_REQUIRES_(hpx::is_vectorpack_execution_policy_v<ExPolicy>)>
    HPX_HOST_DEVICE HPX_FORCEINLINE Iterator tag_invoke(
        sequential_find_if_not_t<ExPolicy>, Iterator first, Sentinel last,
        Pred pred, Proj proj = Proj())
    {
        if constexpr (hpx::parallel::util::detail::iterator_datapar_compatible<
                          Iterator>::value)
        {
            return datapar_find_if_not<ExPolicy>::call(first, last, pred, proj);
        }
        else
        {
            using base_policy_type =
                decltype((hpx::execution::experimental::to_non_simd(
                    std::declval<ExPolicy>())));
            return sequential_find_if_not<base_policy_type>(
                first, last, pred, proj);
        }
    }

    template <typename ExPolicy, typename FwdIter, typename Token, typename F,
        typename Proj,
        HPX_CONCEPT_REQUIRES_(hpx::is_vectorpack_execution_policy_v<ExPolicy>)>
    HPX_HOST_DEVICE HPX_FORCEINLINE void tag_invoke(
        sequential_find_if_not_t<ExPolicy>, FwdIter part_begin,
        std::size_t part_count, Token& tok, F&& op, Proj&& proj)
    {
        return datapar_find_if_not<ExPolicy>::call(part_begin, part_count, tok,
            HPX_FORWARD(F, op), HPX_FORWARD(Proj, proj));
    }

    template <typename ExPolicy, typename FwdIter, typename Token, typename F,
        typename Proj,
        HPX_CONCEPT_REQUIRES_(hpx::is_vectorpack_execution_policy_v<ExPolicy>)>
    HPX_HOST_DEVICE HPX_FORCEINLINE void tag_invoke(
        sequential_find_if_not_t<ExPolicy>, std::size_t base_idx,
        FwdIter part_begin, std::size_t part_count, Token& tok, F&& op,
        Proj&& proj)
    {
        return datapar_find_if_not<ExPolicy>::call(base_idx, part_begin,
            part_count, tok, HPX_FORWARD(F, op), HPX_FORWARD(Proj, proj));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename ExPolicy>
    struct datapar_find_end_t
    {
        template <typename Iter1, typename Sent1, typename Iter2,
            typename Sent2, typename Pred, typename Proj1, typename Proj2>
        static inline constexpr Iter1 call(Iter1 first1, Sent1 last1,
            Iter2 first2, Sent2 last2, Pred&& op, Proj1&& proj1, Proj2&& proj2)
        {
            using difference_type =
                typename std::iterator_traits<Iter1>::difference_type;
            difference_type diff = detail::distance(first2, last2);
            difference_type count = detail::distance(first1, last1);
            util::cancellation_token<difference_type,
                std::greater<difference_type>>
                tok(-1);

            call(first1, first2, 0, count - diff + 1, diff, tok,
                HPX_FORWARD(Pred, op), HPX_FORWARD(Proj1, proj1),
                HPX_FORWARD(Proj2, proj2));

            difference_type find_end_res = tok.get_data();

            if (find_end_res >= 0 && find_end_res != count)
                std::advance(first1, find_end_res);
            else
                first1 = last1;
            return first1;
        }

        template <typename Iter1, typename Iter2, typename Token, typename Pred,
            typename Proj1, typename Proj2>
        static inline constexpr void call(Iter1 it, Iter2 first2,
            std::size_t base_idx, std::size_t part_size, std::size_t diff,
            Token& tok, Pred&& op, Proj1&& proj1, Proj2&& proj2)
        {
            std::size_t idx = 0;
            util::loop_idx_n<hpx::execution::parallel_policy>(base_idx, it,
                part_size, tok,
                [=, &tok, &op, &proj1, &proj2, &idx](
                    auto, std::size_t i) -> void {
                    auto begin = hpx::util::zip_iterator(it + idx, first2);
                    ++idx;
                    util::cancellation_token<> local_tok;
                    util::loop_n<hpx::execution::simd_policy>(begin, diff,
                        local_tok,
                        [&op, &proj1, &proj2, &local_tok](auto t) -> void {
                            using hpx::get;
                            if (!hpx::parallel::traits::all_of(hpx::invoke(op,
                                    hpx::invoke(proj1, get<0>(*t)),
                                    hpx::invoke(proj2, get<1>(*t)))))
                            {
                                local_tok.cancel();
                            }
                        });
                    if (!local_tok.was_cancelled())
                        tok.cancel(i);
                });
        }
    };

    template <typename ExPolicy, typename Iter1, typename Sent1, typename Iter2,
        typename Sent2, typename Pred, typename Proj1, typename Proj2,
        HPX_CONCEPT_REQUIRES_(hpx::is_vectorpack_execution_policy_v<ExPolicy>)>
    HPX_HOST_DEVICE HPX_FORCEINLINE Iter1 tag_invoke(
        sequential_find_end_t<ExPolicy>, Iter1 first1, Sent1 last1,
        Iter2 first2, Sent2 last2, Pred&& op, Proj1&& proj1, Proj2&& proj2)
    {
        if constexpr (hpx::parallel::util::detail::iterator_datapar_compatible<
                          Iter1>::value &&
            hpx::parallel::util::detail::iterator_datapar_compatible<
                Iter2>::value)
        {
            return datapar_find_end_t<ExPolicy>::call(first1, last1, first2,
                last2, HPX_FORWARD(Pred, op), HPX_FORWARD(Proj1, proj1),
                HPX_FORWARD(Proj2, proj2));
        }
        else
        {
            using base_policy_type =
                decltype((hpx::execution::experimental::to_non_simd(
                    std::declval<ExPolicy>())));
            return sequential_find_end<base_policy_type>(first1, last1, first2,
                last2, HPX_FORWARD(Pred, op), HPX_FORWARD(Proj1, proj1),
                HPX_FORWARD(Proj2, proj2));
        }
    }

    template <typename ExPolicy, typename Iter1, typename Iter2, typename Token,
        typename Pred, typename Proj1, typename Proj2,
        HPX_CONCEPT_REQUIRES_(hpx::is_vectorpack_execution_policy_v<ExPolicy>)>
    HPX_HOST_DEVICE HPX_FORCEINLINE void tag_invoke(
        sequential_find_end_t<ExPolicy>, Iter1 it, Iter2 first2,
        std::size_t base_idx, std::size_t part_size, std::size_t diff,
        Token& tok, Pred&& op, Proj1&& proj1, Proj2&& proj2)
    {
        if constexpr (hpx::parallel::util::detail::iterator_datapar_compatible<
                          Iter1>::value &&
            hpx::parallel::util::detail::iterator_datapar_compatible<
                Iter2>::value)
        {
            return datapar_find_end_t<ExPolicy>::call(it, first2, base_idx,
                part_size, diff, tok, HPX_FORWARD(Pred, op),
                HPX_FORWARD(Proj1, proj1), HPX_FORWARD(Proj2, proj2));
        }
        else
        {
            using base_policy_type =
                decltype((hpx::execution::experimental::to_non_simd(
                    std::declval<ExPolicy>())));
            return sequential_find_end<base_policy_type>(it, first2, base_idx,
                part_size, diff, tok, HPX_FORWARD(Pred, op),
                HPX_FORWARD(Proj1, proj1), HPX_FORWARD(Proj2, proj2));
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename ExPolicy>
    struct datapar_find_first_of
    {
        template <typename InIter1, typename InIter2, typename Pred,
            typename Proj1, typename Proj2>
        static inline InIter1 call(InIter1 first, InIter1 last, InIter2 s_first,
            InIter2 s_last, Pred&& op, Proj1&& proj1, Proj2&& proj2)
        {
            if (first == last)
                return last;

            std::size_t count = std::distance(first, last);
            util::cancellation_token<std::size_t> tok(count);

            call(first, s_first, s_last, 0, count, tok, HPX_FORWARD(Pred, op),
                HPX_FORWARD(Proj1, proj1), HPX_FORWARD(Proj2, proj2));

            std::size_t find_first_of_res = tok.get_data();

            if (find_first_of_res != count)
                std::advance(first, find_first_of_res);
            else
                first = last;

            return first;
        }

        template <typename FwdIter, typename FwdIter2, typename Token,
            typename Pred, typename Proj1, typename Proj2>
        static inline void call(FwdIter it, FwdIter2 s_first, FwdIter2 s_last,
            std::size_t base_idx, std::size_t part_size, Token& tok, Pred&& op,
            Proj1&& proj1, Proj2&& proj2)
        {
            std::size_t idx = 0;
            util::loop_idx_n<hpx::execution::sequenced_policy>(base_idx, it,
                part_size, tok,
                [&it, &proj1, &s_first, &s_last, &proj2, &op, &tok, &idx](
                    auto, std::size_t i) {
                    auto val = *hpx::invoke(proj1, it + idx);

                    util::cancellation_token<> local_tok;
                    util::loop_n<hpx::execution::simd_policy>(s_first,
                        std::distance(s_first, s_last), local_tok,
                        [&local_tok, &proj2, &op, &val](auto curr) {
                            auto msk =
                                HPX_INVOKE(op, val, HPX_INVOKE(proj2, *curr));
                            if (hpx::parallel::traits::any_of(msk))
                            {
                                local_tok.cancel();
                            }
                        });

                    if (local_tok.was_cancelled())
                        tok.cancel(i);
                    ++idx;
                });
        }
    };

    template <typename ExPolicy, typename InIter1, typename InIter2,
        typename Pred, typename Proj1, typename Proj2,
        HPX_CONCEPT_REQUIRES_(hpx::is_vectorpack_execution_policy_v<ExPolicy>)>
    HPX_HOST_DEVICE HPX_FORCEINLINE InIter1 tag_invoke(
        sequential_find_first_of_t<ExPolicy>, InIter1 first, InIter1 last,
        InIter2 s_first, InIter2 s_last, Pred&& op, Proj1&& proj1,
        Proj2&& proj2)
    {
        if constexpr ((hpx::parallel::util::detail::iterator_datapar_compatible<
                           InIter1>::value &&
                          hpx::parallel::util::detail::
                              iterator_datapar_compatible_v<InIter2>) )
        {
            return datapar_find_first_of<std::decay_t<ExPolicy>>::call(first,
                last, s_first, s_last, HPX_FORWARD(Pred, op),
                HPX_FORWARD(Proj1, proj1), HPX_FORWARD(Proj2, proj2));
        }
        else
        {
            using base_policy_type =
                decltype((hpx::execution::experimental::to_non_simd(
                    std::declval<ExPolicy>())));
            return sequential_find_first_of<base_policy_type>(first, last,
                s_first, s_last, HPX_FORWARD(Pred, op),
                HPX_FORWARD(Proj1, proj1), HPX_FORWARD(Proj2, proj2));
        }
    }

    template <typename ExPolicy, typename FwdIter, typename FwdIter2,
        typename Token, typename Pred, typename Proj1, typename Proj2,
        HPX_CONCEPT_REQUIRES_(hpx::is_vectorpack_execution_policy_v<ExPolicy>)>
    HPX_HOST_DEVICE HPX_FORCEINLINE void tag_invoke(
        sequential_find_first_of_t<ExPolicy>, FwdIter it, FwdIter2 s_first,
        FwdIter2 s_last, std::size_t base_idx, std::size_t part_size,
        Token& tok, Pred&& op, Proj1&& proj1, Proj2&& proj2)
    {
        if constexpr (hpx::parallel::util::detail::iterator_datapar_compatible<
                          FwdIter>::value &&
            hpx::parallel::util::detail::iterator_datapar_compatible<
                FwdIter2>::value)
        {
            return datapar_find_first_of<ExPolicy>::call(it, s_first, s_last,
                base_idx, part_size, tok, HPX_FORWARD(Pred, op),
                HPX_FORWARD(Proj1, proj1), HPX_FORWARD(Proj2, proj2));
        }
        else
        {
            using base_policy_type =
                decltype((hpx::execution::experimental::to_non_simd(
                    std::declval<ExPolicy>())));
            return sequential_find_first_of<base_policy_type>(it, s_first,
                s_last, base_idx, part_size, tok, HPX_FORWARD(Pred, op),
                HPX_FORWARD(Proj1, proj1), HPX_FORWARD(Proj2, proj2));
        }
    }
}}}    // namespace hpx::parallel::detail
#endif
