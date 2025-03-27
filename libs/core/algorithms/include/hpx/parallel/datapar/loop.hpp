//  Copyright (c) 2007-2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_DATAPAR)
#include <hpx/execution/traits/is_execution_policy.hpp>
#include <hpx/execution/traits/vector_pack_alignment_size.hpp>
#include <hpx/execution/traits/vector_pack_get_set.hpp>
#include <hpx/execution/traits/vector_pack_load_store.hpp>
#include <hpx/execution/traits/vector_pack_type.hpp>
#include <hpx/executors/datapar/execution_policy.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/parallel/datapar/iterator_helpers.hpp>
#include <hpx/parallel/util/loop.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <type_traits>
#include <utility>

namespace hpx::parallel::util {

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {

        ///////////////////////////////////////////////////////////////////////
        // Helper class to repeatedly call a function starting from a given
        // iterator position.
        template <typename Iterator>
        struct datapar_loop
        {
            using iterator_type = std::decay_t<Iterator>;
            using value_type =
                typename std::iterator_traits<iterator_type>::value_type;

            using V = traits::vector_pack_type_t<value_type>;

            template <typename Begin, typename End, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static constexpr Begin call(
                Begin first, End last, F&& f)
            {
                constexpr bool datapar_compatible =
                    iterator_datapar_compatible_v<Begin>;

                if constexpr (datapar_compatible)
                {
                    while (!is_data_aligned(first) && first != last)
                    {
                        datapar_loop_step<Begin>::call1(f, first);
                    }

                    constexpr std::size_t size = traits::vector_pack_size_v<V>;

                    End const lastV = last - (size + 1);
                    while (first < lastV)
                    {
                        datapar_loop_step<Begin>::callv(f, first);
                    }

                    while (first != last)
                    {
                        datapar_loop_step<Begin>::call1(f, first);
                    }

                    return first;
                }
                else
                {
                    while (first != last)
                    {
                        datapar_loop_step<Begin>::call1(f, first);
                    }
                    return first;
                }
            }

            template <typename Begin, typename End, typename CancelToken,
                typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static constexpr Begin call(
                Begin first, End last, CancelToken& tok, F&& f)
            {
                // check at the start of a partition only
                if (tok.was_cancelled())
                    return first;

                return call(first, last, HPX_FORWARD(F, f));
            }
        };

        ///////////////////////////////////////////////////////////////////////
        // Helper class to repeatedly call a function starting from a given
        // iterator position till the predicate returns true.
        template <typename Iterator>
        struct datapar_loop_pred
        {
            using iterator_type = std::decay_t<Iterator>;
            using value_type =
                typename std::iterator_traits<iterator_type>::value_type;

            using V = traits::vector_pack_type_t<value_type>;

            template <typename Begin, typename End, typename Pred>
            HPX_HOST_DEVICE HPX_FORCEINLINE static constexpr Begin call(
                Begin first, End last, Pred&& pred)
            {
                while (!is_data_aligned(first) && first != last)
                {
                    if (datapar_loop_pred_step<Begin>::call1(pred, first) != -1)
                        return first;
                    ++first;
                }

                constexpr std::size_t size = traits::vector_pack_size_v<V>;

                End const lastV = last - (size + 1);

                while (first < lastV)
                {
                    int offset =
                        datapar_loop_pred_step<Begin>::callv(pred, first);
                    if (offset != -1)
                    {
                        std::advance(first, offset);
                        return first;
                    }
                    std::advance(first, size);
                }

                while (first != last)
                {
                    if (datapar_loop_pred_step<Begin>::call1(pred, first) != -1)
                        return first;
                    ++first;
                }

                return first;
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Iterator>
        struct datapar_loop_ind
        {
            using iterator_type = std::decay_t<Iterator>;
            using value_type =
                typename std::iterator_traits<iterator_type>::value_type;

            using V = traits::vector_pack_type_t<value_type>;

            template <typename Begin, typename End, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static constexpr Begin call(
                Begin first, End last, F&& f)
            {
                constexpr bool datapar_compatible =
                    iterator_datapar_compatible_v<Begin>;

                if constexpr (datapar_compatible)
                {
                    while (!is_data_aligned(first) && first != last)
                    {
                        datapar_loop_step_ind<Begin>::call1(f, first);
                    }

                    constexpr std::size_t size = traits::vector_pack_size_v<V>;

                    End const lastV = last - (size + 1);
                    while (first < lastV)
                    {
                        datapar_loop_step_ind<Begin>::callv(f, first);
                    }

                    while (first != last)
                    {
                        datapar_loop_step_ind<Begin>::call1(f, first);
                    }

                    return first;
                }
                else
                {
                    while (first != last)
                    {
                        datapar_loop_step_ind<Begin>::call1(f, first);
                    }
                    return first;
                }
            }
        };

        ///////////////////////////////////////////////////////////////////////
        struct datapar_loop2
        {
            template <typename InIter1, typename InIter2, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static constexpr std::enable_if_t<
                iterators_datapar_compatible_v<InIter1, InIter2> &&
                    iterator_datapar_compatible_v<InIter1> &&
                    iterator_datapar_compatible_v<InIter2>,
                std::pair<InIter1, InIter2>>
            call(InIter1 it1, InIter1 last1, InIter2 it2, F&& f)
            {
                using iterator_type = std::decay_t<InIter1>;
                using value_type =
                    typename std::iterator_traits<iterator_type>::value_type;

                using V = traits::vector_pack_type_t<value_type>;

                while ((!is_data_aligned(it1) || !is_data_aligned(it2)) &&
                    it1 != last1)
                {
                    datapar_loop_step2_ind<InIter1, InIter2>::call1(
                        f, it1, it2);
                }

                constexpr std::size_t size = traits::vector_pack_size_v<V>;

                InIter1 const last1V = last1 - (size + 1);
                while (it1 < last1V)
                {
                    datapar_loop_step2_ind<InIter1, InIter2>::callv(
                        f, it1, it2);
                }

                while (it1 != last1)
                {
                    datapar_loop_step2_ind<InIter1, InIter2>::call1(
                        f, it1, it2);
                }

                return std::make_pair(HPX_MOVE(it1), HPX_MOVE(it2));
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Iterator, typename Enable = void>
        struct datapar_loop_n;

        template <typename Iterator>
        struct datapar_loop_n<Iterator,
            std::enable_if_t<hpx::traits::is_iterator_v<Iterator>>>
        {
            using iterator_type = std::decay_t<Iterator>;
            using value_type =
                typename std::iterator_traits<iterator_type>::value_type;

            using V = traits::vector_pack_type_t<value_type>;

            template <typename InIter, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static constexpr InIter call(
                InIter first, std::size_t count, F&& f)
            {
                constexpr bool datapar_compatible =
                    iterator_datapar_compatible_v<InIter>;

                if constexpr (datapar_compatible)
                {
                    std::size_t len = count;

                    // clang-format off
                    for (/* */; !detail::is_data_aligned(first) && len != 0;
                        --len)
                    {
                        datapar_loop_step<InIter>::call1(f, first);
                    }

                    constexpr std::size_t size = traits::vector_pack_size_v<V>;

                    for (auto len_v =
                             static_cast<std::int64_t>(len - (size + 1));
                        len_v > 0;
                        len_v -= static_cast<std::int64_t>(size), len -= size)
                    {
                        datapar_loop_step<InIter>::callv(f, first);
                    }
                    // clang-format on

                    for (/* */; len != 0; --len)
                    {
                        datapar_loop_step<InIter>::call1(f, first);
                    }

                    return first;
                }
                else
                {
                    for (/* */; count != 0; --count)
                    {
                        datapar_loop_step<InIter>::call1(f, first);
                    }
                    return first;
                }
            }

            template <typename InIter, typename CancelToken, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static constexpr InIter call(
                InIter first, std::size_t count, CancelToken& tok, F&& f)
            {
                // check at the start of a partition only
                if (tok.was_cancelled())
                    return first;

                return call(first, count, HPX_FORWARD(F, f));
            }
        };

        template <typename I>
        struct datapar_loop_n<I, std::enable_if_t<std::is_integral_v<I>>>
        {
            using V = traits::vector_pack_type_t<I>;

            template <typename Iter, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static constexpr Iter call(
                Iter first, std::size_t count, F&& f)
            {
                std::size_t len = count;
                constexpr std::size_t size = traits::vector_pack_size_v<V>;

                for (size_t i = first % size; i != 0 && len != 0; --i, --len)
                {
                    datapar_loop_step<Iter>::call1(f, first);
                }

                // clang-format off
                for (auto len_v = static_cast<std::int64_t>(len - (size + 1));
                    len_v > 0;
                    len_v -= static_cast<std::int64_t>(size), len -= size)
                {
                    datapar_loop_step<Iter>::callv(f, first);
                }
                // clang-format on

                for (/* */; len != 0; --len)
                {
                    datapar_loop_step<Iter>::call1(f, first);
                }
                return first;
            }

            template <typename Iter, typename CancelToken, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static constexpr Iter call(
                Iter first, std::size_t count, CancelToken& tok, F&& f)
            {
                // check at the start of a partition only
                if (tok.was_cancelled())
                    return first;

                return call(first, count, HPX_FORWARD(F, f));
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Iterator>
        struct datapar_loop_n_ind
        {
            using iterator_type = std::decay_t<Iterator>;
            using value_type =
                typename std::iterator_traits<iterator_type>::value_type;

            using V = traits::vector_pack_type_t<value_type>;

            template <typename InIter, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static constexpr InIter call(
                InIter first, std::size_t count, F&& f)
            {
                constexpr bool datapar_compatible =
                    iterator_datapar_compatible_v<InIter>;

                if constexpr (datapar_compatible)
                {
                    std::size_t len = count;

                    // clang-format off
                    for (/* */; !detail::is_data_aligned(first) && len != 0;
                        --len)
                    {
                        datapar_loop_step_ind<InIter>::call1(f, first);
                    }

                    constexpr std::size_t size = traits::vector_pack_size_v<V>;

                    for (auto len_v =
                             static_cast<std::int64_t>(len - (size + 1));
                        len_v > 0;
                        len_v -= static_cast<std::int64_t>(size), len -= size)
                    {
                        datapar_loop_step_ind<InIter>::callv(f, first);
                    }
                    // clang-format on

                    for (/* */; len != 0; --len)
                    {
                        datapar_loop_step_ind<InIter>::call1(f, first);
                    }
                    return first;
                }
                else
                {
                    for (/* */; count != 0; --count)
                    {
                        datapar_loop_step_ind<InIter>::call1(f, first);
                    }
                    return first;
                }
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Iterator>
        struct datapar_loop_idx_n
        {
            using iterator_type = std::decay_t<Iterator>;
            using value_type =
                typename std::iterator_traits<iterator_type>::value_type;

            using V = traits::vector_pack_type_t<value_type>;

            template <typename Iter, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static constexpr Iter call(
                std::size_t base_idx, Iter it, std::size_t count, F&& f)
            {
                std::size_t len = count;

                for (/* */; !detail::is_data_aligned(it) && len != 0; --len)
                {
                    datapar_loop_idx_step<Iter>::call1(f, it, base_idx);
                    ++it;
                    ++base_idx;
                }

                constexpr std::size_t size = traits::vector_pack_size_v<V>;

                // clang-format off
                for (auto len_v = static_cast<std::int64_t>(len - (size + 1));
                    len_v > 0;
                    len_v -= static_cast<std::int64_t>(size), len -= size)
                {
                    datapar_loop_idx_step<Iter>::callv(f, it, base_idx);
                    std::advance(it, size);
                    base_idx += size;
                }
                // clang-format on

                for (/* */; len != 0; --len)
                {
                    datapar_loop_idx_step<Iter>::call1(f, it, base_idx);
                    ++it;
                    ++base_idx;
                }
                return it;
            }

            template <typename Iter, typename CancelToken, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static constexpr Iter call(
                std::size_t base_idx, Iter it, std::size_t count,
                CancelToken& tok, F&& f)
            {
                if (tok.was_cancelled(base_idx))
                    return it;

                return call(base_idx, it, count, HPX_FORWARD(F, f));
            }
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    template <typename Begin, typename End, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE constexpr Begin tag_invoke(
        hpx::parallel::util::loop_t, hpx::execution::simd_policy, Begin begin,
        End end, F&& f)
    {
        return detail::datapar_loop<Begin>::call(begin, end, HPX_FORWARD(F, f));
    }

    template <typename Begin, typename End, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE constexpr Begin tag_invoke(
        hpx::parallel::util::loop_t, hpx::execution::simd_task_policy,
        Begin begin, End end, F&& f)
    {
        return detail::datapar_loop<Begin>::call(begin, end, HPX_FORWARD(F, f));
    }

    template <typename Begin, typename End, typename CancelToken, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE constexpr Begin tag_invoke(
        hpx::parallel::util::loop_t, hpx::execution::simd_policy, Begin begin,
        End end, CancelToken& tok, F&& f)
    {
        return detail::datapar_loop<Begin>::call(
            begin, end, tok, HPX_FORWARD(F, f));
    }

    template <typename Begin, typename End, typename CancelToken, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE constexpr Begin tag_invoke(
        hpx::parallel::util::loop_t, hpx::execution::simd_task_policy,
        Begin begin, End end, CancelToken& tok, F&& f)
    {
        return detail::datapar_loop<Begin>::call(
            begin, end, tok, HPX_FORWARD(F, f));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename ExPolicy, typename Begin, typename End, typename Pred>
    HPX_HOST_DEVICE HPX_FORCEINLINE constexpr std::enable_if_t<
        hpx::is_vectorpack_execution_policy_v<ExPolicy>, Begin>
    tag_invoke(hpx::parallel::util::loop_pred_t<ExPolicy>, Begin first, End end,
        Pred&& pred)
    {
        constexpr bool datapar_compatible =
            detail::iterator_datapar_compatible_v<Begin>;

        if constexpr (datapar_compatible)
        {
            return hpx::parallel::util::detail::datapar_loop_pred<Begin>::call(
                first, end, HPX_FORWARD(Pred, pred));
        }
        else
        {
            using base_policy_type =
                decltype(hpx::execution::experimental::to_non_simd(
                    std::declval<ExPolicy>()));

            return loop_pred<base_policy_type>(
                first, end, HPX_FORWARD(Pred, pred));
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Begin, typename End, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE Begin tag_invoke(
        hpx::parallel::util::loop_ind_t<hpx::execution::simd_policy>,
        Begin begin, End end, F&& f)
    {
        return detail::datapar_loop_ind<Begin>::call(
            begin, end, HPX_FORWARD(F, f));
    }

    template <typename Begin, typename End, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE Begin tag_invoke(
        hpx::parallel::util::loop_ind_t<hpx::execution::simd_task_policy>,
        Begin begin, End end, F&& f)
    {
        return detail::datapar_loop_ind<Begin>::call(
            begin, end, HPX_FORWARD(F, f));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename ExPolicy, typename Iter1, typename Iter2, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE
        std::enable_if_t<hpx::is_vectorpack_execution_policy_v<ExPolicy>,
            std::pair<Iter1, Iter2>>
        tag_invoke(hpx::parallel::util::loop2_t<ExPolicy>, Iter1 first1,
            Iter1 last1, Iter2 first2, F&& f)
    {
        if constexpr (detail::iterator_datapar_compatible_v<Iter1> &&
            detail::iterator_datapar_compatible_v<Iter2>)
        {
            return detail::datapar_loop2::call(
                first1, last1, first2, HPX_FORWARD(F, f));
        }
        else
        {
            using base_policy_type =
                decltype(hpx::execution::experimental::to_non_simd(
                    std::declval<ExPolicy>()));

            return loop2<base_policy_type>(
                first1, last1, first2, HPX_FORWARD(F, f));
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename ExPolicy, typename Iter, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE constexpr std::enable_if_t<
        hpx::is_vectorpack_execution_policy_v<ExPolicy>, Iter>
    tag_invoke(hpx::parallel::util::loop_n_t<ExPolicy>, Iter it,
        std::size_t count, F&& f)
    {
        return hpx::parallel::util::detail::datapar_loop_n<Iter>::call(
            it, count, HPX_FORWARD(F, f));
    }

    template <typename ExPolicy, typename Iter, typename CancelToken,
        typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE constexpr std::enable_if_t<
        hpx::is_vectorpack_execution_policy_v<ExPolicy>, Iter>
    tag_invoke(hpx::parallel::util::loop_n_t<ExPolicy>, Iter it,
        std::size_t count, CancelToken& tok, F&& f)
    {
        return hpx::parallel::util::detail::datapar_loop_n<Iter>::call(
            it, count, tok, HPX_FORWARD(F, f));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename ExPolicy, typename Iter, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE constexpr std::enable_if_t<
        hpx::is_vectorpack_execution_policy_v<ExPolicy>, Iter>
    tag_invoke(hpx::parallel::util::loop_n_ind_t<ExPolicy>, Iter it,
        std::size_t count, F&& f)
    {
        return hpx::parallel::util::detail::datapar_loop_n_ind<Iter>::call(
            it, count, HPX_FORWARD(F, f));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename ExPolicy, typename Iter, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE constexpr std::enable_if_t<
        hpx::is_vectorpack_execution_policy_v<ExPolicy>, Iter>
    tag_invoke(hpx::parallel::util::loop_idx_n_t<ExPolicy>,
        std::size_t base_idx, Iter it, std::size_t count, F&& f)
    {
        return hpx::parallel::util::detail::datapar_loop_idx_n<Iter>::call(
            base_idx, it, count, HPX_FORWARD(F, f));
    }

    template <typename ExPolicy, typename Iter, typename CancelToken,
        typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE constexpr std::enable_if_t<
        hpx::is_vectorpack_execution_policy_v<ExPolicy>, Iter>
    tag_invoke(hpx::parallel::util::loop_idx_n_t<ExPolicy>,
        std::size_t base_idx, Iter it, std::size_t count, CancelToken& tok,
        F&& f)
    {
        return hpx::parallel::util::detail::datapar_loop_idx_n<Iter>::call(
            base_idx, it, count, tok, HPX_FORWARD(F, f));
    }
}    // namespace hpx::parallel::util

#endif
