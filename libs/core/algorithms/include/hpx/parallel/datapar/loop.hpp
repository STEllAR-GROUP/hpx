//  Copyright (c) 2007-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_DATAPAR)
#include <hpx/execution/algorithms/detail/predicates.hpp>
#include <hpx/execution/traits/is_execution_policy.hpp>
#include <hpx/execution/traits/vector_pack_alignment_size.hpp>
#include <hpx/execution/traits/vector_pack_load_store.hpp>
#include <hpx/execution/traits/vector_pack_type.hpp>
#include <hpx/executors/datapar/execution_policy.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/functional/tag_invoke.hpp>
#include <hpx/parallel/datapar/iterator_helpers.hpp>
#include <hpx/parallel/util/loop.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { namespace util {

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {

        ///////////////////////////////////////////////////////////////////////
        template <typename ExPolicy, typename F, typename Vector>
        HPX_HOST_DEVICE HPX_FORCEINLINE typename std::enable_if<
            hpx::is_vectorpack_execution_policy<ExPolicy>::value,
            typename traits::vector_pack_type<
                typename std::decay<Vector>::type::value_type, 1>::type>::type
        tag_invoke(hpx::parallel::util::detail::accumulate_values_t<ExPolicy>,
            F&& f, Vector const& value)
        {
            typedef typename std::decay<Vector>::type vector_type;
            typedef typename vector_type::value_type entry_type;

            entry_type accum = value[0];
            for (size_t i = 1; i != value.size(); ++i)
            {
                accum = f(accum, entry_type(value[i]));
            }

            return
                typename traits::vector_pack_type<entry_type, 1>::type(accum);
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename ExPolicy, typename F, typename Vector, typename T>
        HPX_HOST_DEVICE HPX_FORCEINLINE typename std::enable_if<
            hpx::is_vectorpack_execution_policy<ExPolicy>::value,
            typename traits::vector_pack_type<T, 1>::type>::type
        tag_invoke(hpx::parallel::util::detail::accumulate_values_t<ExPolicy>,
            F&& f, Vector const& value, T accum)
        {
            for (size_t i = 0; i != value.size(); ++i)
            {
                accum = f(accum, T(value[i]));
            }

            return typename traits::vector_pack_type<T, 1>::type(accum);
        }

        ///////////////////////////////////////////////////////////////////////
        // Helper class to repeatedly call a function starting from a given
        // iterator position.
        template <typename Iterator>
        struct datapar_loop
        {
            typedef typename std::decay<Iterator>::type iterator_type;
            typedef typename std::iterator_traits<iterator_type>::value_type
                value_type;

            typedef typename traits::vector_pack_type<value_type>::type V;

            template <typename Begin, typename End, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static typename std::enable_if<
                iterator_datapar_compatible<Begin>::value, Begin>::type
            call(Begin first, End last, F&& f)
            {
                while (!is_data_aligned(first) && first != last)
                {
                    datapar_loop_step<Begin>::call1(f, first);
                }

                static std::size_t constexpr size =
                    traits::vector_pack_size<V>::value;

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

            template <typename Begin, typename End, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static typename std::enable_if<
                !iterator_datapar_compatible<Begin>::value, Begin>::type
            call(Begin it, End end, F&& f)
            {
                while (it != end)
                {
                    datapar_loop_step<Begin>::call1(f, it);
                }
                return it;
            }

            template <typename Begin, typename End, typename CancelToken,
                typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static typename std::enable_if<
                iterator_datapar_compatible<Begin>::value, Begin>::type
            call(Begin first, End last, CancelToken& tok, F&& f)
            {
                while (!is_data_aligned(first) && first != last)
                {
                    datapar_loop_step_tok<Begin>::call1(f, first);
                    if (tok.was_cancelled())
                        return first;
                    ++first;
                }

                static std::size_t constexpr size =
                    traits::vector_pack_size<V>::value;

                End const lastV = last - (size + 1);
                while (first < lastV)
                {
                    std::size_t incr =
                        datapar_loop_step_tok<Begin>::callv(f, first);
                    if (tok.was_cancelled())
                        return first;
                    std::advance(first, incr);
                }

                while (first != last)
                {
                    datapar_loop_step_tok<Begin>::call1(f, first);
                    if (tok.was_cancelled())
                        return first;
                    ++first;
                }

                return first;
            }

            template <typename Begin, typename End, typename CancelToken,
                typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static typename std::enable_if<
                !iterator_datapar_compatible<Begin>::value, Begin>::type
            call(Begin it, End end, CancelToken& tok, F&& f)
            {
                while (it != end)
                {
                    datapar_loop_step_tok<Begin>::call1(f, it);
                    if (tok.was_cancelled())
                        return it;
                    ++it;
                }
                return it;
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Iterator>
        struct datapar_loop_ind
        {
            typedef typename std::decay<Iterator>::type iterator_type;
            typedef typename std::iterator_traits<iterator_type>::value_type
                value_type;

            typedef typename traits::vector_pack_type<value_type>::type V;

            template <typename Begin, typename End, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static typename std::enable_if<
                iterator_datapar_compatible<Begin>::value, Begin>::type
            call(Begin first, End last, F&& f)
            {
                while (!is_data_aligned(first) && first != last)
                {
                    datapar_loop_step_ind<Begin>::call1(f, first);
                }

                static std::size_t constexpr size =
                    traits::vector_pack_size<V>::value;

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

            template <typename Begin, typename End, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static typename std::enable_if<
                !iterator_datapar_compatible<Begin>::value, Begin>::type
            call(Begin it, End end, F&& f)
            {
                while (it != end)
                {
                    datapar_loop_step_ind<Begin>::call1(f, it);
                }
                return it;
            }
        };

        ///////////////////////////////////////////////////////////////////////
        struct datapar_loop2
        {
            template <typename InIter1, typename InIter2, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static typename std::enable_if<
                iterators_datapar_compatible<InIter1, InIter2>::value &&
                    iterator_datapar_compatible<InIter1>::value &&
                    iterator_datapar_compatible<InIter2>::value,
                std::pair<InIter1, InIter2>>::type
            call(InIter1 it1, InIter1 last1, InIter2 it2, F&& f)
            {
                typedef typename std::decay<InIter1>::type iterator_type;
                typedef typename std::iterator_traits<iterator_type>::value_type
                    value_type;

                typedef typename traits::vector_pack_type<value_type>::type V;

                while ((!is_data_aligned(it1) || !is_data_aligned(it2)) &&
                    it1 != last1)
                {
                    datapar_loop_step2_ind<InIter1, InIter2>::call1(
                        f, it1, it2);
                }

                static std::size_t constexpr size =
                    traits::vector_pack_size<V>::value;

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
        template <typename Iterator>
        struct datapar_loop_n
        {
            typedef typename std::decay<Iterator>::type iterator_type;
            typedef typename std::iterator_traits<iterator_type>::value_type
                value_type;

            typedef typename traits::vector_pack_type<value_type>::type V;

            template <typename InIter, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static typename std::enable_if<
                iterator_datapar_compatible<InIter>::value, InIter>::type
            call(InIter first, std::size_t count, F&& f)
            {
                std::size_t len = count;

                for (/* */; !detail::is_data_aligned(first) && len != 0; --len)
                {
                    datapar_loop_step<InIter>::call1(f, first);
                }

                static std::size_t constexpr size =
                    traits::vector_pack_size<V>::value;

                for (std::int64_t len_v = std::int64_t(len - (size + 1));
                     len_v > 0; len_v -= size, len -= size)
                {
                    datapar_loop_step<InIter>::callv(f, first);
                }

                for (/* */; len != 0; --len)
                {
                    datapar_loop_step<InIter>::call1(f, first);
                }

                return first;
            }

            template <typename InIter, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static typename std::enable_if<
                !iterator_datapar_compatible<InIter>::value, InIter>::type
            call(InIter first, std::size_t count, F&& f)
            {
                for (/* */; count != 0; --count)
                {
                    datapar_loop_step<InIter>::call1(f, first);
                }
                return first;
            }

            template <typename InIter, typename CancelToken, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static typename std::enable_if<
                iterator_datapar_compatible<InIter>::value, InIter>::type
            call(InIter first, std::size_t count, CancelToken& tok, F&& f)
            {
                std::size_t len = count;

                for (/* */; !detail::is_data_aligned(first) && len != 0; --len)
                {
                    datapar_loop_step_tok<InIter>::call1(f, first);
                    if (tok.was_cancelled())
                        return first;
                    ++first;
                }

                static std::size_t constexpr size =
                    traits::vector_pack_size<V>::value;

                for (std::int64_t len_v = std::int64_t(len - (size + 1));
                     len_v > 0; len_v -= size, len -= size)
                {
                    std::size_t incr =
                        datapar_loop_step_tok<InIter>::callv(f, first);
                    if (tok.was_cancelled())
                        return first;
                    std::advance(first, incr);
                }

                for (/* */; len != 0; --len)
                {
                    datapar_loop_step_tok<InIter>::call1(f, first);
                    if (tok.was_cancelled())
                        return first;
                    ++first;
                }

                return first;
            }

            template <typename InIter, typename CancelToken, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static typename std::enable_if<
                !iterator_datapar_compatible<InIter>::value, InIter>::type
            call(InIter first, std::size_t count, CancelToken& tok, F&& f)
            {
                for (/* */; count != 0; --count)
                {
                    datapar_loop_step_tok<InIter>::call1(f, first);
                    if (tok.was_cancelled())
                        return first;
                    ++first;
                }
                return first;
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Iterator>
        struct datapar_loop_n_ind
        {
            typedef typename std::decay<Iterator>::type iterator_type;
            typedef typename std::iterator_traits<iterator_type>::value_type
                value_type;

            typedef typename traits::vector_pack_type<value_type>::type V;

            template <typename InIter, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static typename std::enable_if<
                iterator_datapar_compatible<InIter>::value, InIter>::type
            call(InIter first, std::size_t count, F&& f)
            {
                std::size_t len = count;

                for (/* */; !detail::is_data_aligned(first) && len != 0; --len)
                {
                    datapar_loop_step_ind<InIter>::call1(f, first);
                }

                static std::size_t constexpr size =
                    traits::vector_pack_size<V>::value;

                for (std::int64_t len_v = std::int64_t(len - (size + 1));
                     len_v > 0; len_v -= size, len -= size)
                {
                    datapar_loop_step_ind<InIter>::callv(f, first);
                }

                for (/* */; len != 0; --len)
                {
                    datapar_loop_step_ind<InIter>::call1(f, first);
                }

                return first;
            }

            template <typename InIter, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static typename std::enable_if<
                !iterator_datapar_compatible<InIter>::value, InIter>::type
            call(InIter first, std::size_t count, F&& f)
            {
                for (/* */; count != 0; --count)
                {
                    datapar_loop_step_ind<InIter>::call1(f, first);
                }
                return first;
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Iterator>
        struct datapar_loop_idx_n
        {
            typedef typename std::decay<Iterator>::type iterator_type;
            typedef typename std::iterator_traits<iterator_type>::value_type
                value_type;

            typedef typename traits::vector_pack_type<value_type>::type V;

            template <typename Iter, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static Iter call(
                std::size_t base_idx, Iter it, std::size_t count, F&& f)
            {
                std::size_t len = count;

                for (/* */; !detail::is_data_aligned(it) && len != 0; --len)
                {
                    datapar_loop_idx_step<Iter>::call1(f, it, base_idx);
                    ++it;
                    ++base_idx;
                }

                static std::size_t constexpr size =
                    traits::vector_pack_size<V>::value;

                for (std::int64_t len_v = std::int64_t(len - (size + 1));
                     len_v > 0; len_v -= size, len -= size)
                {
                    std::size_t incr =
                        datapar_loop_idx_step<Iter>::callv(f, it, base_idx);
                    std::advance(it, incr);
                    base_idx += incr;
                }

                for (/* */; len != 0; --len)
                {
                    datapar_loop_idx_step<Iter>::call1(f, it, base_idx);
                    ++it;
                    ++base_idx;
                }
                return it;
            }

            template <typename Iter, typename CancelToken, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static Iter call(
                std::size_t base_idx, Iter it, std::size_t count,
                CancelToken& tok, F&& f)
            {
                std::size_t len = count;

                for (/* */; !detail::is_data_aligned(it) && len != 0; --len)
                {
                    datapar_loop_idx_step<Iter>::call1(f, it, base_idx);
                    if (tok.was_cancelled(base_idx))
                        return it;
                    ++it;
                    ++base_idx;
                }

                static std::size_t constexpr size =
                    traits::vector_pack_size<V>::value;

                for (std::int64_t len_v = std::int64_t(len - (size + 1));
                     len_v > 0; len_v -= size, len -= size)
                {
                    datapar_loop_idx_step<Iter>::callv(f, it, base_idx);
                    if (tok.was_cancelled(base_idx))
                        return it;
                    std::advance(it, size);
                    base_idx += size;
                }

                for (/* */; len != 0; --len)
                {
                    datapar_loop_idx_step<Iter>::call1(f, it, base_idx);
                    if (tok.was_cancelled(base_idx))
                        return it;
                    ++it;
                    ++base_idx;
                }
                return it;
            }
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    template <typename Begin, typename End, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE Begin tag_invoke(
        hpx::parallel::util::loop_t, hpx::execution::simd_policy, Begin begin,
        End end, F&& f)
    {
        return detail::datapar_loop<Begin>::call(begin, end, HPX_FORWARD(F, f));
    }

    template <typename Begin, typename End, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE Begin tag_invoke(
        hpx::parallel::util::loop_t, hpx::execution::simd_task_policy,
        Begin begin, End end, F&& f)
    {
        return detail::datapar_loop<Begin>::call(begin, end, HPX_FORWARD(F, f));
    }

    template <typename Begin, typename End, typename CancelToken, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE Begin tag_invoke(
        hpx::parallel::util::loop_t, hpx::execution::simd_policy, Begin begin,
        End end, CancelToken& tok, F&& f)
    {
        return detail::datapar_loop<Begin>::call(
            begin, end, tok, HPX_FORWARD(F, f));
    }

    template <typename Begin, typename End, typename CancelToken, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE Begin tag_invoke(
        hpx::parallel::util::loop_t, hpx::execution::simd_task_policy,
        Begin begin, End end, CancelToken& tok, F&& f)
    {
        return detail::datapar_loop<Begin>::call(
            begin, end, tok, HPX_FORWARD(F, f));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Begin, typename End, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE Begin tag_invoke(
        hpx::parallel::util::loop_ind_t, hpx::execution::simd_policy,
        Begin begin, End end, F&& f)
    {
        return detail::datapar_loop_ind<Begin>::call(
            begin, end, HPX_FORWARD(F, f));
    }

    template <typename Begin, typename End, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE Begin tag_invoke(
        hpx::parallel::util::loop_ind_t, hpx::execution::simd_task_policy,
        Begin begin, End end, F&& f)
    {
        return detail::datapar_loop_ind<Begin>::call(
            begin, end, HPX_FORWARD(F, f));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename ExPolicy, typename Iter1, typename Iter2, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE typename std::enable_if<
        hpx::is_vectorpack_execution_policy<ExPolicy>::value,
        std::pair<Iter1, Iter2>>::type
    tag_invoke(hpx::parallel::util::loop2_t<ExPolicy>, Iter1 first1,
        Iter1 last1, Iter2 first2, F&& f)
    {
        if constexpr (hpx::parallel::util::detail::iterator_datapar_compatible<
                          Iter1>::value &&
            hpx::parallel::util::detail::iterator_datapar_compatible<
                Iter2>::value)
        {
            return detail::datapar_loop2::call(
                first1, last1, first2, HPX_FORWARD(F, f));
        }
        else
        {
            using execution_policy_type = typename std::decay_t<ExPolicy>;
            using base_policy_type =
                typename execution_policy_type::base_policy_type;
            return loop2<base_policy_type>(
                first1, last1, first2, HPX_FORWARD(F, f));
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename ExPolicy, typename Iter, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE constexpr typename std::enable_if<
        hpx::is_vectorpack_execution_policy<ExPolicy>::value, Iter>::type
    tag_invoke(hpx::parallel::util::loop_n_t<ExPolicy>, Iter it,
        std::size_t count, F&& f)
    {
        return hpx::parallel::util::detail::datapar_loop_n<Iter>::call(
            it, count, HPX_FORWARD(F, f));
    }

    template <typename ExPolicy, typename Iter, typename CancelToken,
        typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE constexpr typename std::enable_if<
        hpx::is_vectorpack_execution_policy<ExPolicy>::value, Iter>::type
    tag_invoke(hpx::parallel::util::loop_n_t<ExPolicy>, Iter it,
        std::size_t count, CancelToken& tok, F&& f)
    {
        return hpx::parallel::util::detail::datapar_loop_n<Iter>::call(
            it, count, tok, HPX_FORWARD(F, f));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename ExPolicy, typename Iter, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE constexpr typename std::enable_if<
        hpx::is_vectorpack_execution_policy<ExPolicy>::value, Iter>::type
    tag_invoke(hpx::parallel::util::loop_n_ind_t<ExPolicy>, Iter it,
        std::size_t count, F&& f)
    {
        return hpx::parallel::util::detail::datapar_loop_n_ind<Iter>::call(
            it, count, HPX_FORWARD(F, f));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename ExPolicy, typename Iter, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE constexpr typename std::enable_if<
        hpx::is_vectorpack_execution_policy<ExPolicy>::value, Iter>::type
    tag_invoke(hpx::parallel::util::loop_idx_n_t<ExPolicy>,
        std::size_t base_idx, Iter it, std::size_t count, F&& f)
    {
        return hpx::parallel::util::detail::datapar_loop_idx_n<Iter>::call(
            base_idx, it, count, HPX_FORWARD(F, f));
    }

    template <typename ExPolicy, typename Iter, typename CancelToken,
        typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE constexpr typename std::enable_if<
        hpx::is_vectorpack_execution_policy<ExPolicy>::value, Iter>::type
    tag_invoke(hpx::parallel::util::loop_idx_n_t<ExPolicy>,
        std::size_t base_idx, Iter it, std::size_t count, CancelToken& tok,
        F&& f)
    {
        return hpx::parallel::util::detail::datapar_loop_idx_n<Iter>::call(
            base_idx, it, count, tok, HPX_FORWARD(F, f));
    }
}}}    // namespace hpx::parallel::util

#endif
