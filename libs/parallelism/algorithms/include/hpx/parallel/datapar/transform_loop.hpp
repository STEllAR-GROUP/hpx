//  Copyright (c) 2007-2018 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_DATAPAR)
#include <hpx/datastructures/tuple.hpp>
#include <hpx/execution/traits/is_execution_policy.hpp>
#include <hpx/executors/datapar/execution_policy_fwd.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/functional/invoke.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/parallel/datapar/iterator_helpers.hpp>
#include <hpx/parallel/util/cancellation_token.hpp>
#include <hpx/parallel/util/transform_loop.hpp>
#include <hpx/type_support/decay.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { namespace util {
    namespace detail {
        ///////////////////////////////////////////////////////////////////////
        template <typename Iterator>
        struct datapar_transform_loop_n
        {
            typedef typename hpx::util::decay<Iterator>::type iterator_type;

            typedef typename traits::vector_pack_type<
                typename std::iterator_traits<iterator_type>::value_type>::type
                V;

            template <typename InIter, typename OutIter, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static typename std::enable_if<
                iterators_datapar_compatible<InIter, OutIter>::value &&
                    iterator_datapar_compatible<InIter>::value &&
                    iterator_datapar_compatible<OutIter>::value,
                std::pair<InIter, OutIter>>::type
            call(InIter first, std::size_t count, OutIter dest, F&& f)
            {
                std::size_t len = count;

                for (/* */; is_data_aligned(first) && len != 0; --len)
                {
                    datapar_transform_loop_step::call1(f, first, dest);
                }

                static constexpr std::size_t size =
                    traits::vector_pack_size<V>::value;

                for (std::int64_t lenV = std::int64_t(count - (size + 1));
                     lenV > 0; lenV -= size, len -= size)
                {
                    datapar_transform_loop_step::callv(f, first, dest);
                }

                for (/* */; len != 0; --len)
                {
                    datapar_transform_loop_step::call1(f, first, dest);
                }

                return std::make_pair(std::move(first), std::move(dest));
            }

            template <typename InIter, typename OutIter, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static typename std::enable_if<
                !iterators_datapar_compatible<InIter, OutIter>::value ||
                    !iterator_datapar_compatible<InIter>::value ||
                    !iterator_datapar_compatible<OutIter>::value,
                std::pair<InIter, OutIter>>::type
            call(InIter first, std::size_t count, OutIter dest, F&& f)
            {
                return util::transform_loop_n<sequenced_policy>(
                    first, count, dest, std::forward<F>(f));
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Iterator>
        struct datapar_transform_loop
        {
            typedef typename hpx::util::decay<Iterator>::type iterator_type;
            typedef typename std::iterator_traits<iterator_type>::value_type
                value_type;

            typedef typename traits::vector_pack_type<value_type>::type V;
            typedef typename traits::vector_pack_type<value_type, 1>::type V1;

            template <typename InIter, typename OutIter, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static typename std::enable_if<
                iterators_datapar_compatible<InIter, OutIter>::value &&
                    iterator_datapar_compatible<InIter>::value &&
                    iterator_datapar_compatible<OutIter>::value,
                std::pair<InIter, OutIter>>::type
            call(InIter first, InIter last, OutIter dest, F&& f)
            {
                return util::transform_loop_n<
                    parallel::execution::datapar_policy>(first,
                    std::distance(first, last), dest, std::forward<F>(f));
            }

            template <typename InIter, typename OutIter, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static typename std::enable_if<
                !iterators_datapar_compatible<InIter, OutIter>::value ||
                    !iterator_datapar_compatible<InIter>::value ||
                    !iterator_datapar_compatible<OutIter>::value,
                std::pair<InIter, OutIter>>::type
            call(InIter first, InIter last, OutIter dest, F&& f)
            {
                return util::transform_loop(
                    seq, first, last, dest, std::forward<F>(f));
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Iter1, typename Iter2>
        struct datapar_transform_binary_loop_n
        {
            typedef typename hpx::util::decay<Iter1>::type iterator1_type;
            typedef typename std::iterator_traits<iterator1_type>::value_type
                value_type;

            typedef typename traits::vector_pack_type<value_type>::type V;

            template <typename InIter1, typename InIter2, typename OutIter,
                typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static typename std::enable_if<
                iterators_datapar_compatible<InIter1, OutIter>::value &&
                    iterators_datapar_compatible<InIter2, OutIter>::value &&
                    iterator_datapar_compatible<InIter1>::value &&
                    iterator_datapar_compatible<InIter2>::value &&
                    iterator_datapar_compatible<OutIter>::value,
                hpx::tuple<InIter1, InIter2, OutIter>>::type
            call(InIter1 first1, std::size_t count, InIter2 first2,
                OutIter dest, F&& f)
            {
                std::size_t len = count;

                for (/* */; is_data_aligned(first1) && len != 0; --len)
                {
                    datapar_transform_loop_step::call1(f, first1, first2, dest);
                }

                static constexpr std::size_t size =
                    traits::vector_pack_size<V>::value;

                for (std::int64_t lenV = std::int64_t(count - (size + 1));
                     lenV > 0; lenV -= size, len -= size)
                {
                    datapar_transform_loop_step::callv(f, first1, first2, dest);
                }

                for (/* */; len != 0; --len)
                {
                    datapar_transform_loop_step::call1(f, first1, first2, dest);
                }

                return hpx::make_tuple(
                    std::move(first1), std::move(first2), std::move(dest));
            }

            template <typename InIter1, typename InIter2, typename OutIter,
                typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static typename std::enable_if<
                !iterators_datapar_compatible<InIter1, OutIter>::value ||
                    !iterators_datapar_compatible<InIter2, OutIter>::value ||
                    !iterator_datapar_compatible<InIter1>::value ||
                    !iterator_datapar_compatible<InIter2>::value ||
                    !iterator_datapar_compatible<OutIter>::value,
                hpx::tuple<InIter1, InIter2, OutIter>>::type
            call(InIter1 first1, std::size_t count, InIter2 first2,
                OutIter dest, F&& f)
            {
                return util::transform_binary_loop_n<sequenced_policy>(
                    first1, count, first2, dest, std::forward<F>(f));
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Iter1, typename Iter2>
        struct datapar_transform_binary_loop
        {
            typedef typename hpx::util::decay<Iter1>::type iterator1_type;
            typedef typename hpx::util::decay<Iter2>::type iterator2_type;

            typedef typename std::iterator_traits<iterator1_type>::value_type
                value1_type;
            typedef typename std::iterator_traits<iterator2_type>::value_type
                value2_type;

            typedef typename traits::vector_pack_type<value1_type, 1>::type V11;
            typedef typename traits::vector_pack_type<value2_type, 1>::type V12;

            typedef typename traits::vector_pack_type<value1_type>::type V1;
            typedef typename traits::vector_pack_type<value2_type>::type V2;

            template <typename InIter1, typename InIter2, typename OutIter,
                typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static typename std::enable_if<
                iterators_datapar_compatible<InIter1, OutIter>::value &&
                    iterators_datapar_compatible<InIter2, OutIter>::value &&
                    iterator_datapar_compatible<InIter1>::value &&
                    iterator_datapar_compatible<InIter2>::value &&
                    iterator_datapar_compatible<OutIter>::value,
                hpx::tuple<InIter1, InIter2, OutIter>>::type
            call(InIter1 first1, InIter1 last1, InIter2 first2, OutIter dest,
                F&& f)
            {
                return util::transform_binary_loop_n<
                    parallel::execution::datapar_policy>(first1,
                    std::distance(first1, last1), first2, dest,
                    std::forward<F>(f));
            }

            template <typename InIter1, typename InIter2, typename OutIter,
                typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static typename std::enable_if<
                !iterators_datapar_compatible<InIter1, OutIter>::value ||
                    !iterators_datapar_compatible<InIter2, OutIter>::value ||
                    !iterator_datapar_compatible<InIter1>::value ||
                    !iterator_datapar_compatible<InIter2>::value ||
                    !iterator_datapar_compatible<OutIter>::value,
                hpx::tuple<InIter1, InIter2, OutIter>>::type
            call(InIter1 first1, InIter1 last1, InIter2 first2, OutIter dest,
                F&& f)
            {
                return util::transform_binary_loop<sequenced_policy>(
                    first1, last1, first2, dest, std::forward<F>(f));
            }

            template <typename InIter1, typename InIter2, typename OutIter,
                typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static typename std::enable_if<
                iterators_datapar_compatible<InIter1, OutIter>::value &&
                    iterators_datapar_compatible<InIter2, OutIter>::value &&
                    iterator_datapar_compatible<InIter1>::value &&
                    iterator_datapar_compatible<InIter2>::value &&
                    iterator_datapar_compatible<OutIter>::value,
                hpx::tuple<InIter1, InIter2, OutIter>>::type
            call(InIter1 first1, InIter1 last1, InIter2 first2, InIter2 last2,
                OutIter dest, F&& f)
            {
                std::size_t count = (std::min)(
                    std::distance(first1, last1), std::distance(first2, last2));

                return util::transform_binary_loop_n<
                    parallel::execution::datapar_policy>(
                    first1, count, first2, dest, std::forward<F>(f));
            }

            template <typename InIter1, typename InIter2, typename OutIter,
                typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static typename std::enable_if<
                !iterators_datapar_compatible<InIter1, OutIter>::value ||
                    !iterators_datapar_compatible<InIter2, OutIter>::value ||
                    !iterator_datapar_compatible<InIter1>::value ||
                    !iterator_datapar_compatible<InIter2>::value ||
                    !iterator_datapar_compatible<OutIter>::value,
                hpx::tuple<InIter1, InIter2, OutIter>>::type
            call(InIter1 first1, InIter1 last1, InIter2 first2, InIter2 last2,
                OutIter dest, F&& f)
            {
                return util::transform_binary_loop<sequenced_policy>(
                    first1, last1, first2, last2, dest, std::forward<F>(f));
            }
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    template <typename ExPolicy, typename Iter, typename OutIter, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE typename std::enable_if<
        hpx::is_vectorpack_execution_policy<ExPolicy>::value,
        std::pair<Iter, OutIter>>::type
    transform_loop_n(Iter it, std::size_t count, OutIter dest, F&& f)
    {
        return detail::datapar_transform_loop_n<Iter>::call(
            it, count, dest, std::forward<F>(f));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iter, typename OutIter, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE std::pair<Iter, OutIter> transform_loop(
        parallel::execution::datapar_policy, Iter it, Iter end, OutIter dest,
        F&& f)
    {
        return detail::datapar_transform_loop<Iter>::call(
            it, end, dest, std::forward<F>(f));
    }

    template <typename Iter, typename OutIter, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE std::pair<Iter, OutIter> transform_loop(
        parallel::execution::datapar_task_policy, Iter it, Iter end,
        OutIter dest, F&& f)
    {
        return detail::datapar_transform_loop<Iter>::call(
            it, end, dest, std::forward<F>(f));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename ExPolicy, typename InIter1, typename InIter2,
        typename OutIter, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE typename std::enable_if<
        hpx::is_vectorpack_execution_policy<ExPolicy>::value,
        hpx::tuple<InIter1, InIter2, OutIter>>::type
    transform_binary_loop_n(
        InIter1 first1, std::size_t count, InIter2 first2, OutIter dest, F&& f)
    {
        return detail::datapar_transform_binary_loop_n<InIter1, InIter2>::call(
            first1, count, first2, dest, std::forward<F>(f));
    }

    ///////////////////////////////////////////////////////////////////////////

    template <typename ExPolicy, typename InIter1, typename InIter2,
        typename OutIter, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE typename std::enable_if<
        hpx::is_vectorpack_execution_policy<ExPolicy>::value,
        hpx::tuple<InIter1, InIter2, OutIter>>::type
    transform_binary_loop(
        InIter1 first1, InIter1 last1, InIter2 first2, OutIter dest, F&& f)
    {
        return detail::datapar_transform_binary_loop<InIter1, InIter2>::call(
            first1, last1, first2, dest, std::forward<F>(f));
    }

    template <typename ExPolicy, typename InIter1, typename InIter2,
        typename OutIter, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE typename std::enable_if<
        hpx::is_vectorpack_execution_policy<ExPolicy>::value,
        hpx::tuple<InIter1, InIter2, OutIter>>::type
    transform_binary_loop(InIter1 first1, InIter1 last1, InIter2 first2,
        InIter2 last2, OutIter dest, F&& f)
    {
        return detail::datapar_transform_binary_loop<InIter1, InIter2>::call(
            first1, last1, first2, last2, dest, std::forward<F>(f));
    }
}}}    // namespace hpx::parallel::util

#endif
