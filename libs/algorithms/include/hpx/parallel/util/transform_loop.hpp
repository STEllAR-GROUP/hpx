//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/execution/traits/is_execution_policy.hpp>
#include <hpx/functional/invoke.hpp>
#include <hpx/parallel/util/cancellation_token.hpp>
#include <hpx/parallel/util/result_types.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { namespace util {
    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        template <typename Iter>
        struct transform_loop
        {
            template <typename InIter, typename OutIter, typename F>
            HPX_HOST_DEVICE
                HPX_FORCEINLINE static util::in_out_result<InIter, OutIter>
                call(InIter first, InIter last, OutIter dest, F&& f)
            {
                for (/* */; first != last; (void) ++first, ++dest)
                {
                    *dest = hpx::util::invoke(std::forward<F>(f), first);
                }
                return util::in_out_result<InIter, OutIter>{
                    std::move(first), std::move(dest)};
            }
        };
    }    // namespace detail

    template <typename ExPolicy, typename Iter, typename OutIter, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE util::in_out_result<Iter, OutIter>
    transform_loop(ExPolicy&&, Iter it, Iter end, OutIter dest, F&& f)
    {
        return detail::transform_loop<Iter>::call(
            it, end, dest, std::forward<F>(f));
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        template <typename Iter1, typename Iter2>
        struct transform_binary_loop
        {
            template <typename InIter1, typename InIter2, typename OutIter,
                typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static util::in_in_out_result<
                InIter1, InIter2, OutIter>
            call(InIter1 first1, InIter1 last1, InIter2 first2, OutIter dest,
                F&& f)
            {
                for (/* */; first1 != last1; (void) ++first1, ++first2, ++dest)
                {
                    *dest =
                        hpx::util::invoke(std::forward<F>(f), first1, first2);
                }
                return util::in_in_out_result<InIter1, InIter2, OutIter>{
                    std::move(first1), std::move(first2), std::move(dest)};
            }

            template <typename InIter1, typename InIter2, typename OutIter,
                typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static util::in_in_out_result<
                InIter1, InIter2, OutIter>
            call(InIter1 first1, InIter1 last1, InIter2 first2, InIter2 last2,
                OutIter dest, F&& f)
            {
                for (/* */; first1 != last1 && first2 != last2;
                     (void) ++first1, ++first2, ++dest)
                {
                    *dest =
                        hpx::util::invoke(std::forward<F>(f), first1, first2);
                }
                return util::in_in_out_result<InIter1, InIter2, OutIter>{
                    first1, first2, dest};
            }
        };
    }    // namespace detail

    template <typename ExPolicy, typename InIter1, typename InIter2,
        typename OutIter, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE typename std::enable_if<
        !execution::is_vectorpack_execution_policy<ExPolicy>::value,
        util::in_in_out_result<InIter1, InIter2, OutIter>>::type
    transform_binary_loop(
        InIter1 first1, InIter1 last1, InIter2 first2, OutIter dest, F&& f)
    {
        return detail::transform_binary_loop<InIter1, InIter2>::call(
            first1, last1, first2, dest, std::forward<F>(f));
    }

    template <typename ExPolicy, typename InIter1, typename InIter2,
        typename OutIter, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE typename std::enable_if<
        !execution::is_vectorpack_execution_policy<ExPolicy>::value,
        util::in_in_out_result<InIter1, InIter2, OutIter>>::type
    transform_binary_loop(InIter1 first1, InIter1 last1, InIter2 first2,
        InIter2 last2, OutIter dest, F&& f)
    {
        return detail::transform_binary_loop<InIter1, InIter2>::call(
            first1, last1, first2, last2, dest, std::forward<F>(f));
    }

    namespace detail {
        template <typename Iter>
        struct transform_loop_n
        {
            template <typename InIter, typename OutIter, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static util::in_out_result<InIter, OutIter>
            call(InIter first, std::size_t count, OutIter dest, F&& f)
            {
                for (/* */; count != 0; (void) --count, ++first, ++dest)
                {
                    *dest = hpx::util::invoke(f, first);
                }
                return util::in_out_result<InIter, OutIter>{std::move(first), std::move(dest)};
            }
        };
    }    // namespace detail

    template <typename ExPolicy, typename Iter, typename OutIter, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE typename std::enable_if<
        !execution::is_vectorpack_execution_policy<ExPolicy>::value,
        util::in_out_result<Iter, OutIter>>::type
    transform_loop_n(Iter it, std::size_t count, OutIter dest, F&& f)
    {
        return detail::transform_loop_n<Iter>::call(
            it, count, dest, std::forward<F>(f));
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        template <typename Iter1, typename Inter2>
        struct transform_binary_loop_n
        {
            template <typename InIter1, typename InIter2, typename OutIter,
                typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static util::in_in_out_result<InIter1,
                InIter2, OutIter>
            call(InIter1 first1, std::size_t count, InIter2 first2,
                OutIter dest, F&& f)
            {
                for (/* */; count != 0;
                     (void) --count, ++first1, first2++, ++dest)
                {
                    *dest = hpx::util::invoke(f, first1, first2);
                }
                return util::in_in_out_result<InIter1,
                InIter2, OutIter>{
                    std::move(first1), std::move(first2), std::move(dest)};
            }
        };
    }    // namespace detail

    template <typename ExPolicy, typename InIter1, typename InIter2,
        typename OutIter, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE typename std::enable_if<
        !execution::is_vectorpack_execution_policy<ExPolicy>::value,
        util::in_in_out_result<InIter1, InIter2, OutIter>>::type
    transform_binary_loop_n(
        InIter1 first1, std::size_t count, InIter2 first2, OutIter dest, F&& f)
    {
        return detail::transform_binary_loop_n<InIter1, InIter2>::call(
            first1, count, first2, dest, std::forward<F>(f));
    }
}}}    // namespace hpx::parallel::util

#if defined(HPX_HAVE_DATAPAR)
#include <hpx/parallel/datapar/transform_loop.hpp>
#endif
