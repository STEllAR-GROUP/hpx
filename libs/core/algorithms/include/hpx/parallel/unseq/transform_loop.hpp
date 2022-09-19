//  Copyright (c) 2022 A Kishore Kumar
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <hpx/executors/execution_policy.hpp>
#include <hpx/functional/tag_invoke.hpp>
#include <hpx/parallel/util/cancellation_token.hpp>
#include <hpx/parallel/util/transform_loop.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { namespace util {

    namespace detail {
        ///////////////////////////////////////////////////////////////////////
        struct unseq_transform_loop_n
        {
            template <typename InIter, typename OutIter, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static typename std::enable_if<
                hpx::traits::is_random_access_iterator_v<InIter> and
                    hpx::traits::is_random_access_iterator_v<OutIter>,
                std::pair<InIter, OutIter>>::type
            call(InIter HPX_RESTRICT it, std::size_t count,
                OutIter HPX_RESTRICT dest, F&& f)
            {
                // clang-format off
                HPX_IVDEP HPX_UNROLL HPX_VECTORIZE
                for (std::size_t i = 0; i < count; i++)
                {
                    *dest++ = HPX_INVOKE(f, it);
                    ++it;
                }
                // clang-format on
                return std::make_pair(HPX_MOVE(it), HPX_MOVE(dest));
            }

            template <typename InIter, typename OutIter, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static typename std::enable_if<
                !hpx::traits::is_random_access_iterator_v<InIter> or
                    !hpx::traits::is_random_access_iterator_v<OutIter>,
                std::pair<InIter, OutIter>>::type
            call(InIter HPX_RESTRICT it, std::size_t count,
                OutIter HPX_RESTRICT dest, F&& f)
            {
                return util::transform_loop_n<hpx::execution::sequenced_policy>(
                    it, count, dest, HPX_FORWARD(F, f));
            }
        };
    }    // namespace detail

    template <typename ExPolicy, typename Iter, typename OutIter, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE typename std::enable_if<
        hpx::is_unsequenced_execution_policy_v<ExPolicy>,
        std::pair<Iter, OutIter>>::type
    tag_invoke(hpx::parallel::util::transform_loop_n_t<ExPolicy>,
        Iter HPX_RESTRICT it, std::size_t count, OutIter HPX_RESTRICT dest,
        F&& f)
    {
        return detail::unseq_transform_loop_n::call(
            it, count, dest, HPX_FORWARD(F, f));
    }

    namespace detail {
        ///////////////////////////////////////////////////////////////////////
        struct unseq_transform_loop_n_ind
        {
            template <typename InIter, typename OutIter, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static typename std::enable_if<
                hpx::traits::is_random_access_iterator_v<InIter> and
                    hpx::traits::is_random_access_iterator_v<OutIter>,
                std::pair<InIter, OutIter>>::type
            call(InIter HPX_RESTRICT it, std::size_t count,
                OutIter HPX_RESTRICT dest, F&& f)
            {
                // clang-format off
                HPX_IVDEP HPX_UNROLL HPX_VECTORIZE
                for (std::size_t i = 0; i < count; i++)
                {
                    *dest++ = HPX_INVOKE(f, *it);
                    ++it;
                }
                // clang-format on
                return std::make_pair(HPX_MOVE(it), HPX_MOVE(dest));
            }

            template <typename InIter, typename OutIter, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static typename std::enable_if<
                !hpx::traits::is_random_access_iterator_v<InIter> or
                    !hpx::traits::is_random_access_iterator_v<OutIter>,
                std::pair<InIter, OutIter>>::type
            call(InIter HPX_RESTRICT it, std::size_t count,
                OutIter HPX_RESTRICT dest, F&& f)
            {
                return util::transform_loop_n_ind<
                    hpx::execution::sequenced_policy>(
                    it, count, dest, HPX_FORWARD(F, f));
            }
        };
    }    // namespace detail

    template <typename ExPolicy, typename Iter, typename OutIter, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE constexpr typename std::enable_if<
        hpx::is_unsequenced_execution_policy_v<ExPolicy>,
        std::pair<Iter, OutIter>>::type
    tag_invoke(hpx::parallel::util::transform_loop_n_ind_t<ExPolicy>,
        Iter HPX_RESTRICT it, std::size_t count, OutIter HPX_RESTRICT dest,
        F&& f)
    {
        return detail::unseq_transform_loop_n_ind::call(
            it, count, dest, HPX_FORWARD(F, f));
    }

    namespace detail {
        ///////////////////////////////////////////////////////////////////////
        struct unseq_transform_loop
        {
            template <typename InIter, typename OutIter, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static typename std::enable_if<
                hpx::traits::is_random_access_iterator_v<InIter> and
                    hpx::traits::is_random_access_iterator_v<OutIter>,
                util::in_out_result<InIter, OutIter>>::type
            call(InIter HPX_RESTRICT it, InIter HPX_RESTRICT last,
                OutIter HPX_RESTRICT dest, F&& f)
            {
                auto ret =
                    util::transform_loop_n<hpx::execution::unsequenced_policy>(
                        it, std::distance(it, last), dest, HPX_FORWARD(F, f));
                return util::in_out_result<InIter, OutIter>{
                    HPX_MOVE(ret.first), HPX_MOVE(ret.second)};
            }

            template <typename InIter, typename OutIter, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static typename std::enable_if<
                !hpx::traits::is_random_access_iterator_v<InIter> or
                    !hpx::traits::is_random_access_iterator_v<OutIter>,
                util::in_out_result<InIter, OutIter>>::type
            call(InIter HPX_RESTRICT it, InIter HPX_RESTRICT last,
                OutIter HPX_RESTRICT dest, F&& f)
            {
                return util::transform_loop(
                    hpx::execution::seq, it, last, dest, HPX_FORWARD(F, f));
            }
        };
    }    // namespace detail

    template <typename IterB, typename IterE, typename OutIter, typename F>
    HPX_HOST_DEVICE
        HPX_FORCEINLINE constexpr util::in_out_result<IterB, OutIter>
        tag_invoke(hpx::parallel::util::transform_loop_t,
            hpx::execution::unsequenced_policy, IterB HPX_RESTRICT it,
            IterE HPX_RESTRICT end, OutIter HPX_RESTRICT dest, F&& f)
    {
        return detail::unseq_transform_loop::call(
            it, end, dest, HPX_FORWARD(F, f));
    }

    template <typename IterB, typename IterE, typename OutIter, typename F>
    HPX_HOST_DEVICE
        HPX_FORCEINLINE constexpr util::in_out_result<IterB, OutIter>
        tag_invoke(hpx::parallel::util::transform_loop_t,
            hpx::execution::unsequenced_task_policy, IterB HPX_RESTRICT it,
            IterE HPX_RESTRICT end, OutIter HPX_RESTRICT dest, F&& f)
    {
        return detail::unseq_transform_loop::call(
            it, end, dest, HPX_FORWARD(F, f));
    }

    namespace detail {
        ///////////////////////////////////////////////////////////////////////
        struct unseq_transform_loop_ind
        {
            template <typename InIter, typename OutIter, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static typename std::enable_if<
                hpx::traits::is_random_access_iterator_v<InIter> and
                    hpx::traits::is_random_access_iterator_v<OutIter>,
                util::in_out_result<InIter, OutIter>>::type
            call(InIter HPX_RESTRICT it, InIter HPX_RESTRICT last,
                OutIter HPX_RESTRICT dest, F&& f)
            {
                auto ret = util::transform_loop_n_ind<
                    hpx::execution::unsequenced_policy>(
                    it, std::distance(it, last), dest, HPX_FORWARD(F, f));
                return util::in_out_result<InIter, OutIter>{
                    HPX_MOVE(ret.first), HPX_MOVE(ret.second)};
            }

            template <typename InIter, typename OutIter, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static typename std::enable_if<
                !hpx::traits::is_random_access_iterator_v<InIter> or
                    !hpx::traits::is_random_access_iterator_v<OutIter>,
                util::in_out_result<InIter, OutIter>>::type
            call(InIter HPX_RESTRICT it, InIter HPX_RESTRICT last,
                OutIter HPX_RESTRICT dest, F&& f)
            {
                return util::transform_loop_ind(
                    hpx::execution::seq, it, last, dest, HPX_FORWARD(F, f));
            }
        };
    }    // namespace detail

    template <typename IterB, typename IterE, typename OutIter, typename F>
    HPX_HOST_DEVICE
        HPX_FORCEINLINE constexpr util::in_out_result<IterB, OutIter>
        tag_invoke(hpx::parallel::util::transform_loop_ind_t,
            hpx::execution::unsequenced_policy, IterB HPX_RESTRICT it,
            IterE HPX_RESTRICT end, OutIter HPX_RESTRICT dest, F&& f)
    {
        return detail::unseq_transform_loop_ind::call(
            it, end, dest, HPX_FORWARD(F, f));
    }

    template <typename IterB, typename IterE, typename OutIter, typename F>
    HPX_HOST_DEVICE
        HPX_FORCEINLINE constexpr util::in_out_result<IterB, OutIter>
        tag_invoke(hpx::parallel::util::transform_loop_ind_t,
            hpx::execution::unsequenced_task_policy, IterB HPX_RESTRICT it,
            IterE HPX_RESTRICT end, OutIter HPX_RESTRICT dest, F&& f)
    {
        return detail::unseq_transform_loop_ind::call(
            it, end, dest, HPX_FORWARD(F, f));
    }

    namespace detail {
        ///////////////////////////////////////////////////////////////////////
        struct unseq_transform_binary_loop_n
        {
            template <typename InIter1, typename InIter2, typename OutIter,
                typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static typename std::enable_if<
                hpx::traits::is_random_access_iterator_v<InIter1> and
                    hpx::traits::is_random_access_iterator_v<InIter2> and
                    hpx::traits::is_random_access_iterator_v<OutIter>,
                hpx::tuple<InIter1, InIter2, OutIter>>::type
            call(InIter1 HPX_RESTRICT first1, std::size_t count,
                InIter2 HPX_RESTRICT first2, OutIter HPX_RESTRICT dest, F&& f)
            {
                // clang-format off
                HPX_IVDEP HPX_UNROLL HPX_VECTORIZE
                for (std::size_t i = 0; i < count; i++)
                {
                    *dest++ = HPX_INVOKE(f, first1, first2);
                    ++first1, ++first2;
                }
                // clang-format on
                return hpx::make_tuple(
                    HPX_MOVE(first1), HPX_MOVE(first2), HPX_MOVE(dest));
            }

            template <typename InIter1, typename InIter2, typename OutIter,
                typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static typename std::enable_if<
                !hpx::traits::is_random_access_iterator_v<InIter1> or
                    !hpx::traits::is_random_access_iterator_v<InIter2> or
                    !hpx::traits::is_random_access_iterator_v<OutIter>,
                hpx::tuple<InIter1, InIter2, OutIter>>::type
            call(InIter1 HPX_RESTRICT first1, std::size_t count,
                InIter2 HPX_RESTRICT first2, OutIter HPX_RESTRICT dest, F&& f)
            {
                return util::transform_binary_loop_n<
                    hpx::execution::sequenced_policy>(
                    first1, count, first2, dest, HPX_FORWARD(F, f));
            }
        };
    }    // namespace detail

    template <typename ExPolicy, typename InIter1, typename InIter2,
        typename OutIter, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE typename std::enable_if<
        hpx::is_unsequenced_execution_policy_v<ExPolicy>,
        hpx::tuple<InIter1, InIter2, OutIter>>::type
    tag_invoke(hpx::parallel::util::transform_binary_loop_n_t<ExPolicy>,
        InIter1 HPX_RESTRICT first1, std::size_t count,
        InIter2 HPX_RESTRICT first2, OutIter HPX_RESTRICT dest, F&& f)
    {
        return detail::unseq_transform_binary_loop_n::call(
            first1, count, first2, dest, HPX_FORWARD(F, f));
    }

    namespace detail {
        ///////////////////////////////////////////////////////////////////////
        struct unseq_transform_binary_loop
        {
            template <typename InIter1, typename InIter2, typename OutIter,
                typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static typename std::enable_if<
                hpx::traits::is_random_access_iterator_v<InIter1> and
                    hpx::traits::is_random_access_iterator_v<InIter2> and
                    hpx::traits::is_random_access_iterator_v<OutIter>,
                util::in_in_out_result<InIter1, InIter2, OutIter>>::type
            call(InIter1 HPX_RESTRICT first1, InIter1 HPX_RESTRICT last1,
                InIter2 HPX_RESTRICT first2, OutIter HPX_RESTRICT dest, F&& f)
            {
                auto ret = util::transform_binary_loop_n<
                    hpx::execution::unsequenced_policy>(first1,
                    std::distance(first1, last1), first2, dest,
                    HPX_FORWARD(F, f));
                return util::in_in_out_result<InIter1, InIter2, OutIter>{
                    hpx::get<0>(ret), hpx::get<1>(ret), hpx::get<2>(ret)};
            }

            template <typename InIter1, typename InIter2, typename OutIter,
                typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static typename std::enable_if<
                !hpx::traits::is_random_access_iterator_v<InIter1> or
                    !hpx::traits::is_random_access_iterator_v<InIter2> or
                    !hpx::traits::is_random_access_iterator_v<OutIter>,
                util::in_in_out_result<InIter1, InIter2, OutIter>>::type
            call(InIter1 HPX_RESTRICT first1, InIter1 HPX_RESTRICT last1,
                InIter2 HPX_RESTRICT first2, OutIter HPX_RESTRICT dest, F&& f)
            {
                return util::transform_binary_loop<
                    hpx::execution::sequenced_policy>(
                    first1, last1, first2, dest, HPX_FORWARD(F, f));
            }

            template <typename InIter1, typename InIter2, typename OutIter,
                typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static typename std::enable_if<
                hpx::traits::is_random_access_iterator_v<InIter1> and
                    hpx::traits::is_random_access_iterator_v<InIter2> and
                    hpx::traits::is_random_access_iterator_v<OutIter>,
                util::in_in_out_result<InIter1, InIter2, OutIter>>::type
            call(InIter1 HPX_RESTRICT first1, InIter1 HPX_RESTRICT last1,
                InIter2 HPX_RESTRICT first2, InIter2 HPX_RESTRICT last2,
                OutIter dest, F&& f)
            {
                // clang-format off
                std::size_t count = (std::min) (std::distance(first1, last1),
                    std::distance(first2, last2));
                // clang-format on
                auto ret = util::transform_binary_loop_n<
                    hpx::execution::unsequenced_policy>(
                    first1, count, first2, dest, HPX_FORWARD(F, f));
                return util::in_in_out_result<InIter1, InIter2, OutIter>{
                    hpx::get<0>(ret), hpx::get<1>(ret), hpx::get<2>(ret)};
            }

            template <typename InIter1, typename InIter2, typename OutIter,
                typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static typename std::enable_if<
                !hpx::traits::is_random_access_iterator_v<InIter1> or
                    !hpx::traits::is_random_access_iterator_v<InIter2> or
                    !hpx::traits::is_random_access_iterator_v<OutIter>,
                util::in_in_out_result<InIter1, InIter2, OutIter>>::type
            call(InIter1 HPX_RESTRICT first1, InIter1 HPX_RESTRICT last1,
                InIter2 HPX_RESTRICT first2, InIter2 HPX_RESTRICT last2,
                OutIter dest, F&& f)
            {
                return util::transform_binary_loop<
                    hpx::execution::sequenced_policy>(
                    first1, last1, first2, last2, dest, HPX_FORWARD(F, f));
            }
        };
    }    // namespace detail

    template <typename ExPolicy, typename InIter1, typename InIter2,
        typename OutIter, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE typename std::enable_if<
        hpx::is_unsequenced_execution_policy_v<ExPolicy>,
        util::in_in_out_result<InIter1, InIter2, OutIter>>::type
    tag_invoke(hpx::parallel::util::transform_binary_loop_t<ExPolicy>,
        InIter1 HPX_RESTRICT first1, InIter1 HPX_RESTRICT last1,
        InIter2 HPX_RESTRICT first2, OutIter HPX_RESTRICT dest, F&& f)
    {
        return detail::unseq_transform_binary_loop::call(
            first1, last1, first2, dest, HPX_FORWARD(F, f));
    }

    template <typename ExPolicy, typename InIter1, typename InIter2,
        typename OutIter, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE typename std::enable_if<
        hpx::is_unsequenced_execution_policy_v<ExPolicy>,
        util::in_in_out_result<InIter1, InIter2, OutIter>>::type
    tag_invoke(hpx::parallel::util::transform_binary_loop_t<ExPolicy>,
        InIter1 HPX_RESTRICT first1, InIter1 HPX_RESTRICT last1,
        InIter2 HPX_RESTRICT first2, InIter2 HPX_RESTRICT last2, OutIter dest,
        F&& f)
    {
        return detail::unseq_transform_binary_loop::call(
            first1, last1, first2, last2, dest, HPX_FORWARD(F, f));
    }

    namespace detail {
        ///////////////////////////////////////////////////////////////////////
        struct unseq_transform_binary_loop_ind_n
        {
            template <typename InIter1, typename InIter2, typename OutIter,
                typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static typename std::enable_if<
                hpx::traits::is_random_access_iterator_v<InIter1> and
                    hpx::traits::is_random_access_iterator_v<InIter2> and
                    hpx::traits::is_random_access_iterator_v<OutIter>,
                hpx::tuple<InIter1, InIter2, OutIter>>::type
            call(InIter1 HPX_RESTRICT first1, std::size_t count,
                InIter2 HPX_RESTRICT first2, OutIter HPX_RESTRICT dest, F&& f)
            {
                // clang-format off
                HPX_IVDEP HPX_UNROLL HPX_VECTORIZE
                for (std::size_t i = 0; i < count; i++)
                {
                    *dest++ = HPX_INVOKE(f, *first1, *first2);
                    ++first1, ++first2;
                }
                // clang-format on
                return hpx::make_tuple(
                    HPX_MOVE(first1), HPX_MOVE(first2), HPX_MOVE(dest));
            }

            template <typename InIter1, typename InIter2, typename OutIter,
                typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static typename std::enable_if<
                !hpx::traits::is_random_access_iterator_v<InIter1> or
                    !hpx::traits::is_random_access_iterator_v<InIter2> or
                    !hpx::traits::is_random_access_iterator_v<OutIter>,
                hpx::tuple<InIter1, InIter2, OutIter>>::type
            call(InIter1 HPX_RESTRICT first1, std::size_t count,
                InIter2 HPX_RESTRICT first2, OutIter HPX_RESTRICT dest, F&& f)
            {
                return util::transform_binary_loop_ind_n<
                    hpx::execution::sequenced_policy>(
                    first1, count, first2, dest, HPX_FORWARD(F, f));
            }
        };
    }    // namespace detail

    template <typename ExPolicy, typename InIter1, typename InIter2,
        typename OutIter, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE typename std::enable_if<
        hpx::is_unsequenced_execution_policy_v<ExPolicy>,
        hpx::tuple<InIter1, InIter2, OutIter>>::type
    tag_invoke(hpx::parallel::util::transform_binary_loop_ind_n_t<ExPolicy>,
        InIter1 HPX_RESTRICT first1, std::size_t count,
        InIter2 HPX_RESTRICT first2, OutIter HPX_RESTRICT dest, F&& f)
    {
        return detail::unseq_transform_binary_loop_ind_n::call(
            first1, count, first2, dest, HPX_FORWARD(F, f));
    }

    namespace detail {
        ///////////////////////////////////////////////////////////////////////
        struct unseq_transform_binary_loop_ind
        {
            template <typename InIter1, typename InIter2, typename OutIter,
                typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static typename std::enable_if<
                hpx::traits::is_random_access_iterator_v<InIter1> and
                    hpx::traits::is_random_access_iterator_v<InIter2> and
                    hpx::traits::is_random_access_iterator_v<OutIter>,
                util::in_in_out_result<InIter1, InIter2, OutIter>>::type
            call(InIter1 HPX_RESTRICT first1, InIter1 HPX_RESTRICT last1,
                InIter2 HPX_RESTRICT first2, OutIter HPX_RESTRICT dest, F&& f)
            {
                auto ret = util::transform_binary_loop_ind_n<
                    hpx::execution::unsequenced_policy>(first1,
                    std::distance(first1, last1), first2, dest,
                    HPX_FORWARD(F, f));
                return util::in_in_out_result<InIter1, InIter2, OutIter>{
                    hpx::get<0>(ret), hpx::get<1>(ret), hpx::get<2>(ret)};
            }

            template <typename InIter1, typename InIter2, typename OutIter,
                typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static typename std::enable_if<
                !hpx::traits::is_random_access_iterator_v<InIter1> or
                    !hpx::traits::is_random_access_iterator_v<InIter2> or
                    !hpx::traits::is_random_access_iterator_v<OutIter>,
                util::in_in_out_result<InIter1, InIter2, OutIter>>::type
            call(InIter1 HPX_RESTRICT first1, InIter1 HPX_RESTRICT last1,
                InIter2 HPX_RESTRICT first2, OutIter HPX_RESTRICT dest, F&& f)
            {
                return util::transform_binary_loop_ind<
                    hpx::execution::sequenced_policy>(
                    first1, last1, first2, dest, HPX_FORWARD(F, f));
            }

            template <typename InIter1, typename InIter2, typename OutIter,
                typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static typename std::enable_if<
                hpx::traits::is_random_access_iterator_v<InIter1> and
                    hpx::traits::is_random_access_iterator_v<InIter2> and
                    hpx::traits::is_random_access_iterator_v<OutIter>,
                util::in_in_out_result<InIter1, InIter2, OutIter>>::type
            call(InIter1 HPX_RESTRICT first1, InIter1 HPX_RESTRICT last1,
                InIter2 HPX_RESTRICT first2, InIter2 HPX_RESTRICT last2,
                OutIter HPX_RESTRICT dest, F&& f)
            {
                // clang-format off
                std::size_t count = (std::min) (std::distance(first1, last1),
                    std::distance(first2, last2));
                // clang-format on

                auto ret = util::transform_binary_loop_ind_n<
                    hpx::execution::unsequenced_policy>(
                    first1, count, first2, dest, HPX_FORWARD(F, f));
                return util::in_in_out_result<InIter1, InIter2, OutIter>{
                    hpx::get<0>(ret), hpx::get<1>(ret), hpx::get<2>(ret)};
            }

            template <typename InIter1, typename InIter2, typename OutIter,
                typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static typename std::enable_if<
                !hpx::traits::is_random_access_iterator_v<InIter1> or
                    !hpx::traits::is_random_access_iterator_v<InIter2> or
                    !hpx::traits::is_random_access_iterator_v<OutIter>,
                util::in_in_out_result<InIter1, InIter2, OutIter>>::type
            call(InIter1 HPX_RESTRICT first1, InIter1 HPX_RESTRICT last1,
                InIter2 HPX_RESTRICT first2, InIter2 HPX_RESTRICT last2,
                OutIter HPX_RESTRICT dest, F&& f)
            {
                return util::transform_binary_loop_ind<
                    hpx::execution::sequenced_policy>(
                    first1, last1, first2, last2, dest, HPX_FORWARD(F, f));
            }
        };
    }    // namespace detail

    template <typename ExPolicy, typename InIter1, typename InIter2,
        typename OutIter, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE typename std::enable_if<
        hpx::is_unsequenced_execution_policy_v<ExPolicy>,
        util::in_in_out_result<InIter1, InIter2, OutIter>>::type
    tag_invoke(hpx::parallel::util::transform_binary_loop_ind_t<ExPolicy>,
        InIter1 HPX_RESTRICT first1, InIter1 HPX_RESTRICT last1,
        InIter2 HPX_RESTRICT first2, OutIter HPX_RESTRICT dest, F&& f)
    {
        return detail::unseq_transform_binary_loop_ind::call(
            first1, last1, first2, dest, HPX_FORWARD(F, f));
    }

    template <typename ExPolicy, typename InIter1, typename InIter2,
        typename OutIter, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE typename std::enable_if<
        hpx::is_unsequenced_execution_policy_v<ExPolicy>,
        util::in_in_out_result<InIter1, InIter2, OutIter>>::type
    tag_invoke(hpx::parallel::util::transform_binary_loop_ind_t<ExPolicy>,
        InIter1 HPX_RESTRICT first1, InIter1 HPX_RESTRICT last1,
        InIter2 HPX_RESTRICT first2, InIter2 HPX_RESTRICT last2,
        OutIter HPX_RESTRICT dest, F&& f)
    {
        return detail::unseq_transform_binary_loop_ind::call(
            first1, last1, first2, last2, dest, HPX_FORWARD(F, f));
    }
}}}    // namespace hpx::parallel::util
