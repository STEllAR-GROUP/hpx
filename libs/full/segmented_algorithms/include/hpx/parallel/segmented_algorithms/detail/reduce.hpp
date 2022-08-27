//  Copyright (c) 2017 Ajai V George
//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/algorithms/traits/segmented_iterator_traits.hpp>
#include <hpx/functional/invoke.hpp>
#include <hpx/parallel/segmented_algorithms/traits/zip_iterator.hpp>

#include <hpx/executors/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/segmented_algorithms/detail/dispatch.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/zip_iterator.hpp>

#include <algorithm>
#include <cstddef>
#include <exception>
#include <iterator>
#include <list>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace parallel { inline namespace v1 { namespace detail {
    template <typename T>
    struct seg_reduce : public detail::algorithm<seg_reduce<T>, T>
    {
        seg_reduce()
          : seg_reduce::algorithm("reduce")
        {
        }

        template <typename ExPolicy, typename FwdIter, typename Reduce>
        static T sequential(ExPolicy, FwdIter first, FwdIter last, Reduce&& r)
        {
            return util::accumulate<T>(first, last, r);
        }

        template <typename ExPolicy, typename FwdIter, typename Reduce>
        static typename util::detail::algorithm_result<ExPolicy, T>::type
        parallel(ExPolicy&& policy, FwdIter first, FwdIter last, Reduce&& r)
        {
            return util::partitioner<ExPolicy, T>::call(
                HPX_FORWARD(ExPolicy, policy), first,
                std::distance(first, last),
                [r](FwdIter part_begin, std::size_t part_size) -> T {
                    T val = *part_begin;
                    return util::accumulate_n(
                        ++part_begin, --part_size, HPX_MOVE(val), r);
                },
                hpx::unwrapping([r](auto&& results) -> T {
                    auto rfirst = hpx::util::begin(results);
                    auto rlast = hpx::util::end(results);
                    return util::accumulate<T>(rfirst, rlast, r);
                }));
        }
    };

    template <typename T>
    struct seg_transform_reduce
      : public detail::algorithm<seg_transform_reduce<T>, T>
    {
        seg_transform_reduce()
          : seg_transform_reduce::algorithm("transform_reduce")
        {
        }

        template <typename ExPolicy, typename FwdIter, typename Reduce,
            typename Convert>
        static T sequential(
            ExPolicy, FwdIter first, FwdIter last, Reduce&& r, Convert&& conv)
        {
            return util::accumulate<T>(first, last, HPX_FORWARD(Reduce, r),
                HPX_FORWARD(Convert, conv));
        }

        template <typename ExPolicy, typename FwdIter, typename Reduce,
            typename Convert>
        static typename util::detail::algorithm_result<ExPolicy, T>::type
        parallel(ExPolicy&& policy, FwdIter first, FwdIter last, Reduce&& r,
            Convert&& conv)
        {
            return util::partitioner<ExPolicy, T>::call(
                HPX_FORWARD(ExPolicy, policy), first,
                std::distance(first, last),
                [r, conv](
                    FwdIter part_begin, std::size_t part_size) mutable -> T {
                    T val = HPX_INVOKE(conv, *part_begin);
                    return util::accumulate_n(++part_begin, --part_size,
                        HPX_MOVE(val),
                        // MSVC14 bails out if r and conv are captured by
                        // reference
                        [=](T res, auto const& next) mutable -> T {
                            return HPX_INVOKE(
                                r, HPX_MOVE(res), HPX_INVOKE(conv, next));
                        });
                },
                hpx::unwrapping([r](auto&& results) -> T {
                    auto rfirst = hpx::util::begin(results);
                    auto rlast = hpx::util::end(results);
                    return util::accumulate<T>(rfirst, rlast, r);
                }));
        }
    };

    template <typename T>
    struct seg_transform_reduce_binary
      : public detail::algorithm<seg_transform_reduce_binary<T>, T>
    {
        seg_transform_reduce_binary()
          : seg_transform_reduce_binary::algorithm("transform_reduce_binary")
        {
        }

        template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
            typename Reduce, typename Convert>
        static T sequential(ExPolicy&& /* policy */, FwdIter1 first1,
            FwdIter1 last1, FwdIter2 first2, Reduce&& r, Convert&& conv)
        {
            return util::accumulate<T>(first1, last1, first2,
                HPX_FORWARD(Reduce, r), HPX_FORWARD(Convert, conv));
        }

        template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
            typename Reduce, typename Convert>
        static typename util::detail::algorithm_result<ExPolicy, T>::type
        parallel(ExPolicy&& policy, FwdIter1 first1, FwdIter1 last1,
            FwdIter2 first2, Reduce&& r, Convert&& conv)
        {
            typedef hpx::util::zip_iterator<FwdIter1, FwdIter2> zip_iterator;
            using hpx::util::make_zip_iterator;
            return util::partitioner<ExPolicy, T>::call(
                HPX_FORWARD(ExPolicy, policy),
                make_zip_iterator(first1, first2), std::distance(first1, last1),
                [&r, &conv](
                    zip_iterator part_begin, std::size_t part_size) -> T {
                    auto iters = part_begin.get_iterator_tuple();
                    FwdIter1 it1 = hpx::get<0>(iters);
                    FwdIter2 it2 = hpx::get<1>(iters);
                    FwdIter1 last1 = it1;
                    std::advance(last1, part_size);
                    return util::accumulate<T>(it1, last1, it2,
                        HPX_FORWARD(Reduce, r), HPX_FORWARD(Convert, conv));
                },
                hpx::unwrapping([r](auto&& results) -> T {
                    auto rfirst1 = hpx::util::begin(results);
                    auto rlast1 = hpx::util::end(results);
                    return util::accumulate<T>(rfirst1, rlast1, r);
                }));
        }
    };
}}}}    // namespace hpx::parallel::v1::detail
