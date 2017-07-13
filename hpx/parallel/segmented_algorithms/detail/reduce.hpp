//  Copyright (c) 2017 Ajai V George
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_SEGMENTED_ALGORITHM_REDUCE)
#define HPX_PARALLEL_SEGMENTED_ALGORITHM_REDUCE
#include <hpx/config.hpp>
#include <hpx/traits/segmented_iterator_traits.hpp>
#include <hpx/util/invoke.hpp>

#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/segmented_algorithms/detail/dispatch.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>

#include <exception>
#include <algorithm>
#include <iterator>
#include <list>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace parallel { inline namespace v1 { namespace detail
{
    template <typename T>
    struct segmented_reduce : public detail::algorithm<segmented_reduce<T>, T>
    {
        segmented_reduce()
          : segmented_reduce::algorithm("reduce")
        {}

        template <typename ExPolicy, typename InIter, typename Reduce>
        static T
        sequential(ExPolicy, InIter first, InIter last, Reduce && r)
        {
            return std::accumulate(first, last, std::forward<Reduce>(r));
        }

        template <typename ExPolicy, typename FwdIter, typename Reduce>
        static typename util::detail::algorithm_result<ExPolicy, T>::type
        parallel(ExPolicy && policy, FwdIter first, FwdIter last, Reduce && r)
        {
            return util::partitioner<ExPolicy, T>::call(
                std::forward<ExPolicy>(policy),
                first, std::distance(first, last),
                [r](FwdIter part_begin, std::size_t part_size) -> T
                {
                    T val = *part_begin;
                    return util::accumulate_n(++part_begin, --part_size,
                        std::move(val), r);
                },
                hpx::util::unwrapped([r](std::vector<T> && results)
                {
                    return util::accumulate_n(boost::begin(results),
                        boost::size(results), r);
                }));
        }
    };

    template <typename T>
    struct segmented_transform_reduce
      : public detail::algorithm<segmented_transform_reduce<T>, T>
    {
        segmented_transform_reduce()
          : segmented_transform_reduce::algorithm("transform_reduce")
        {}

        template <typename ExPolicy, typename InIter, typename Reduce,
            typename Convert>
        static T
        sequential(ExPolicy, InIter first, InIter last, Reduce && r,
            Convert && conv)
        {
            typedef typename std::iterator_traits<InIter>::value_type
                value_type;

            return std::accumulate(first, last,
                [&r, &conv](T const& res, value_type const& next)
                {
                    return hpx::util::invoke(r, res,
                        hpx::util::invoke(conv, next));
                });
        }

        template <typename ExPolicy, typename FwdIter, typename Reduce,
            typename Convert>
        static typename util::detail::algorithm_result<ExPolicy, T>::type
        parallel(ExPolicy && policy, FwdIter first, FwdIter last, Reduce && r,
            Convert && conv)
        {
            typedef typename std::iterator_traits<FwdIter>::reference
                reference;

            return util::partitioner<ExPolicy, T>::call(
                std::forward<ExPolicy>(policy),
                first, std::distance(first, last),
                [r, conv](FwdIter part_begin, std::size_t part_size) -> T
                {
                    T val = hpx::util::invoke(conv, *part_begin);
                    return util::accumulate_n(++part_begin, --part_size,
                        std::move(val),
                        // MSVC14 bails out if r and conv are captured by
                        // reference
                        [=](T const& res, reference next)
                        {
                            return hpx::util::invoke(r, res,
                                hpx::util::invoke(conv, next));
                        });
                },
                hpx::util::unwrapped([r](std::vector<T> && results)
                {
                    return util::accumulate_n(boost::begin(results),
                        boost::size(results), r);
                }));
        }
    };
}}}}
#endif
