//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/algorithms/traits/segmented_iterator_traits.hpp>

#include <hpx/executors/execution_policy.hpp>
#include <hpx/parallel/algorithms/count.hpp>
#include <hpx/parallel/algorithms/detail/accumulate.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/detail/distance.hpp>
#include <hpx/parallel/segmented_algorithms/detail/dispatch.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/detail/handle_remote_exceptions.hpp>

#include <algorithm>
#include <exception>
#include <iterator>
#include <list>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace parallel {
    ///////////////////////////////////////////////////////////////////////////
    // segmented_count
    namespace detail {
        ///////////////////////////////////////////////////////////////////////
        /// \cond NOINTERNAL

        // sequential remote implementation
        template <typename Algo, typename ExPolicy, typename SegIterB,
            typename SegIterE, typename T, typename Proj>
        static typename util::detail::algorithm_result<ExPolicy,
            typename std::iterator_traits<SegIterB>::difference_type>::type
        segmented_count(Algo&& algo, ExPolicy const& policy, SegIterB first,
            SegIterE last, T const& value, Proj&& proj, std::true_type)
        {
            typedef hpx::traits::segmented_iterator_traits<SegIterB> traits;
            typedef typename traits::segment_iterator segment_iterator;
            typedef typename traits::local_iterator local_iterator_type;
            typedef typename std::iterator_traits<SegIterB>::difference_type
                value_type;
            typedef util::detail::algorithm_result<ExPolicy, value_type> result;

            segment_iterator sit = traits::segment(first);
            segment_iterator send = traits::segment(last);

            value_type overall_result = value_type();

            if (sit == send)
            {
                // all elements are on the same partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::local(last);
                if (beg != end)
                {
                    overall_result = dispatch(traits::get_id(sit), algo, policy,
                        std::true_type(), beg, end, value, proj);
                }
            }
            else
            {
                // handle the remaining part of the first partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::end(sit);
                if (beg != end)
                {
                    overall_result += dispatch(traits::get_id(sit), algo,
                        policy, std::true_type(), beg, end, value, proj);
                }

                // handle all of the full partitions
                for (++sit; sit != send; ++sit)
                {
                    beg = traits::begin(sit);
                    end = traits::end(sit);
                    if (beg != end)
                    {
                        overall_result += dispatch(traits::get_id(sit), algo,
                            policy, std::true_type(), beg, end, value, proj);
                    }
                }

                // handle the beginning of the last partition
                beg = traits::begin(sit);
                end = traits::local(last);
                if (beg != end)
                {
                    overall_result += dispatch(traits::get_id(sit), algo,
                        policy, std::true_type(), beg, end, value, proj);
                }
            }

            return result::get(HPX_MOVE(overall_result));
        }

        // parallel remote implementation
        template <typename Algo, typename ExPolicy, typename SegIterB,
            typename SegIterE, typename T, typename Proj>
        static typename util::detail::algorithm_result<ExPolicy,
            typename std::iterator_traits<SegIterB>::difference_type>::type
        segmented_count(Algo&& algo, ExPolicy const& policy, SegIterB first,
            SegIterE last, T const& value, Proj&& proj, std::false_type)
        {
            typedef hpx::traits::segmented_iterator_traits<SegIterB> traits;
            typedef typename traits::segment_iterator segment_iterator;
            typedef typename traits::local_iterator local_iterator_type;

            typedef std::integral_constant<bool,
                !hpx::traits::is_forward_iterator<SegIterB>::value>
                forced_seq;

            typedef typename std::iterator_traits<SegIterB>::difference_type
                value_type;
            typedef util::detail::algorithm_result<ExPolicy, value_type> result;

            segment_iterator sit = traits::segment(first);
            segment_iterator send = traits::segment(last);

            std::vector<shared_future<value_type>> segments;
            segments.reserve(detail::distance(sit, send));

            if (sit == send)
            {
                // all elements are on the same partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::local(last);
                if (beg != end)
                {
                    segments.push_back(dispatch_async(traits::get_id(sit), algo,
                        policy, forced_seq(), beg, end, value, proj));
                }
            }
            else
            {
                // handle the remaining part of the first partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::end(sit);
                if (beg != end)
                {
                    segments.push_back(dispatch_async(traits::get_id(sit), algo,
                        policy, forced_seq(), beg, end, value, proj));
                }

                // handle all of the full partitions
                for (++sit; sit != send; ++sit)
                {
                    beg = traits::begin(sit);
                    end = traits::end(sit);
                    if (beg != end)
                    {
                        segments.push_back(dispatch_async(traits::get_id(sit),
                            algo, policy, forced_seq(), beg, end, value, proj));
                    }
                }

                // handle the beginning of the last partition
                beg = traits::begin(sit);
                end = traits::local(last);
                if (beg != end)
                {
                    segments.push_back(dispatch_async(traits::get_id(sit), algo,
                        policy, forced_seq(), beg, end, value, proj));
                }
            }

            return result::get(
                dataflow(hpx::unwrapping([=](std::vector<value_type>&& r) {
                    return detail::accumulate(r.begin(), r.end(), value_type());
                }),
                    segments));
        }
        /// \endcond
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    // segmented_count_if
    namespace detail {
        ///////////////////////////////////////////////////////////////////////
        /// \cond NOINTERNAL

        // sequential remote implementation
        template <typename Algo, typename ExPolicy, typename SegIterB,
            typename SegIterE, typename F, typename Proj>
        static typename util::detail::algorithm_result<ExPolicy,
            typename std::iterator_traits<SegIterB>::difference_type>::type
        segmented_count_if(Algo&& algo, ExPolicy const& policy, SegIterB first,
            SegIterE last, F&& f, Proj&& proj, std::true_type)
        {
            typedef hpx::traits::segmented_iterator_traits<SegIterB> traits;
            typedef typename traits::segment_iterator segment_iterator;
            typedef typename traits::local_iterator local_iterator_type;
            typedef typename std::iterator_traits<SegIterB>::difference_type
                value_type;
            typedef util::detail::algorithm_result<ExPolicy, value_type> result;

            segment_iterator sit = traits::segment(first);
            segment_iterator send = traits::segment(last);

            value_type overall_result = value_type();

            if (sit == send)
            {
                // all elements are on the same partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::local(last);
                if (beg != end)
                {
                    overall_result = dispatch(traits::get_id(sit), algo, policy,
                        std::true_type(), beg, end, HPX_FORWARD(F, f),
                        HPX_FORWARD(Proj, proj));
                }
            }
            else
            {
                // handle the remaining part of the first partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::end(sit);
                if (beg != end)
                {
                    overall_result += dispatch(traits::get_id(sit), algo,
                        policy, std::true_type(), beg, end, HPX_FORWARD(F, f),
                        HPX_FORWARD(Proj, proj));
                }

                // handle all of the full partitions
                for (++sit; sit != send; ++sit)
                {
                    beg = traits::begin(sit);
                    end = traits::end(sit);
                    if (beg != end)
                    {
                        overall_result += dispatch(traits::get_id(sit), algo,
                            policy, std::true_type(), beg, end,
                            HPX_FORWARD(F, f), HPX_FORWARD(Proj, proj));
                    }
                }

                // handle the beginning of the last partition
                beg = traits::begin(sit);
                end = traits::local(last);
                if (beg != end)
                {
                    overall_result += dispatch(traits::get_id(sit), algo,
                        policy, std::true_type(), beg, end, HPX_FORWARD(F, f),
                        HPX_FORWARD(Proj, proj));
                }
            }

            return result::get(HPX_MOVE(overall_result));
        }

        // parallel remote implementation
        template <typename Algo, typename ExPolicy, typename SegIterB,
            typename SegIterE, typename F, typename Proj>
        static typename util::detail::algorithm_result<ExPolicy,
            typename std::iterator_traits<SegIterB>::difference_type>::type
        segmented_count_if(Algo&& algo, ExPolicy const& policy, SegIterB first,
            SegIterE last, F&& f, Proj&& proj, std::false_type)
        {
            typedef hpx::traits::segmented_iterator_traits<SegIterB> traits;
            typedef typename traits::segment_iterator segment_iterator;
            typedef typename traits::local_iterator local_iterator_type;

            typedef std::integral_constant<bool,
                !hpx::traits::is_forward_iterator<SegIterB>::value>
                forced_seq;

            typedef typename std::iterator_traits<SegIterB>::difference_type
                value_type;
            typedef util::detail::algorithm_result<ExPolicy, value_type> result;

            segment_iterator sit = traits::segment(first);
            segment_iterator send = traits::segment(last);

            std::vector<shared_future<value_type>> segments;
            segments.reserve(detail::distance(sit, send));

            if (sit == send)
            {
                // all elements are on the same partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::local(last);
                if (beg != end)
                {
                    segments.push_back(dispatch_async(traits::get_id(sit), algo,
                        policy, forced_seq(), beg, end, HPX_FORWARD(F, f),
                        HPX_FORWARD(Proj, proj)));
                }
            }
            else
            {
                // handle the remaining part of the first partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::end(sit);
                if (beg != end)
                {
                    segments.push_back(dispatch_async(traits::get_id(sit), algo,
                        policy, forced_seq(), beg, end, HPX_FORWARD(F, f),
                        HPX_FORWARD(Proj, proj)));
                }

                // handle all of the full partitions
                for (++sit; sit != send; ++sit)
                {
                    beg = traits::begin(sit);
                    end = traits::end(sit);
                    if (beg != end)
                    {
                        segments.push_back(dispatch_async(traits::get_id(sit),
                            algo, policy, forced_seq(), beg, end,
                            HPX_FORWARD(F, f), HPX_FORWARD(Proj, proj)));
                    }
                }

                // handle the beginning of the last partition
                beg = traits::begin(sit);
                end = traits::local(last);
                if (beg != end)
                {
                    segments.push_back(dispatch_async(traits::get_id(sit), algo,
                        policy, forced_seq(), beg, end, HPX_FORWARD(F, f),
                        HPX_FORWARD(Proj, proj)));
                }
            }

            return result::get(dataflow(
                [=](std::vector<shared_future<value_type>>&& r) -> value_type {
                    // handle any remote exceptions, will throw on error
                    std::list<std::exception_ptr> errors;
                    parallel::util::detail::handle_remote_exceptions<
                        ExPolicy>::call(r, errors);

                    return detail::accumulate(r.begin(), r.end(), value_type(),
                        [](value_type const& val,
                            shared_future<value_type>& curr) {
                            return val + curr.get();
                        });
                },
                HPX_MOVE(segments)));
        }
        /// \endcond
    }    // namespace detail
}}       // namespace hpx::parallel

// The segmented iterators we support all live in namespace hpx::segmented
namespace hpx { namespace segmented {

    // clang-format off
    template <typename InIter,
        typename T,
        HPX_CONCEPT_REQUIRES_(
            hpx::traits::is_iterator<InIter>::value &&
            hpx::traits::is_segmented_iterator<InIter>::value
        )>
    // clang-format on
    typename std::iterator_traits<InIter>::difference_type tag_invoke(
        hpx::count_t, InIter first, InIter last, T const& value)
    {
        static_assert((hpx::traits::is_input_iterator<InIter>::value),
            "Requires at least input iterator.");

        using difference_type =
            typename std::iterator_traits<InIter>::difference_type;

        if (first == last)
        {
            return difference_type();
        }

        return hpx::parallel::detail::segmented_count(
            hpx::parallel::detail::count<difference_type>(),
            hpx::execution::seq, first, last, value, hpx::identity_v,
            std::true_type());
    }

    // clang-format off
    template <typename ExPolicy, typename SegIter,
        typename T,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy<ExPolicy>::value &&
            hpx::traits::is_iterator<SegIter>::value &&
            hpx::traits::is_segmented_iterator<SegIter>::value
        )>
    // clang-format on
    typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
        typename std::iterator_traits<SegIter>::difference_type>::type
    tag_invoke(hpx::count_t, ExPolicy&& policy, SegIter first, SegIter last,
        T const& value)
    {
        static_assert((hpx::traits::is_forward_iterator<SegIter>::value),
            "Requires at least forward iterator.");

        using difference_type =
            typename std::iterator_traits<SegIter>::difference_type;

        using is_seq = hpx::is_sequenced_execution_policy<ExPolicy>;

        if (first == last)
        {
            return parallel::util::detail::algorithm_result<ExPolicy,
                difference_type>::get(difference_type());
        }

        return hpx::parallel::detail::segmented_count(
            hpx::parallel::detail::count<difference_type>(),
            HPX_FORWARD(ExPolicy, policy), first, last, value, hpx::identity_v,
            is_seq());
    }

    // clang-format off
    template <typename InIter,
        typename F,
        HPX_CONCEPT_REQUIRES_(
            hpx::traits::is_iterator<InIter>::value &&
            hpx::traits::is_segmented_iterator<InIter>::value
        )>
    // clang-format on
    typename std::iterator_traits<InIter>::difference_type tag_invoke(
        hpx::count_if_t, InIter first, InIter last, F&& f)
    {
        static_assert((hpx::traits::is_input_iterator<InIter>::value),
            "Requires at least input iterator.");

        using difference_type =
            typename std::iterator_traits<InIter>::difference_type;

        if (first == last)
        {
            return difference_type();
        }

        return hpx::parallel::detail::segmented_count_if(
            hpx::parallel::detail::count_if<difference_type>(),
            hpx::execution::seq, first, last, HPX_FORWARD(F, f),
            hpx::identity_v, std::true_type());
    }

    // clang-format off
    template <typename ExPolicy, typename SegIter,
        typename F,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy<ExPolicy>::value &&
            hpx::traits::is_iterator<SegIter>::value &&
            hpx::traits::is_segmented_iterator<SegIter>::value
        )>
    // clang-format on
    typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
        typename std::iterator_traits<SegIter>::difference_type>::type
    tag_invoke(
        hpx::count_if_t, ExPolicy&& policy, SegIter first, SegIter last, F&& f)
    {
        static_assert((hpx::traits::is_forward_iterator<SegIter>::value),
            "Requires at least forward iterator.");

        using difference_type =
            typename std::iterator_traits<SegIter>::difference_type;

        using is_seq = hpx::is_sequenced_execution_policy<ExPolicy>;

        if (first == last)
        {
            return parallel::util::detail::algorithm_result<ExPolicy,
                difference_type>::get(difference_type());
        }

        return hpx::parallel::detail::segmented_count_if(
            hpx::parallel::detail::count_if<difference_type>(),
            HPX_FORWARD(ExPolicy, policy), first, last, HPX_FORWARD(F, f),
            hpx::identity_v, is_seq());
    }
}}    // namespace hpx::segmented
