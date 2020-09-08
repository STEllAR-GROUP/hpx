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

namespace hpx { namespace parallel { inline namespace v1 {
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

            return result::get(std::move(overall_result));
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

            return result::get(dataflow(
                hpx::util::unwrapping([=](std::vector<value_type>&& r) {
                    return detail::accumulate(r.begin(), r.end(), value_type());
                }),
                segments));
        }

        ///////////////////////////////////////////////////////////////////////
        // segmented implementation
        template <typename ExPolicy, typename InIterB, typename InIterE,
            typename T, typename Proj>
        inline typename util::detail::algorithm_result<ExPolicy,
            typename std::iterator_traits<InIterB>::difference_type>::type
        count_(ExPolicy&& policy, InIterB first, InIterE last, T const& value,
            Proj&& proj, std::true_type)
        {
            typedef parallel::execution::is_sequenced_execution_policy<ExPolicy>
                is_seq;

            typedef typename std::iterator_traits<InIterB>::difference_type
                difference_type;

            if (first == last)
            {
                return util::detail::algorithm_result<ExPolicy,
                    difference_type>::get(difference_type());
            }

            return segmented_count(count<difference_type>(),
                std::forward<ExPolicy>(policy), first, last, value,
                std::forward<Proj>(proj), is_seq());
        }

        // forward declare the non-segmented version of this algorithm
        template <typename ExPolicy, typename InIterB, typename InIterE,
            typename T, typename Proj>
        typename util::detail::algorithm_result<ExPolicy,
            typename std::iterator_traits<InIterB>::difference_type>::type
        count_(ExPolicy&& policy, InIterB first, InIterE last, T const& value,
            Proj&& proj, std::false_type);

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
                        std::true_type(), beg, end, std::forward<F>(f),
                        std::forward<Proj>(proj));
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
                        policy, std::true_type(), beg, end, std::forward<F>(f),
                        std::forward<Proj>(proj));
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
                            std::forward<F>(f), std::forward<Proj>(proj));
                    }
                }

                // handle the beginning of the last partition
                beg = traits::begin(sit);
                end = traits::local(last);
                if (beg != end)
                {
                    overall_result += dispatch(traits::get_id(sit), algo,
                        policy, std::true_type(), beg, end, std::forward<F>(f),
                        std::forward<Proj>(proj));
                }
            }

            return result::get(std::move(overall_result));
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
                        policy, forced_seq(), beg, end, std::forward<F>(f),
                        std::forward<Proj>(proj)));
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
                        policy, forced_seq(), beg, end, std::forward<F>(f),
                        std::forward<Proj>(proj)));
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
                            std::forward<F>(f), std::forward<Proj>(proj)));
                    }
                }

                // handle the beginning of the last partition
                beg = traits::begin(sit);
                end = traits::local(last);
                if (beg != end)
                {
                    segments.push_back(dispatch_async(traits::get_id(sit), algo,
                        policy, forced_seq(), beg, end, std::forward<F>(f),
                        std::forward<Proj>(proj)));
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
                std::move(segments)));
        }

        template <typename ExPolicy, typename InIterB, typename InIterE,
            typename F, typename Proj>
        inline typename util::detail::algorithm_result<ExPolicy,
            typename std::iterator_traits<InIterB>::difference_type>::type
        count_if_(ExPolicy&& policy, InIterB first, InIterE last, F&& f,
            Proj&& proj, std::true_type)
        {
            typedef parallel::execution::is_sequenced_execution_policy<ExPolicy>
                is_seq;

            typedef typename std::iterator_traits<InIterB>::difference_type
                difference_type;

            if (first == last)
            {
                return util::detail::algorithm_result<ExPolicy,
                    difference_type>::get(difference_type());
            }

            return segmented_count_if(count_if<difference_type>(),
                std::forward<ExPolicy>(policy), first, last, std::forward<F>(f),
                std::forward<Proj>(proj), is_seq());
        }

        // forward declare the non-segmented version of this algorithm
        template <typename ExPolicy, typename InIterB, typename InIterE,
            typename F, typename Proj>
        typename util::detail::algorithm_result<ExPolicy,
            typename std::iterator_traits<InIterB>::difference_type>::type
        count_if_(ExPolicy&& policy, InIterB first, InIterE last, F&& f,
            Proj&& proj, std::false_type);

        /// \endcond
    }    // namespace detail
}}}      // namespace hpx::parallel::v1
