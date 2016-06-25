//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_SEGMENTED_ALGORITHM_COUNT_DEC_25_2014_0207PM)
#define HPX_PARALLEL_SEGMENTED_ALGORITHM_COUNT_DEC_25_2014_0207PM

#include <hpx/config.hpp>
#include <hpx/traits/segmented_iterator_traits.hpp>

#include <hpx/parallel/algorithms/count.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/segmented_algorithms/detail/dispatch.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/detail/handle_remote_exceptions.hpp>

#include <boost/exception_ptr.hpp>

#include <algorithm>
#include <iterator>
#include <list>
#include <numeric>
#include <type_traits>
#include <vector>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
    ///////////////////////////////////////////////////////////////////////////
    // segmented_count
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        /// \cond NOINTERNAL

        // sequential remote implementation
        template <typename Algo, typename ExPolicy, typename SegIter,
            typename T>
        static typename util::detail::algorithm_result<
            ExPolicy, typename std::iterator_traits<SegIter>::difference_type
        >::type
        segmented_count(Algo && algo, ExPolicy const& policy,
            SegIter first, SegIter last, T const& value, std::true_type)
        {
            typedef hpx::traits::segmented_iterator_traits<SegIter> traits;
            typedef typename traits::segment_iterator segment_iterator;
            typedef typename traits::local_iterator local_iterator_type;
            typedef typename std::iterator_traits<SegIter>::difference_type
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
                    overall_result = dispatch(traits::get_id(sit),
                        algo, policy, std::true_type(), beg, end, value);
                }
            }
            else {
                // handle the remaining part of the first partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::end(sit);
                if (beg != end)
                {
                    overall_result += dispatch(traits::get_id(sit),
                        algo, policy, std::true_type(), beg, end, value);
                }

                // handle all of the full partitions
                for (++sit; sit != send; ++sit)
                {
                    beg = traits::begin(sit);
                    end = traits::end(sit);
                    if (beg != end)
                    {
                        overall_result += dispatch(traits::get_id(sit),
                            algo, policy, std::true_type(), beg, end, value);
                    }
                }

                // handle the beginning of the last partition
                beg = traits::begin(sit);
                end = traits::local(last);
                if (beg != end)
                {
                    overall_result += dispatch(traits::get_id(sit),
                        algo, policy, std::true_type(), beg, end, value);
                }
            }

            return result::get(std::move(overall_result));
        }

        // parallel remote implementation
        template <typename Algo, typename ExPolicy, typename SegIter,
            typename T>
        static typename util::detail::algorithm_result<
            ExPolicy, typename std::iterator_traits<SegIter>::difference_type
        >::type
        segmented_count(Algo && algo, ExPolicy const& policy,
            SegIter first, SegIter last, T const& value, std::false_type)
        {
            typedef hpx::traits::segmented_iterator_traits<SegIter> traits;
            typedef typename traits::segment_iterator segment_iterator;
            typedef typename traits::local_iterator local_iterator_type;

            typedef std::integral_constant<bool,
                    !hpx::traits::is_forward_iterator<SegIter>::value
                > forced_seq;

            typedef typename std::iterator_traits<SegIter>::difference_type
                value_type;
            typedef util::detail::algorithm_result<ExPolicy, value_type> result;

            segment_iterator sit = traits::segment(first);
            segment_iterator send = traits::segment(last);

            std::vector<shared_future<value_type> > segments;
            segments.reserve(std::distance(sit, send));

            if (sit == send)
            {
                // all elements are on the same partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::local(last);
                if (beg != end)
                {
                    segments.push_back(dispatch_async(traits::get_id(sit),
                        algo, policy, forced_seq(), beg, end, value));
                }
            }
            else {
                // handle the remaining part of the first partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::end(sit);
                if (beg != end)
                {
                    segments.push_back(dispatch_async(traits::get_id(sit),
                        algo, policy, forced_seq(), beg, end, value));
                }

                // handle all of the full partitions
                for (++sit; sit != send; ++sit)
                {
                    beg = traits::begin(sit);
                    end = traits::end(sit);
                    if (beg != end)
                    {
                        segments.push_back(dispatch_async(traits::get_id(sit),
                            algo, policy, forced_seq(), beg, end, value));
                    }
                }

                // handle the beginning of the last partition
                beg = traits::begin(sit);
                end = traits::local(last);
                if (beg != end)
                {
                    segments.push_back(dispatch_async(traits::get_id(sit),
                        algo, policy, forced_seq(), beg, end, value));
                }
            }

            return result::get(
                dataflow(
                    hpx::util::unwrapped([=](std::vector<value_type> && r)
                    {
                        return std::accumulate(r.begin(), r.end(), value_type());
                    }),
                    segments));
        }

        ///////////////////////////////////////////////////////////////////////
        // segmented implementation
        template <typename ExPolicy, typename InIter, typename T>
        inline typename util::detail::algorithm_result<
            ExPolicy, typename std::iterator_traits<InIter>::difference_type
        >::type
        count_(ExPolicy&& policy, InIter first, InIter last, T const& value,
            std::true_type)
        {
            typedef parallel::is_sequential_execution_policy<ExPolicy>is_seq;

            typedef typename std::iterator_traits<InIter>::difference_type
                difference_type;

            if (first == last)
            {
                return util::detail::algorithm_result<
                    ExPolicy, difference_type>::get(difference_type());
            }

            return segmented_count(
                count<difference_type>(), std::forward<ExPolicy>(policy),
                first, last, value, is_seq());
        }

        // forward declare the non-segmented version of this algorithm
        template <typename ExPolicy, typename InIter, typename T>
        typename util::detail::algorithm_result<
            ExPolicy, typename std::iterator_traits<InIter>::difference_type
        >::type
        count_(ExPolicy&& policy, InIter first, InIter last, T const& value,
            std::false_type);

        /// \endcond
    }

    ///////////////////////////////////////////////////////////////////////////
    // segmented_count_if
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        /// \cond NOINTERNAL

        // sequential remote implementation
        template <typename Algo, typename ExPolicy, typename SegIter,
            typename F>
        static typename util::detail::algorithm_result<
            ExPolicy, typename std::iterator_traits<SegIter>::difference_type
        >::type
        segmented_count_if(Algo && algo, ExPolicy const& policy,
            SegIter first, SegIter last, F && f, std::true_type)
        {
            typedef hpx::traits::segmented_iterator_traits<SegIter> traits;
            typedef typename traits::segment_iterator segment_iterator;
            typedef typename traits::local_iterator local_iterator_type;
            typedef typename std::iterator_traits<SegIter>::difference_type
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
                    overall_result = dispatch(traits::get_id(sit),
                        algo, policy, std::true_type(),
                        beg, end, std::forward<F>(f));
                }
            }
            else {
                // handle the remaining part of the first partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::end(sit);
                if (beg != end)
                {
                    overall_result += dispatch(traits::get_id(sit),
                        algo, policy, std::true_type(),
                        beg, end, std::forward<F>(f));
                }

                // handle all of the full partitions
                for (++sit; sit != send; ++sit)
                {
                    beg = traits::begin(sit);
                    end = traits::end(sit);
                    if (beg != end)
                    {
                        overall_result += dispatch(traits::get_id(sit),
                            algo, policy, std::true_type(),
                            beg, end, std::forward<F>(f));
                    }
                }

                // handle the beginning of the last partition
                beg = traits::begin(sit);
                end = traits::local(last);
                if (beg != end)
                {
                    overall_result += dispatch(traits::get_id(sit),
                        algo, policy, std::true_type(),
                        beg, end, std::forward<F>(f));
                }
            }

            return result::get(std::move(overall_result));
        }

        // parallel remote implementation
        template <typename Algo, typename ExPolicy, typename SegIter,
            typename F>
        static typename util::detail::algorithm_result<
            ExPolicy, typename std::iterator_traits<SegIter>::difference_type
        >::type
        segmented_count_if(Algo && algo, ExPolicy const& policy,
            SegIter first, SegIter last, F && f, std::false_type)
        {
            typedef hpx::traits::segmented_iterator_traits<SegIter> traits;
            typedef typename traits::segment_iterator segment_iterator;
            typedef typename traits::local_iterator local_iterator_type;

            typedef std::integral_constant<bool,
                    !hpx::traits::is_forward_iterator<SegIter>::value
                > forced_seq;

            typedef typename std::iterator_traits<SegIter>::difference_type
                value_type;
            typedef util::detail::algorithm_result<ExPolicy, value_type> result;

            segment_iterator sit = traits::segment(first);
            segment_iterator send = traits::segment(last);

            std::vector<shared_future<value_type> > segments;
            segments.reserve(std::distance(sit, send));

            if (sit == send)
            {
                // all elements are on the same partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::local(last);
                if (beg != end)
                {
                    segments.push_back(dispatch_async(traits::get_id(sit),
                        algo, policy, forced_seq(),
                        beg, end, std::forward<F>(f)));
                }
            }
            else {
                // handle the remaining part of the first partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::end(sit);
                if (beg != end)
                {
                    segments.push_back(dispatch_async(traits::get_id(sit),
                        algo, policy, forced_seq(),
                        beg, end, std::forward<F>(f)));
                }

                // handle all of the full partitions
                for (++sit; sit != send; ++sit)
                {
                    beg = traits::begin(sit);
                    end = traits::end(sit);
                    if (beg != end)
                    {
                        segments.push_back(dispatch_async(traits::get_id(sit),
                            algo, policy, forced_seq(),
                            beg, end, std::forward<F>(f)));
                    }
                }

                // handle the beginning of the last partition
                beg = traits::begin(sit);
                end = traits::local(last);
                if (beg != end)
                {
                    segments.push_back(dispatch_async(traits::get_id(sit),
                        algo, policy, forced_seq(),
                        beg, end, std::forward<F>(f)));
                }
            }

            return result::get(
                dataflow(
                    [=](std::vector<shared_future<value_type> > && r) -> value_type
                    {
                        // handle any remote exceptions, will throw on error
                        std::list<boost::exception_ptr> errors;
                        parallel::util::detail::handle_remote_exceptions<
                            ExPolicy
                        >::call(r, errors);

                        return std::accumulate(
                            r.begin(), r.end(), value_type(),
                            [](value_type const& val, shared_future<value_type>& curr)
                            {
                                return val + curr.get();
                            });
                    },
                    std::move(segments)));
        }

        template <typename ExPolicy, typename InIter, typename F>
        inline typename util::detail::algorithm_result<
            ExPolicy, typename std::iterator_traits<InIter>::difference_type
        >::type
        count_if_(ExPolicy && policy, InIter first, InIter last, F && f,
            std::true_type)
        {
            typedef parallel::is_sequential_execution_policy<ExPolicy>is_seq;

            typedef typename std::iterator_traits<InIter>::difference_type
                difference_type;

            if (first == last)
            {
                return util::detail::algorithm_result<
                    ExPolicy, difference_type>::get(difference_type());
            }

            return segmented_count_if(
                count_if<difference_type>(), std::forward<ExPolicy>(policy),
                first, last, std::forward<F>(f), is_seq());
        }

        // forward declare the non-segmented version of this algorithm
        template <typename ExPolicy, typename InIter, typename F>
        typename util::detail::algorithm_result<
            ExPolicy, typename std::iterator_traits<InIter>::difference_type
        >::type
        count_if_(ExPolicy && policy, InIter first, InIter last, F && f,
            std::false_type);

        /// \endcond
    }
}}}

#endif
