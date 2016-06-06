//  Copyright (c) 2016 Minh-Khanh Do
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_SEGMENTED_ALGORITHM_FILL_MAY_30_2016)
#define HPX_PARALLEL_SEGMENTED_ALGORITHM_FILL_MAY_30_2016

#include <hpx/config.hpp>
#include <hpx/traits/segmented_iterator_traits.hpp>

#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/fill.hpp>
#include <hpx/parallel/segmented_algorithms/detail/dispatch.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/detail/handle_remote_exceptions.hpp>

#include <boost/exception_ptr.hpp>

#include <algorithm>
#include <iterator>
#include <list>
#include <type_traits>
#include <vector>


namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
    ///////////////////////////////////////////////////////////////////////////
    // segmented_fill
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        /// \cond NOINTERNAL

        // sequential remote implementation
        template <typename Algo, typename ExPolicy, typename SegIter, typename T>
        inline typename std::enable_if<
            is_execution_policy<ExPolicy>::value,
            typename util::detail::algorithm_result<ExPolicy, void>::type
        >::type
        segmented_fill(Algo && algo, ExPolicy const& policy,
            SegIter first, SegIter last, T const& value, std::true_type)
        {
            typedef hpx::traits::segmented_iterator_traits<SegIter> traits;
            typedef typename traits::segment_iterator segment_iterator;
            typedef typename traits::local_iterator local_iterator_type;
            typedef util::detail::algorithm_result<ExPolicy, void> result;

            segment_iterator sit = traits::segment(first);
            segment_iterator send = traits::segment(last);

            if (sit == send) {
                // all elements are on the same partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::local(last);

                if (beg != end)
                {
                    dispatch(traits::get_id(sit), algo, policy,
                        std::true_type(), beg, end, value);
                }
            }
            else {
                // handle remaining part of the first partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::end(sit);
                if (beg != end)
                {
                    dispatch(traits::get_id(sit), algo, policy,
                        std::true_type(), beg, end, value);
                }

                // handle all of the full partitions
                for (++sit; sit != send; ++sit)
                {
                    beg = traits::begin(sit);
                    end = traits::end(sit);
                    if (beg != end)
                    {
                        dispatch(traits::get_id(sit), algo, policy,
                            std::true_type(), beg, end, value);
                    }
                }

                // handle the beginning of the last partition
                beg = traits::begin(sit);
                end = traits::local(last);
                if (beg != end)
                {
                    dispatch(traits::get_id(sit), algo, policy,
                        std::true_type(), beg, end, value);
                }
            }

            return result::get();
        }

        // parallel remote implementation
        template <typename Algo, typename ExPolicy, typename SegIter,
            typename T>
        inline typename std::enable_if<
            is_execution_policy<ExPolicy>::value,
            typename util::detail::algorithm_result<ExPolicy, void>::type
        >::type
        segmented_fill(Algo && algo, ExPolicy const& policy,
            SegIter first, SegIter last, T const& value, std::false_type)
        {
            typedef hpx::traits::segmented_iterator_traits<SegIter> traits;
            typedef typename traits::segment_iterator segment_iterator;
            typedef typename traits::local_iterator local_iterator_type;

            typedef util::detail::algorithm_result<ExPolicy, void> result;

            typedef std::integral_constant<bool,
                    !hpx::traits::is_forward_iterator<SegIter>::value
                > forced_seq;

            segment_iterator sit = traits::segment(first);
            segment_iterator send = traits::segment(last);

            std::vector<shared_future<void>> segments;
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
                // handle remaining part of the first partition
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
                    [=](std::vector<shared_future<void>> && r)
                    {
                        // handle any remote exceptions, will throw on error
                        std::list<boost::exception_ptr> errors;
                        parallel::util::detail::handle_remote_exceptions<
                            ExPolicy
                        >::call(r, errors);

                        hpx::util::unwrapped(r);
                        return;
                    },
                    std::move(segments)));
        }

        ///////////////////////////////////////////////////////////////////////
        // segmented implementation
        template <typename ExPolicy, typename InIter, typename T>
        inline typename util::detail::algorithm_result<
            ExPolicy, void
        >::type
        fill_(ExPolicy && policy, InIter first, InIter last, T const& value,
            std::true_type)
        {
            typedef parallel::is_sequential_execution_policy<ExPolicy> is_seq;

            if (first == last)
            {
                return util::detail::algorithm_result<ExPolicy, void>::get();
            }

            return segmented_fill(
                fill(), std::forward<ExPolicy>(policy), first, last, value, is_seq()
            );
        }

        // forward declare the non-segmented version of this algorithm
        template <typename ExPolicy, typename InIter, typename T>
        inline typename util::detail::algorithm_result<
            ExPolicy, void
        >::type
        fill_(ExPolicy && policy, InIter first, InIter last, T const& value,
            std::false_type);
    }
        /// \endcond
}}}

#endif
