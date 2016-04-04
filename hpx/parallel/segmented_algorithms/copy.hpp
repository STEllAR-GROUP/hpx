//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_SEGMENTED_ALGORITHM_COPY_JAN_05_2014_0125PM)
#define HPX_PARALLEL_SEGMENTED_ALGORITHM_COPY_JAN_05_2014_0125PM

#include <hpx/config.hpp>
#include <hpx/traits/segmented_iterator_traits.hpp>

#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/copy.hpp>
#include <hpx/parallel/segmented_algorithms/detail/dispatch.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/detail/handle_remote_exceptions.hpp>

#include <algorithm>
#include <iterator>
#include <type_traits>
#include <list>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
    ///////////////////////////////////////////////////////////////////////////
    // segmented_copy
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        /// \cond NOINTERNAL

        // sequential remote implementation
        template <typename Algo, typename ExPolicy, typename SegIter,
            typename SegOutIter>
        static typename util::detail::algorithm_result<
            ExPolicy, std::pair<SegIter, SegOutIter>
        >::type
        segmented_copy(Algo && algo, ExPolicy const& policy, std::true_type,
            SegIter first, SegIter last, SegOutIter dest)
        {
            typedef hpx::traits::segmented_iterator_traits<SegIter> traits;
            typedef typename traits::segment_iterator segment_iterator;
            typedef typename traits::local_iterator local_iterator_type;

            typedef hpx::traits::segmented_iterator_traits<SegOutIter>
                output_traits;
            typedef typename output_traits::segment_iterator
                segment_output_iterator;
            typedef typename output_traits::local_iterator
                local_output_iterator_type;

            typedef std::pair<
                    local_iterator_type, local_output_iterator_type
                > local_iterator_pair;

            segment_iterator sit = traits::segment(first);
            segment_iterator send = traits::segment(last);

            segment_output_iterator sdest = traits::segment(dest);

            if (sit == send)
            {
                // all elements are on the same partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::local(last);

                if (beg != end)
                {
                    local_iterator_pair p = dispatch(
                        traits::get_id(sit),
                        algo, policy, std::true_type(),
                        beg, end, traits::local(dest));

                    dest = output_traits::compose(sdest, p.second);
                }
            }
            else {
                // handle the remaining part of the first partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::end(sit);
                local_output_iterator_type out = traits::local(dest);

                if (beg != end)
                {
                    local_iterator_pair p = dispatch(
                        traits::get_id(sit),
                        algo, policy, std::true_type(), beg, end, out);
                    out = p.second;
                }

                // handle all of the full partitions
                for ((void) ++sit, ++sdest; sit != send; (void) ++sit, ++sdest)
                {
                    beg = traits::begin(sit);
                    end = traits::end(sit);
                    out = traits::begin(sdest);

                    if (beg != end)
                    {
                        local_iterator_pair p = dispatch(
                            traits::get_id(sit),
                            algo, policy, std::true_type(), beg, end, out);
                        out = p.second;
                    }
                }

                // handle the beginning of the last partition
                beg = traits::begin(sit);
                end = traits::local(last);
                if (beg != end)
                {
                    local_iterator_pair p = dispatch(
                        traits::get_id(sit),
                        algo, policy, std::true_type(), beg, end,
                        traits::begin(sdest));
                    out = p.second;
                }

                dest = output_traits::compose(sdest, out);
            }

            return util::detail::algorithm_result<
                    ExPolicy, std::pair<SegIter, SegOutIter>
                >::get(std::make_pair(last, dest));
        }

        // parallel remote implementation
        template <typename Algo, typename ExPolicy, typename SegIter,
            typename SegOutIter>
        static typename util::detail::algorithm_result<
            ExPolicy, std::pair<SegIter, SegOutIter>
        >::type
        segmented_copy(Algo && algo, ExPolicy const& policy, std::false_type,
            SegIter first, SegIter last, SegOutIter dest)
        {
            typedef hpx::traits::segmented_iterator_traits<SegIter> traits;
            typedef typename traits::segment_iterator segment_iterator;
            typedef typename traits::local_iterator local_iterator_type;

            typedef hpx::traits::segmented_iterator_traits<SegOutIter>
                output_traits;
            typedef typename output_traits::segment_iterator
                segment_output_iterator;
            typedef typename output_traits::local_iterator
                local_output_iterator_type;

            typedef std::pair<
                    local_iterator_type, local_output_iterator_type
                > local_iterator_pair;

            typedef std::integral_constant<bool,
                    !hpx::traits::is_forward_iterator<SegIter>::value
                > forced_seq;

            segment_iterator sit = traits::segment(first);
            segment_iterator send = traits::segment(last);

            segment_output_iterator sdest = traits::segment(dest);

            std::vector<shared_future<local_iterator_pair> > segments;
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
                        beg, end, traits::local(dest)));
                }
            }
            else {
                // handle the remaining part of the first partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::end(sit);
                local_output_iterator_type out = traits::local(dest);

                if (beg != end)
                {
                    segments.push_back(dispatch_async(traits::get_id(sit),
                        algo, policy, forced_seq(), beg, end, out));
                }

                // handle all of the full partitions
                for ((void) ++sit, ++sdest; sit != send; (void) ++sit, ++sdest)
                {
                    beg = traits::begin(sit);
                    end = traits::end(sit);
                    out = traits::begin(sdest);

                    if (beg != end)
                    {
                        segments.push_back(dispatch_async(traits::get_id(sit),
                            algo, policy, forced_seq(), beg, end, out));
                    }
                }

                // handle the beginning of the last partition
                beg = traits::begin(sit);
                end = traits::local(last);

                if (beg != end)
                {
                    segments.push_back(dispatch_async(traits::get_id(sit),
                        algo, policy, forced_seq(),
                        beg, end, traits::begin(sdest)));
                }
            }
            HPX_ASSERT(!segments.empty());

            return util::detail::algorithm_result<
                    ExPolicy, std::pair<SegIter, SegOutIter> >::get(
                lcos::local::dataflow(
                    [=](std::vector<shared_future<local_iterator_pair> > && r)
                        ->  std::pair<SegIter, SegOutIter>
                    {
                        // handle any remote exceptions, will throw on error
                        std::list<boost::exception_ptr> errors;
                        parallel::util::detail::handle_remote_exceptions<
                            ExPolicy
                        >::call(r, errors);

                        local_iterator_pair p = r.back().get();
                        return std::make_pair(
                                output_traits::compose(sdest, p.first),
                                output_traits::compose(sdest, p.second)
                            );
                    },
                    std::move(segments)));
        }

        ///////////////////////////////////////////////////////////////////////
        // segmented implementation
        template <typename ExPolicy, typename InIter, typename OutIter>
        typename util::detail::algorithm_result<
            ExPolicy, std::pair<InIter, OutIter>
        >::type
        copy_(ExPolicy&& policy, InIter first, InIter last, OutIter dest,
            std::true_type)
        {
            if (first == last)
            {
                return util::detail::algorithm_result<
                        ExPolicy, std::pair<InIter, OutIter>
                    >::get(std::make_pair(last, dest));
            }

            typedef parallel::is_sequential_execution_policy<ExPolicy> is_seq;

            typedef hpx::traits::segmented_iterator_traits<InIter>
                input_iterator_traits;
            typedef hpx::traits::segmented_iterator_traits<OutIter>
                output_iterator_traits;

            typedef std::pair<
                    typename input_iterator_traits::local_iterator,
                    typename output_iterator_traits::local_iterator
                > result_iterator_pair;

            return segmented_copy(
                copy<result_iterator_pair>(),
                std::forward<ExPolicy>(policy), is_seq(),
                first, last, dest);
        }

        // forward declare the non-segmented version of this algorithm
        template <typename ExPolicy, typename InIter, typename OutIter>
        typename util::detail::algorithm_result<
            ExPolicy, std::pair<InIter, OutIter>
        >::type
        copy_(ExPolicy&& policy, InIter first, InIter last, OutIter dest,
            std::false_type);

        /// \endcond
    }
}}}

#endif
