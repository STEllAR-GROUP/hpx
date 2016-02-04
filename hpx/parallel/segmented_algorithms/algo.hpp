// Copyright (c) 2016 Minh-Khanh Do
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/segmented_algorithms/algo.hpp

#if !defined(HPX_PARALLEL_DETAIL_SEGMENTED_ALGORITHM_ALGO)
#define HPX_PARALLEL_DETAIL_SEGMENTED_ALGORITHM_ALGO

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/move.hpp>
#include <hpx/traits/segmented_iterator_traits.hpp>

#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/detail/is_negative.hpp>
#include <hpx/parallel/segmented_algorithms/detail/dispatch.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/detail/handle_remote_exceptions.hpp>

#include <algorithm>
#include <iterator>
#include <type_traits>
#include <iostream>

#include <boost/type_traits/is_same.hpp>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
    ///////////////////////////////////////////////////////////////////////////
    // segmented algo
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        /// \cond NOINTERNAL

        // sequential remote implementation
        template <typename SegAlgo, typename ExPolicy, typename SegIter,
                                    typename SegOutIter>
        static typename util::detail::algorithm_result<
            ExPolicy, SegOutIter
        >::type
        segmented_algo(SegAlgo && algo, ExPolicy const& policy, boost::mpl::true_,
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

            segment_iterator sit = traits::segment(first);
            segment_iterator send = traits::segment(last);

            segment_output_iterator sdest = traits::segment(dest);

            using boost::mpl::true_;

            if (sit == send)
            {
                // all elements are on the same partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::local(last);

                if (beg != end)
                {
                    local_output_iterator_type out = dispatch(
                        traits::get_id(sit),
                        algo, policy, true_(),
                        beg, end, traits::local(dest));

                    dest = output_traits::compose(sdest, out);
                }
            }
            else {
                // handle the remaining part of the first partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::end(sit);
                local_output_iterator_type out = traits::local(dest);

                if (beg != end)
                {
                    out = dispatch(traits::get_id(sit),
                                   algo, policy, true_(), beg, end, out);
                }

                // handle all of the full partitions
                for ((void) ++sit, ++sdest; sit != send; (void) ++sit, ++sdest)
                {
                    beg = traits::begin(sit);
                    end = traits::end(sit);
                    out = traits::begin(sdest);

                    if (beg != end)
                    {
                        out = dispatch(traits::get_id(sit),
                                       algo, policy, true_(), beg, end, out);
                    }
                }

                // handle the beginning of the last partition
                beg = traits::begin(sit);
                end = traits::local(last);
                if (beg != end)
                {
                    out = dispatch(traits::get_id(sit),
                                   algo, policy, true_(), beg, end, traits::begin(sdest));
                }

                dest = output_traits::compose(sdest, out);
            }

            return util::detail::algorithm_result<ExPolicy, SegOutIter>::get(
                std::move(dest));
        }

        // parallel remote implementation
        template <typename SegAlgo, typename ExPolicy, typename SegIter,
                                    typename SegOutIter>
        static typename util::detail::algorithm_result<
            ExPolicy, SegOutIter
        >::type
        segmented_algo(SegAlgo && algo, ExPolicy const& policy, boost::mpl::false_,
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

            typedef typename std::iterator_traits<SegIter>::iterator_category
                iterator_category;
            typedef typename boost::mpl::bool_<boost::is_same<
                iterator_category, std::input_iterator_tag
            >::value> forced_seq;

            segment_iterator sit = traits::segment(first);
            segment_iterator send = traits::segment(last);

            segment_output_iterator sdest = traits::segment(dest);

            std::vector<shared_future<local_output_iterator_type> > segments;
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
            else
            {
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

            return util::detail::algorithm_result<ExPolicy, SegOutIter>::get(
                lcos::local::dataflow(
                    [=](std::vector<shared_future<local_output_iterator_type> > && r)
                        ->  SegOutIter
                    {
                        // handle any remote exceptions, will throw on error
                        std::list<boost::exception_ptr> errors;
                        parallel::util::detail::handle_remote_exceptions<
                            ExPolicy
                        >::call(r, errors);
                        return output_traits::compose(sdest, r.back().get());
                    },
                    std::move(segments)));
        }

        ///////////////////////////////////////////////////////////////////////
        // segmented implementation
        template <typename ParAlgo, typename SegAlgo, typename ExPolicy, typename InIter, typename OutIter>
        typename util::detail::algorithm_result<ExPolicy, OutIter>::type
        algo_(ExPolicy&& policy, InIter first, InIter last, OutIter dest,
              std::true_type)
        {
            if (first == last)
            {
                return util::detail::algorithm_result<ExPolicy, OutIter>::get(
                    std::move(dest));
            }

            typedef typename parallel::is_sequential_execution_policy<
                ExPolicy
            >::type is_seq;
            typedef hpx::traits::segmented_iterator_traits<OutIter>
                output_iterator_traits;

            return segmented_algo(
                SegAlgo(),
                std::forward<ExPolicy>(policy), is_seq(),
                first, last, dest);
        }
    }
}}}

#endif
