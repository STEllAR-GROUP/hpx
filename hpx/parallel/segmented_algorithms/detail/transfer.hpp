//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_SEGMENTED_ALGORITHMS_TRANSFER)
#define HPX_PARALLEL_SEGMENTED_ALGORITHMS_TRANSFER

#include <hpx/config.hpp>
#include <hpx/lcos/local/dataflow.hpp>
#include <hpx/traits/segmented_iterator_traits.hpp>

#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/segmented_algorithms/detail/dispatch.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/detail/handle_remote_exceptions.hpp>

#include <algorithm>
#include <exception>
#include <iterator>
#include <list>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace parallel { inline namespace v1
{
    ///////////////////////////////////////////////////////////////////////////
    // segmented transfer
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        /// \cond NOINTERNAL

        ///////////////////////////////////////////////////////////////////////
        template <typename FwdIter, typename OutIter>
        struct iterators_are_segmented
          : std::integral_constant<bool,
                hpx::traits::segmented_iterator_traits<FwdIter>
                    ::is_segmented_iterator::value &&
                hpx::traits::segmented_iterator_traits<OutIter>
                    ::is_segmented_iterator::value>
        {};

        template <typename FwdIter, typename OutIter>
        struct iterators_are_not_segmented
          : std::integral_constant<bool,
                !hpx::traits::segmented_iterator_traits<FwdIter>
                    ::is_segmented_iterator::value &&
                !hpx::traits::segmented_iterator_traits<OutIter>
                    ::is_segmented_iterator::value>
        {};

        ///////////////////////////////////////////////////////////////////////
        // sequential remote implementation
        template <typename Algo, typename ExPolicy, typename SegIter,
            typename SegOutIter>
        static typename util::detail::algorithm_result<
            ExPolicy, std::pair<SegIter, SegOutIter>
        >::type
        segmented_transfer(Algo && algo, ExPolicy const& policy, std::true_type,
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
        segmented_transfer(Algo && algo, ExPolicy const& policy, std::false_type,
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
                    ExPolicy, std::pair<SegIter, SegOutIter>
                >::get(hpx::dataflow(
                    [=](std::vector<shared_future<local_iterator_pair> > && r)
                        ->  std::pair<SegIter, SegOutIter>
                    {
                        // handle any remote exceptions, will throw on error
                        std::list<std::exception_ptr> errors;
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
        template <typename Algo, typename ExPolicy, typename FwdIter1,
            typename FwdIter2>
        typename util::detail::algorithm_result<
            ExPolicy, std::pair<FwdIter1, FwdIter2>
        >::type
        transfer_(ExPolicy&& policy, FwdIter1 first, FwdIter1 last,
            FwdIter2 dest, std::true_type)
        {
            if (first == last)
            {
                return util::detail::algorithm_result<
                        ExPolicy, std::pair<FwdIter1, FwdIter2>
                    >::get(std::make_pair(last, dest));
            }

            typedef parallel::execution::is_sequenced_execution_policy<
                    ExPolicy
                > is_seq;
            return segmented_transfer(Algo(),
                std::forward<ExPolicy>(policy), is_seq(),
                first, last, dest);
        }

        // forward declare the non-segmented version of this algorithm
        template <typename Algo, typename ExPolicy, typename FwdIter1,
            typename FwdIter2>
        typename util::detail::algorithm_result<
            ExPolicy, std::pair<FwdIter1, FwdIter2>
        >::type
        transfer_(ExPolicy&& policy, FwdIter1 first, FwdIter1 last,
            FwdIter2 dest, std::false_type);

        /// \endcond
    }
}}}

#endif
