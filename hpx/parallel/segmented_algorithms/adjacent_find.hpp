//  Copyright (c) 2017 Ajai V George
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_SEGMENTED_ALGORITHM_ADJACENT_FIND_AUG_22_2017_1157AM)
#define HPX_PARALLEL_SEGMENTED_ALGORITHM_ADJACENT_FIND_AUG_22_2017_1157AM

#include <hpx/config.hpp>
#include <hpx/traits/segmented_iterator_traits.hpp>

#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/adjacent_find.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/segmented_algorithms/detail/dispatch.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/detail/handle_remote_exceptions.hpp>

#include <algorithm>
#include <exception>
#include <iterator>
#include <list>
#include <cstddef>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace parallel { inline namespace v1
{
    ///////////////////////////////////////////////////////////////////////////
    // segmented_adjacent_find
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        /// \cond NOINTERNAL

        // sequential remote implementation
        template <typename Algo, typename ExPolicy, typename FwdIter,
            typename Pred>
        static typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
        segmented_adjacent_find(Algo && algo, ExPolicy const& policy,
            FwdIter first, FwdIter last, Pred && op, std::true_type)
        {
            typedef hpx::traits::segmented_iterator_traits<FwdIter> traits;
            typedef typename traits::segment_iterator segment_iterator1;
            typedef typename traits::local_iterator local_iterator_type;

            typedef util::detail::algorithm_result<ExPolicy, FwdIter> result;

            segment_iterator1 sit = traits::segment(first);
            segment_iterator1 send = traits::segment(last);

            if (sit == send)
            {
                // all elements are on the same partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::end(sit);
                if (beg != end)
                {
                    local_iterator_type out =
                        dispatch(traits::get_id(sit), algo, policy,
                            std::true_type(), beg, end, op);
                }
            }
            else {
                // handle the remaining part of the first partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::end(sit);
                local_iterator_type out = traits::local(last);
                if (beg != end)
                {
                    out = dispatch(traits::get_id(sit), algo, policy,
                        std::true_type(), beg, end, op);
                }

                // handle all of the full partitions
                for (++sit; sit != send; ++sit)
                {
                    beg = traits::begin(sit);
                    end = traits::end(sit);
                    if (beg != end)
                    {
                        out = dispatch(traits::get_id(sit), algo, policy,
                            std::true_type(), beg, end, op);
                    }
                }

                // handle the beginning of the last partition
                beg = traits::begin(sit);
                end = traits::local(last);
                if (beg != end)
                {
                    out = dispatch(traits::get_id(sit), algo, policy,
                        std::true_type(), beg, end, op);
                }
                last = traits::compose(sit, out);
            }
            return result::get(std::move(last));
        }

        // parallel remote implementation
        template <typename Algo, typename ExPolicy, typename FwdIter,
            typename Pred>
        static typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
        segmented_adjacent_find(Algo && algo, ExPolicy const& policy,
            FwdIter first, FwdIter last, Pred && op, std::false_type)
        {
            typedef hpx::traits::segmented_iterator_traits<FwdIter> traits;
            typedef typename traits::segment_iterator segment_iterator1;
            typedef typename traits::local_iterator local_iterator_type;

            typedef util::detail::algorithm_result<ExPolicy, FwdIter> result;

            typedef std::integral_constant<bool,
                    !hpx::traits::is_forward_iterator<FwdIter>::value
                > forced_seq;

            segment_iterator1 sit = traits::segment(first);
            segment_iterator1 send = traits::segment(last);

            typedef std::vector<future<local_iterator_type> > segment_type;
            segment_type segments;
            segments.reserve(std::distance(sit, send));

            if (sit == send)
            {
                // all elements are on the same partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::local(last);
                if (beg != end)
                {
                    segments.push_back(
                        dispatch_async(traits::get_id(sit),
                            algo, policy, forced_seq(),
                            beg, end, op)
                    );
                }
            }
            else {
                // handle the remaining part of the first partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::end(sit);
                if (beg != end)
                {
                    segments.push_back(
                        dispatch_async(traits::get_id(sit),
                            algo, policy, forced_seq(),
                            beg, end, op)
                    );
                }

                // handle all of the full partitions
                for (++sit; sit != send; ++sit)
                {
                    beg = traits::begin(sit);
                    end = traits::end(sit);
                    if (beg != end)
                    {
                        segments.push_back(
                            dispatch_async(traits::get_id(sit),
                                algo, policy, forced_seq(),
                                beg, end, op)
                        );
                    }
                }

                // handle the beginning of the last partition
                beg = traits::begin(sit);
                end = traits::local(last);
                if (beg != end)
                {
                    segments.push_back(
                        dispatch_async(traits::get_id(sit),
                            algo, policy, forced_seq(),
                            beg, end, op)
                    );
                }
            }

            return result::get(
                dataflow(
                    [=](segment_type && r) -> FwdIter
                    {
                        // handle any remote exceptions, will throw on error
                        std::list<std::exception_ptr> errors;
                        parallel::util::detail::handle_remote_exceptions<
                            ExPolicy
                        >::call(r, errors);
                        auto ft = r.back().get();
                        auto output = traits::compose(sit, ft);
                        return output;
                    },
                    std::move(segments)));
        }

        ///////////////////////////////////////////////////////////////////////
        // segmented implementation
        template <typename ExPolicy, typename FwdIter, typename Pred>
        typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
        adjacent_find_(ExPolicy&& policy, FwdIter first, FwdIter last,
            Pred && op, std::true_type)
        {
            typedef parallel::execution::is_sequenced_execution_policy<
                    ExPolicy
                > is_seq;
            typedef util::detail::algorithm_result<ExPolicy, FwdIter> result;

            if (first == last)
            {
                return result::get(std::move(last));
            }

            typedef hpx::traits::segmented_iterator_traits<FwdIter>
                iterator_traits;

            return segmented_adjacent_find(
                adjacent_find<typename iterator_traits::local_iterator>(),
                std::forward<ExPolicy>(policy), first, last,
                std::forward<Pred>(op), is_seq());
        }

        // forward declare the non-segmented version of this algorithm
        template <typename ExPolicy, typename FwdIter, typename Pred>
        typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
        adjacent_find_(ExPolicy&& policy, FwdIter first, FwdIter last,
            Pred && op, std::false_type);
        /// \endcond
    }
}}}

#endif
