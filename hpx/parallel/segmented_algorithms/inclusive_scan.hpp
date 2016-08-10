//  Copyright (c) 2014-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/segmented_algorithms/inclusive_scan.hpp

#if !defined(HPX_PARALLEL_SEGMENTED_ALGORITHM_INCLUSIVE_SCAN)
#define HPX_PARALLEL_SEGMENTED_ALGORITHM_INCLUSIVE_SCAN

#include <hpx/config.hpp>

#include <hpx/traits/segmented_iterator_traits.hpp>

#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/inclusive_scan.hpp>
#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/segmented_algorithms/detail/dispatch.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/detail/handle_remote_exceptions.hpp>
#include <hpx/parallel/util/detail/handle_local_exceptions.hpp>

#include <algorithm>
#include <iterator>
#include <numeric>
#include <type_traits>
#include <vector>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
    ///////////////////////////////////////////////////////////////////////////
    // inclusive_scan
    namespace detail
    {
        template <typename ExPolicy, typename Result, typename Algo, typename Tag>
        struct segmented_scan_partitioner1;

        template <typename ExPolicy, typename Result, typename Algo>
        struct segmented_scan_partitioner1<ExPolicy, Result, Algo,
            parallel::traits::static_partitioner_tag>
        {
            template <typename ExPolicy_, typename InIter, typename Op>
            static Result call(ExPolicy_ && policy,
                InIter first, InIter last, Op && op)
            {
                typedef typename std::iterator_traits<InIter>::value_type value_type;

                Result result(std::distance(first, last));

                if (result.size() != 0) {
                    result[0] = *first;

                    Algo::parallel(policy, first+1, last, result.begin()+1,
                        std::forward<value_type>(*first), op);
                }

                return result;
            }
        };

        template <typename Result, typename Algo>
        struct segmented_scan_partitioner1<parallel_task_execution_policy, Result, Algo,
            parallel::traits::static_partitioner_tag>
        {
            template <typename ExPolicy, typename InIter, typename Op>
            static hpx::future<Result> call(ExPolicy && policy,
                InIter first, InIter last, Op && op)
            {
                typedef typename std::iterator_traits<InIter>::value_type value_type;

                Result result(std::distance(first, last));
                if (result.size() != 0) {
                    result[0] = *first;
                }

                return dataflow(
                    [=]() mutable
                    {
                        Algo::parallel(policy,
                            first+1, last, result.begin()+1,
                            std::forward<value_type>(*first), std::forward<Op>(op)).wait();

                        return result;
                    });
            }
        };

        template <typename ExPolicy, typename Result, typename Algo>
        struct segmented_scan_partitioner1<ExPolicy, Result, Algo,
                parallel::traits::default_partitioner_tag>
            : segmented_scan_partitioner1<ExPolicy, Result, Algo,
                parallel::traits::static_partitioner_tag>
        {};

        template <typename ExPolicy, typename Result, typename Algo,
            typename PartTag = typename parallel::traits::extract_partitioner<
                typename hpx::util::decay<ExPolicy>::type
            >::type>
        struct segmented_scan_partitioner
            : segmented_scan_partitioner1<
                typename hpx::util::decay<ExPolicy>::type, Result, Algo, PartTag>
        {};

        template <typename Value>
        struct inclusive_scan_segmented
            : public detail::algorithm<inclusive_scan_segmented<Value>, Value>
        {
            typedef Value vector_type;

            inclusive_scan_segmented()
                : inclusive_scan_segmented::algorithm("inclusive_scan_segmented")
            {}

            template <typename ExPolicy, typename InIter, typename Op>
            static vector_type
            sequential(ExPolicy && policy, InIter first, InIter last, Op && op)
            {
                typedef typename std::iterator_traits<InIter>::value_type value_type;

                vector_type result(std::distance(first, last));

                // use first element as init value
                if (result.size() != 0) {
                    result[0] = *first;
                    inclusive_scan<typename vector_type::iterator>().sequential(
                        hpx::parallel::seq, first+1, last, result.begin()+1,
                        std::forward<value_type>(*first), std::forward<Op>(op));
                }
                return result;
            }

            template <typename ExPolicy, typename FwdIter, typename Op>
            static typename util::detail::algorithm_result<
                ExPolicy, vector_type
            >::type
            parallel(ExPolicy policy, FwdIter first, FwdIter last, Op && op)
            {
                typedef typename std::iterator_traits<FwdIter>::value_type value_type;

                return segmented_scan_partitioner<ExPolicy, vector_type,
                    inclusive_scan<typename vector_type::iterator>>
                        ::call(
                    policy,
                    first, last, std::forward<Op>(op));
            }
        };

        template <typename InIter, typename OutIter, typename T, typename Op>
        OutIter merge_inclusive_scan(InIter first, InIter last,
            OutIter dest, T init, Op && op)
        {
            for (/* */; first != last; (void) ++first, ++dest)
            {
                *dest = op(init, *first);
            }
            return dest;
        }

        // sequential remote implementation
        template <typename Algo, typename ExPolicy, typename SegIter, typename OutIter, typename T,
            typename Op>
        static typename util::detail::algorithm_result<ExPolicy, OutIter>::type
        segmented_inclusive_scan(Algo && algo, ExPolicy const& policy, SegIter first,
            SegIter last, OutIter dest, T init, Op && op, std::true_type)
        {
            typedef util::detail::algorithm_result<ExPolicy, OutIter> result;

            if (first == last)
                return result::get(std::move(dest));

            typedef hpx::traits::segmented_iterator_traits<SegIter> traits;
            typedef typename traits::segment_iterator segment_iterator;
            typedef typename traits::local_iterator local_iterator_type;

            segment_iterator sit = traits::segment(first);
            segment_iterator send = traits::segment(last);

            typedef typename std::vector<T> vector_type;

            std::vector<vector_type> results;

            OutIter temp_dest = dest;
            if (sit == send)
            {
                // all elements on the same partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::end(sit);
                if (beg != end)
                {
                    results.push_back(dispatch(traits::get_id(sit),
                        inclusive_scan_segmented<vector_type>(), policy,
                        std::true_type(), beg, end, op));
                }
            }
            else
            {
                // handle the remaining part of the first partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::end(sit);

                if (beg != end)
                {
                     results.push_back(dispatch(traits::get_id(sit),
                        inclusive_scan_segmented<vector_type>(), policy,
                        std::true_type(), beg, end, op));
                }

                // handle all partitions
                for(++sit; sit != send; ++sit) {
                    beg = traits::begin(sit);
                    end = traits::end(sit);
                    if (beg != end)
                    {
                        results.push_back(dispatch(traits::get_id(sit),
                            inclusive_scan_segmented<vector_type>(), policy,
                            std::true_type(), beg, end, op));
                    }
                }

                // handle the beginning of the last partition
                beg = traits::begin(sit);
                end = traits::local(last);
                if (beg != end)
                {
                    results.push_back(dispatch(traits::get_id(sit),
                        inclusive_scan_segmented<vector_type>(), policy,
                        std::true_type(), beg, end, op));
                }
            }

            // merge results
            T last_value = init;
            for (auto res : results) {
                dest = merge_inclusive_scan(res.begin(), res.end(),
                    dest, last_value, op);
                last_value = *(dest-1);
            }

            return result::get(std::move(dest));
        }

        // parallel remote implementation
        template <typename Algo, typename ExPolicy, typename SegIter, typename OutIter, typename T,
            typename Op>
        static typename util::detail::algorithm_result<ExPolicy, OutIter>::type
        segmented_inclusive_scan(Algo && algo, ExPolicy const& policy, SegIter first,
            SegIter last, OutIter dest, T init, Op && op, std::false_type)
        {
            typedef util::detail::algorithm_result<ExPolicy, OutIter> result;

            if (first == last)
                return result::get(std::move(dest));

            typedef hpx::traits::segmented_iterator_traits<SegIter> traits;
            typedef typename traits::segment_iterator segment_iterator;
            typedef typename traits::local_iterator local_iterator_type;
            typedef typename std::iterator_traits<segment_iterator>::difference_type
                difference_type;

            typedef std::integral_constant<bool,
                    !hpx::traits::is_forward_iterator<SegIter>::value
                > forced_seq;

            segment_iterator sit = traits::segment(first);
            segment_iterator send = traits::segment(last);

            difference_type count = std::distance(sit, send);


            typedef typename std::vector<T> vector_type;
            std::vector<shared_future<vector_type>> results;
            results.reserve(count);

            std::vector<std::size_t> segment_sizes;
            segment_sizes.reserve(count);

            OutIter final_dest = dest;
            std::advance(final_dest, std::distance(first, last));

            // dispatch scan on each segment
            if (sit == send)
            {
                // all elements on the same partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::end(sit);
                if (beg != end)
                {
                    results.push_back(dispatch_async(traits::get_id(sit),
                        inclusive_scan_segmented<vector_type>(), policy,
                        forced_seq(), beg, end, op));
                    segment_sizes.push_back(std::distance(beg, end));
                }
            }
            else
            {
                // handle the remaining part of the first partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::end(sit);

                if (beg != end)
                {
                     results.push_back(dispatch_async(traits::get_id(sit),
                        inclusive_scan_segmented<vector_type>(), policy,
                        forced_seq(), beg, end, op));
                    segment_sizes.push_back(std::distance(beg, end));
                }

                // handle all partitions
                for(++sit; sit != send; ++sit) {
                    beg = traits::begin(sit);
                    end = traits::end(sit);
                    if (beg != end)
                    {
                        results.push_back(dispatch_async(traits::get_id(sit),
                            inclusive_scan_segmented<vector_type>(), policy,
                            forced_seq(), beg, end, op));
                        segment_sizes.push_back(std::distance(beg, end));
                    }
                }

                // handle the beginning of the last partition
                beg = traits::begin(sit);
                end = traits::local(last);
                if (beg != end)
                {
                    results.push_back(dispatch_async(traits::get_id(sit),
                        inclusive_scan_segmented<vector_type>(), policy,
                        forced_seq(), beg, end, op));
                    segment_sizes.push_back(std::distance(beg, end));
                }
            }

            std::vector<shared_future<T>> workitems;
            workitems.reserve(results.size()+1);

            std::vector<hpx::future<void>> finalitems;
            finalitems.reserve(results.size());

            typedef typename hpx::util::decay<ExPolicy>::type::executor_type
                executor_type;
            typedef typename hpx::parallel::executor_traits<executor_type>
                executor_traits;

            workitems.push_back(make_ready_future(std::forward<T>(init)));

            std::size_t segment_index = 0;

            for (auto const& res : results) {

                // collect all results with updated init values
                finalitems.push_back(
                    dataflow(
                        policy.executor(),
                        hpx::util::unwrapped(
                            [&, dest](T last_value, vector_type r)
                            {
                                merge_inclusive_scan(r.begin(),
                                    r.end(), dest, last_value, op);
                            }
                        ), workitems.back(), res
                    )
                );

                std::advance(dest, segment_sizes[segment_index++]);

                // propagate results from left to right
                // new init value is most right value combined with old init
                workitems.push_back(
                    dataflow(
                        policy.executor(),
                        hpx::util::unwrapped(op),
                        workitems.back(),
                        executor_traits::async_execute(
                            policy.executor(),
                            hpx::util::unwrapped(
                                [&](vector_type r)
                                {
                                    return *(r.end()-1);
                                }
                            ), res
                        )
                    )
                );
            }

            // wait for all tasks to finish
            return result::get(
                dataflow(
                    [final_dest](
                        std::vector<shared_future<vector_type>> &&r,
                        std::vector<shared_future<T>> &&wi,
                        std::vector<hpx::future<void>> &&fi
                    ) mutable -> OutIter
                    {
                        return final_dest;
                    },
                    std::move(results), std::move(workitems), std::move(finalitems)));

        }

        ///////////////////////////////////////////////////////////////////////
        // segmented implementation
        template <typename ExPolicy, typename InIter, typename OutIter, typename T,
            typename Op>
        static typename util::detail::algorithm_result<ExPolicy, OutIter>::type
        inclusive_scan_(ExPolicy&& policy, InIter first, InIter last, OutIter dest,
            T init, Op && op, std::true_type)
        {
            typedef parallel::is_sequential_execution_policy<ExPolicy> is_seq;

            if (first == last)
                return util::detail::algorithm_result<
                    ExPolicy, OutIter>::get(std::move(dest));

            return segmented_inclusive_scan(
                inclusive_scan_segmented<std::vector<T>>(),
                std::forward<ExPolicy>(policy),
                first, last, dest, std::move(init), std::forward<Op>(op), is_seq());
        }

        // forward declare the non-segmented version of this algorithm
        template <typename ExPolicy, typename InIter, typename OutIter, typename T,
            typename Op>
        static typename util::detail::algorithm_result<ExPolicy, OutIter>::type
        inclusive_scan_(ExPolicy&& policy, InIter first, InIter last, OutIter dest,
            T init, Op && op, std::true_type);

        /// \endcond
    }
}}}
#endif
