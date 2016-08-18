//  Copyright (c) 2016 Minh-Khanh Do
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
#include <hpx/parallel/segmented_algorithms/detail/segmented_scan_partitioner.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/detail/handle_remote_exceptions.hpp>
#include <hpx/parallel/util/detail/handle_local_exceptions.hpp>

#include <algorithm>
#include <iterator>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
    ///////////////////////////////////////////////////////////////////////////
    // inclusive_scan
    namespace detail
    {
        template <typename InIter, typename OutIter, typename T, typename Op>
        OutIter merge_inclusive_scan(InIter first, InIter last,
            OutIter dest, T init, Op && op)
        {
            // add init to each element
            for (/* */; first != last; (void) ++first, ++dest)
            {
                *dest = op(init, *first);
            }
            return dest;
        }

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

                return util::segmented_scan_partitioner<ExPolicy, vector_type,
                    inclusive_scan<typename vector_type::iterator>>
                        ::call(
                    policy,
                    first, last, std::forward<Op>(op));
            }
        };

        template <typename T, typename InIter, typename Op>
        T sequential_inclusive_scan_segmented_ret(InIter first, InIter last,
            Op && op)
        {
            T ret = *first;

            if(first != last) {
                for(++first; first != last; (void) ++first) {
                    ret = op(ret, *first);
                }
            }

            return ret;
        }

        template <typename Value>
        struct inclusive_scan_segmented_ret
            : public detail::algorithm<inclusive_scan_segmented_ret<Value>, Value>
        {
            typedef Value T;

            inclusive_scan_segmented_ret()
                : inclusive_scan_segmented_ret::algorithm("inclusive_scan_segmented_ret")
            {}

            template <typename ExPolicy, typename InIter, typename Op>
            static T
            sequential(ExPolicy && policy, InIter first, InIter last, Op && op)
            {
                return sequential_inclusive_scan_segmented_ret<T>(
                    first, last, std::forward<Op>(op));
            }

            template <typename ExPolicy, typename FwdIter, typename Op>
            static typename util::detail::algorithm_result<
                ExPolicy, T
            >::type
            parallel(ExPolicy policy, FwdIter first, FwdIter last, Op && op)
            {
                typedef typename std::iterator_traits<FwdIter>::value_type value_type;

                return util::partitioner<ExPolicy, T>::call(
                    std::forward<ExPolicy>(policy),
                    first, std::distance(first, last),
                    [&op](FwdIter part_begin, std::size_t part_size) -> T
                    {
                        T ret = *part_begin;
                        if(part_size > 1)
                        {
                            util::loop_n(part_begin+1, part_size-1,
                                [&](FwdIter const& curr)
                                {
                                    ret = op(ret, *curr);
                                });
                        }
                        return ret;
                    },
                    hpx::util::unwrapped(
                        [&op](std::vector<T>&& results)
                        {
                            T ret = *results.begin();

                            if(results.size() > 1) {
                                util::loop_n(results.begin()+1, results.size()-1,
                                    [&](typename std::vector<T>::iterator const& curr)
                                    {
                                        ret = op(ret, *curr);
                                    });
                            }
                            return ret;
                        }
                    ));
            }
        };

        struct inclusive_scan_segmented_void :
            public detail::algorithm<inclusive_scan_segmented_void>
        {
            inclusive_scan_segmented_void()
                : inclusive_scan_segmented_void::algorithm("inclusive_scan_segmented_void")
            {}

            template <typename ExPolicy, typename InIter, typename OutIter, typename T,
                typename Op>
            static hpx::util::unused_type
            sequential(ExPolicy && policy, InIter first, InIter last, OutIter dest,
                T && init, Op && op)
            {
                inclusive_scan<OutIter>().sequential(
                    std::forward<ExPolicy>(policy), first, last, dest,
                    std::forward<T>(init), std::forward<Op>(op));
                return hpx::util::unused;
            }

            template <typename ExPolicy, typename InIter, typename OutIter, typename T,
                typename Op>
            static typename util::detail::algorithm_result<ExPolicy>::type
            parallel(ExPolicy && policy, InIter first, InIter last, OutIter dest,
                T && init, Op && op)
            {
                typedef typename util::detail::algorithm_result<ExPolicy>::type
                    result_type;

                if(first == last)
                    return util::detail::algorithm_result<ExPolicy>::get();


                return hpx::util::void_guard<result_type>(),
                    inclusive_scan<OutIter>().parallel(
                        std::forward<ExPolicy>(policy), first, last, dest,
                        std::forward<T>(init), std::forward<Op>(op));
            }
        };

        // sequential remote implementation
        template <typename Algo, typename ExPolicy, typename SegIter, typename OutIter,
            typename T, typename Op>
        static typename util::detail::algorithm_result<ExPolicy, OutIter>::type
        segmented_inclusive_scan(Algo && algo, ExPolicy && policy, SegIter first,
            SegIter last, OutIter dest, T init, Op && op, std::true_type)
        {
            typedef typename hpx::traits::segmented_iterator_traits<OutIter>
                ::is_segmented_iterator is_out_seg;

            return segmented_inclusive_scan_seq(
                std::forward<Algo>(algo),
                std::forward<ExPolicy>(policy),
                first, last, dest, std::move(init), std::forward<Op>(op),
                is_out_seg());

        }

        // sequential segmented OutIter implementation
        template <typename Algo, typename ExPolicy, typename SegIter, typename OutIter,
            typename T, typename Op>
        static typename util::detail::algorithm_result<ExPolicy, OutIter>::type
        segmented_inclusive_scan_seq(Algo && algo, ExPolicy const& policy, SegIter first,
            SegIter last, OutIter dest, T init, Op && op, std::true_type)
        {
            typedef util::detail::algorithm_result<ExPolicy, OutIter> result;

            if (first == last)
                return result::get(std::move(dest));

            typedef hpx::traits::segmented_iterator_traits<SegIter> traits_in;
            typedef typename traits_in::segment_iterator segment_iterator_in;
            typedef typename traits_in::local_iterator local_iterator_type_in;

            typedef hpx::traits::segmented_iterator_traits<OutIter> traits_out;
            typedef typename traits_out::segment_iterator segment_iterator_out;
            typedef typename traits_out::local_iterator local_iterator_type_out;

            typedef typename hpx::util::tuple<
                    local_iterator_type_in, local_iterator_type_in
                > local_iterator_in_tuple;

            segment_iterator_in sit_in = traits_in::segment(first);
            segment_iterator_in send_in = traits_in::segment(last);

            segment_iterator_out sit_out = traits_out::segment(dest);

            std::vector<T> results;
            std::vector<local_iterator_in_tuple> in_iters;
            std::vector<segment_iterator_out> out_iters;

            // 1. Step: scan on each partition, get last T of the scan
            OutIter temp_dest = dest;
            if (sit_in == send_in)
            {
                // all elements on the same partition
                local_iterator_type_in beg = traits_in::local(first);
                local_iterator_type_in end = traits_in::end(sit_in);
                if (beg != end)
                {
                    results.push_back(dispatch(traits_in::get_id(sit_in),
                        inclusive_scan_segmented_ret<T>(), policy,
                        std::true_type(), beg, end, op));
                    in_iters.push_back(hpx::util::make_tuple(beg, end));
                    out_iters.push_back(sit_out);
                }
            }
            else
            {
                // handle the remaining part of the first partition
                local_iterator_type_in beg = traits_in::local(first);
                local_iterator_type_in end = traits_in::end(sit_in);

                if (beg != end)
                {
                    results.push_back(dispatch(traits_in::get_id(sit_in),
                        inclusive_scan_segmented_ret<T>(), policy,
                        std::true_type(), beg, end, op));
                    in_iters.push_back(hpx::util::make_tuple(beg, end));
                    out_iters.push_back(sit_out);
                }

                // handle all partitions
                for(++sit_in, ++sit_out; sit_in != send_in; ++sit_in, ++sit_out) {
                    beg = traits_in::begin(sit_in);
                    end = traits_in::end(sit_in);
                    if (beg != end)
                    {
                        results.push_back(dispatch(traits_in::get_id(sit_in),
                            inclusive_scan_segmented_ret<T>(), policy,
                            std::true_type(), beg, end, op));
                        in_iters.push_back(hpx::util::make_tuple(beg, end));
                        out_iters.push_back(sit_out);
                    }
                }

                // handle the beginning of the last partition
                beg = traits_in::begin(sit_in);
                end = traits_in::local(last);
                if (beg != end)
                {
                    results.push_back(dispatch(traits_in::get_id(sit_in),
                        inclusive_scan_segmented_ret<T>(), policy,
                        std::true_type(), beg, end, op));
                    in_iters.push_back(hpx::util::make_tuple(beg, end));
                    out_iters.push_back(sit_out);
                }
            }

            // first init value is the given init value
            T last_value = init;
            for (std::size_t i = 0; i < results.size(); ++i) {
                using hpx::util::get;
                local_iterator_type_out out = traits_out::begin(out_iters[i]);

                // 2. Step: use the init values to dispatch final scan for each segment
                dispatch(traits_out::get_id(out_iters[i]),
                    inclusive_scan_segmented_void(),
                    policy, std::true_type(),
                    get<0>(in_iters[i]), get<1>(in_iters[i]),
                    out, last_value, op);

                // 3. Step: compute new init values for the next segment
                last_value += results[i];
            }

            OutIter final_dest = dest;
            std::advance(final_dest, std::distance(first,last));

            return result::get(std::move(final_dest));

        }


        // sequential non segmented OutIter implementation
        template <typename Algo, typename ExPolicy, typename SegIter, typename OutIter,
            typename T, typename Op>
        static typename util::detail::algorithm_result<ExPolicy, OutIter>::type
        segmented_inclusive_scan_seq(Algo && algo, ExPolicy const& policy, SegIter first,
            SegIter last, OutIter dest, T init, Op && op, std::false_type)
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
        template <typename Algo, typename ExPolicy, typename SegIter, typename OutIter,
            typename T, typename Op>
        static typename util::detail::algorithm_result<ExPolicy, OutIter>::type
        segmented_inclusive_scan(Algo && algo, ExPolicy&& policy, SegIter first,
            SegIter last, OutIter dest, T init, Op && op, std::false_type)
        {
            typedef typename hpx::traits::segmented_iterator_traits<OutIter>
                ::is_segmented_iterator is_out_seg;


            return segmented_inclusive_scan_par(
                std::forward<Algo>(algo),
                std::forward<ExPolicy>(policy),
                first, last, dest, std::move(init), std::forward<Op>(op),
                is_out_seg());
        }

        // parallel segmented OutIter implementation
        template <typename Algo, typename ExPolicy, typename SegIter, typename OutIter,
            typename T, typename Op>
        static typename util::detail::algorithm_result<ExPolicy, OutIter>::type
        segmented_inclusive_scan_par(Algo && algo, ExPolicy const& policy, SegIter first,
            SegIter last, OutIter dest, T init, Op && op, std::true_type)
        {
            typedef util::detail::algorithm_result<ExPolicy, OutIter> result;

            if (first == last)
                return result::get(std::move(dest));

            typedef hpx::traits::segmented_iterator_traits<SegIter> traits_in;
            typedef typename traits_in::segment_iterator segment_iterator_in;
            typedef typename traits_in::local_iterator local_iterator_type_in;

            typedef hpx::traits::segmented_iterator_traits<OutIter> traits_out;
            typedef typename traits_out::segment_iterator segment_iterator_out;
            typedef typename traits_out::local_iterator local_iterator_type_out;

            typedef typename std::iterator_traits<segment_iterator_in>::difference_type
                difference_type;

            typedef std::integral_constant<bool,
                    !hpx::traits::is_forward_iterator<SegIter>::value
                > forced_seq;

            typedef typename hpx::util::tuple<
                    local_iterator_type_in, local_iterator_type_in
                > local_iterator_in_tuple;

            segment_iterator_in sit_in = traits_in::segment(first);
            segment_iterator_in send_in = traits_in::segment(last);

            segment_iterator_out sit_out = traits_out::segment(dest);

            difference_type count = std::distance(sit_in, send_in);

            std::vector<shared_future<T>> results;
            std::vector<local_iterator_in_tuple> in_iters;
            std::vector<segment_iterator_out> out_iters;

            results.reserve(count);
            in_iters.reserve(count);
            out_iters.reserve(count);

            // 1. Step: scan on each partition, get last T of the scan
            if (sit_in == send_in)
            {
                // all elements on the same partition
                local_iterator_type_in beg = traits_in::local(first);
                local_iterator_type_in end = traits_in::end(sit_in);
                if (beg != end)
                {
                    in_iters.push_back(hpx::util::make_tuple(beg, end));
                    out_iters.push_back(sit_out);
                    results.push_back(dispatch_async(traits_in::get_id(sit_in),
                        inclusive_scan_segmented_ret<T>(), policy,
                        forced_seq(), beg, end, op));
                }
            }
            else
            {
                // handle the remaining part of the first partition
                local_iterator_type_in beg = traits_in::local(first);
                local_iterator_type_in end = traits_in::end(sit_in);

                if (beg != end)
                {
                    in_iters.push_back(hpx::util::make_tuple(beg, end));
                    out_iters.push_back(sit_out);
                    results.push_back(dispatch_async(traits_in::get_id(sit_in),
                        inclusive_scan_segmented_ret<T>(), policy,
                        forced_seq(), beg, end, op));
                }

                // handle all partitions
                for(++sit_in, ++sit_out; sit_in != send_in; ++sit_in, ++sit_out) {
                    beg = traits_in::begin(sit_in);
                    end = traits_in::end(sit_in);
                    if (beg != end)
                    {
                        in_iters.push_back(hpx::util::make_tuple(beg, end));
                        out_iters.push_back(sit_out);
                        results.push_back(dispatch_async(traits_in::get_id(sit_in),
                            inclusive_scan_segmented_ret<T>(), policy,
                            forced_seq(), beg, end, op));
                    }
                }

                // handle the beginning of the last partition
                beg = traits_in::begin(sit_in);
                end = traits_in::local(last);
                if (beg != end)
                {
                    in_iters.push_back(hpx::util::make_tuple(beg, end));
                    out_iters.push_back(sit_out);
                    results.push_back(dispatch_async(traits_in::get_id(sit_in),
                        inclusive_scan_segmented_ret<T>(), policy,
                        forced_seq(), beg, end, op));
                }
            }

            std::vector<shared_future<T>> workitems;
            workitems.reserve(results.size()+1);

            std::vector<hpx::future<void>> finalitems;
            finalitems.reserve(results.size());

            // first init value is the given init value
            workitems.push_back(make_ready_future(std::forward<T>(init)));

            std::size_t i = 0;

            for (auto const& res : results) {
                using hpx::util::get;
                segment_iterator_out out_it = out_iters[i];
                local_iterator_type_out out = traits_out::begin(out_it);
                local_iterator_in_tuple in_tuple = in_iters[i];

                // 2. Step: use the init values to dispatch final scan for each segment
                // performed as soon as the init values are ready
                // wait for 1. step of current partition to prevent race condition
                // when used in place
                finalitems.push_back(
                    dataflow(
                        policy.executor(),
                        hpx::util::unwrapped(
                            [=, &op](T last_value, T)
                            {
                                dispatch(traits_out::get_id(out_it),
                                    inclusive_scan_segmented_void(),
                                    hpx::parallel::seq, std::true_type(),
                                    get<0>(in_tuple), get<1>(in_tuple),
                                    out, last_value, op);
                            }
                        ), workitems.back(), res
                    )
                );

                // 3. Step: compute new init value for the next segment
                // performed as soon as the needed results are ready
                workitems.push_back(
                    dataflow(
                        policy.executor(),
                        hpx::util::unwrapped(op),
                        workitems.back(),
                        res
                    )
                );
                ++i;
            }


            OutIter final_dest = dest;
            std::advance(final_dest, std::distance(first,last));

            // wait for all tasks to finish
            return result::get(
                dataflow(
                    [final_dest](
                        std::vector<shared_future<T>> &&r,
                        std::vector<shared_future<T>> &&wi,
                        std::vector<hpx::future<void>> &&fi
                    ) mutable -> OutIter
                    {
                        return final_dest;
                    },
                    std::move(results), std::move(workitems), std::move(finalitems)));
        }


        // parallel non-segmented OutIter implementation
        template <typename Algo, typename ExPolicy, typename SegIter, typename OutIter,
            typename T, typename Op>
        static typename util::detail::algorithm_result<ExPolicy, OutIter>::type
        segmented_inclusive_scan_par(Algo && algo, ExPolicy const& policy, SegIter first,
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
