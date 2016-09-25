//  Copyright (c) 2016 Minh-Khanh Do
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/segmented_algorithms/detail/scan.hpp

#if !defined(HPX_PARALLEL_SEGMENTED_ALGORITHMS_SCAN)
#define HPX_PARALLEL_SEGMENTED_ALGORITHMS_SCAN

#include <hpx/config.hpp>
#include <hpx/lcos/local/dataflow.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/traits/is_iterator.hpp>
#include <hpx/traits/segmented_iterator_traits.hpp>
#include <hpx/util/invoke.hpp>
#include <hpx/util/tuple.hpp>
#include <hpx/util/unused.hpp>
#include <hpx/util/unwrapped.hpp>
#include <hpx/util/void_guard.hpp>

#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/executors/executor_traits.hpp>
#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/segmented_algorithms/detail/dispatch.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/loop.hpp>

#include <algorithm>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
    ///////////////////////////////////////////////////////////////////////////
    // segmented scan
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        /// \cond NOINTERNAL

        // returns the last value of the scan
        // used to compute the next init value
        template <typename T, typename InIter, typename Op>
        T sequential_segmented_scan_T(InIter first, InIter last, Op && op)
        {
            T ret = *first;
            if (first != last)
            {
                for(++first; first != last; ++first)
                {
                    ret = hpx::util::invoke(op, ret, *first);
                }
            }
            return ret;
        }

        // does a scan and returns last value of the scan
        template <typename Value>
        struct segmented_scan_T
          : public detail::algorithm<segmented_scan_T<Value>, Value>
        {
            typedef Value T;

            segmented_scan_T()
              : segmented_scan_T::algorithm("segmented_scan_T")
            {}

            template <typename ExPolicy, typename InIter, typename Op>
            static T
            sequential(ExPolicy && policy, InIter first, InIter last, Op && op)
            {
                return sequential_segmented_scan_T<T>(
                    first, last, std::forward<Op>(op));
            }

            template <typename ExPolicy, typename FwdIter, typename Op>
            static typename util::detail::algorithm_result<
                ExPolicy, T
            >::type
            parallel(ExPolicy && policy, FwdIter first, FwdIter last, Op && op)
            {
                return util::partitioner<ExPolicy, T>::call(
                    std::forward<ExPolicy>(policy),
                    first, std::distance(first, last),
                    [op, policy](FwdIter part_begin, std::size_t part_size) -> T
                    {
                        T ret = *part_begin;
                        if(part_size > 1)
                        {
                            // MSVC complains if 'op' is captured by reference
                            util::loop_n(
                                policy, part_begin+1, part_size-1,
                                [&ret, op](FwdIter const& curr)
                                {
                                    ret = hpx::util::invoke(op, ret, *curr);
                                });
                        }
                        return ret;
                    },
                    hpx::util::unwrapped(
                        [op, policy](std::vector<T>&& results) -> T
                        {
                            T ret = *results.begin();
                            if(results.size() > 1)
                            {
                                // MSVC complains if 'op' is captured by reference
                                util::loop_n(
                                    policy,
                                    results.begin()+1, results.size()-1,
                                    [&ret, op](
                                        typename std::vector<T>::iterator const& curr
                                    )
                                    {
                                        ret = hpx::util::invoke(op, ret, *curr);
                                    });
                            }
                            return ret;
                        }
                    ));
            }
        };

        // do the scan (exclusive/inclusive)
        // does not return anything
        template <typename Algo>
        struct segmented_scan_void :
            public detail::algorithm<segmented_scan_void<Algo>>
        {
            segmented_scan_void()
              : segmented_scan_void::algorithm("segmented_scan_void")
            {}

            template <typename ExPolicy, typename InIter,
                typename OutIter, typename T, typename Op>
            static hpx::util::unused_type
            sequential(ExPolicy && policy, InIter first,
                InIter last, OutIter dest, T && init, Op && op)
            {
                Algo().sequential(
                    std::forward<ExPolicy>(policy), first, last, dest,
                    std::forward<T>(init), std::forward<Op>(op));

                return hpx::util::unused;
            }

            template <typename ExPolicy, typename InIter,
                typename OutIter, typename T, typename Op>
            static typename util::detail::algorithm_result<ExPolicy>::type
            parallel(ExPolicy && policy, InIter first,
                InIter last, OutIter dest, T && init, Op && op)
            {
                typedef typename util::detail::algorithm_result<ExPolicy>::type
                    result_type;

                if(first == last)
                    return util::detail::algorithm_result<ExPolicy>::get();

                return hpx::util::void_guard<result_type>(),
                    Algo().parallel(
                        std::forward<ExPolicy>(policy), first, last, dest,
                        std::forward<T>(init), std::forward<Op>(op));
            }
        };

        template <typename SegIter, typename OutIter>
        static bool is_segmented_the_same(SegIter first, SegIter last,
            OutIter dest, std::false_type)
        {
            return false;
        }

        // check if two segmented iterators are partitioned the same
        // partition size and id should be the same
        template <typename SegIter, typename OutIter>
        static bool is_segmented_the_same(SegIter first, SegIter last,
            OutIter dest, std::true_type)
        {
            typedef hpx::traits::segmented_iterator_traits<SegIter> traits_in;
            typedef typename traits_in::segment_iterator segment_iterator_in;
            typedef typename traits_in::local_iterator local_iterator_type_in;

            typedef hpx::traits::segmented_iterator_traits<OutIter> traits_out;
            typedef typename traits_out::segment_iterator segment_iterator_out;
            typedef typename traits_out::local_iterator local_iterator_type_out;

            segment_iterator_in sit_in = traits_in::segment(first);
            segment_iterator_in send_in = traits_in::segment(last);

            segment_iterator_out sit_out = traits_out::segment(dest);

            using hpx::naming::get_locality_from_id;

            if (sit_in == send_in)
            {
                // all elements on the same partition
                local_iterator_type_in beg_in = traits_in::local(first);
                local_iterator_type_in end_in = traits_in::end(sit_in);

                local_iterator_type_out beg_out = traits_out::local(dest);
                local_iterator_type_out end_out = traits_out::end(sit_out);

                if(beg_in != end_in)
                {
                    id_type in_id = get_locality_from_id(
                        traits_in::get_id(sit_in));
                    id_type out_id = get_locality_from_id(
                        traits_out::get_id(sit_out));

                    if (in_id != out_id)
                        return false;

                    std::size_t in_dist = std::distance(beg_in, end_in);
                    std::size_t out_dist = std::distance(beg_out, end_out);

                    if (in_dist != out_dist)
                        return false;
                }
            }
            else
            {
                // handle the remaining part of the first partition
                local_iterator_type_in beg_in = traits_in::local(first);
                local_iterator_type_in end_in = traits_in::end(sit_in);

                local_iterator_type_out beg_out = traits_out::local(dest);
                local_iterator_type_out end_out = traits_out::end(sit_out);

                if(beg_in != end_in)
                {
                    id_type in_id = get_locality_from_id(
                        traits_in::get_id(sit_in));
                    id_type out_id = get_locality_from_id(
                        traits_out::get_id(sit_out));

                    if (in_id != out_id)
                        return false;

                    std::size_t in_dist = std::distance(beg_in, end_in);
                    std::size_t out_dist = std::distance(beg_out, end_out);

                    if (in_dist != out_dist)
                        return false;
                }

                // handle all partitions
                for(++sit_in, ++sit_out; sit_in != send_in; ++sit_in, ++sit_out)
                {
                    beg_in = traits_in::begin(sit_in);
                    end_in = traits_in::end(sit_in);

                    beg_out = traits_out::begin(sit_out);
                    end_out = traits_out::end(sit_out);

                    if(beg_in != end_in)
                    {
                        id_type in_id = get_locality_from_id(
                            traits_in::get_id(sit_in));
                        id_type out_id = get_locality_from_id(
                            traits_out::get_id(sit_out));

                        if (in_id != out_id)
                            return false;

                        std::size_t in_dist = std::distance(beg_in, end_in);
                        std::size_t out_dist = std::distance(beg_out, end_out);

                        if (in_dist != out_dist)
                            return false;
                    }
                }

                // handle the last partition
                beg_in = traits_in::begin(sit_in);
                end_in = traits_in::end(sit_in);

                beg_out = traits_out::begin(sit_out);
                end_out = traits_out::end(sit_out);

                if (beg_in != end_in)
                {
                    id_type in_id = get_locality_from_id(
                        traits_in::get_id(sit_in));
                    id_type out_id = get_locality_from_id(
                        traits_out::get_id(sit_out));

                    if (in_id != out_id)
                        return false;

                    std::size_t in_dist = std::distance(beg_in, end_in);
                    std::size_t out_dist = std::distance(beg_out, end_out);

                    if (in_dist != out_dist)
                        return false;
                }
            }
            return true;
        }

        ///////////////////////////////////////////////////////////////////////
        // sequential implementation

        // sequential segmented OutIter implementation
        template <typename Algo, typename ExPolicy, typename SegIter,
            typename OutIter, typename T, typename Op>
        static typename util::detail::algorithm_result<ExPolicy, OutIter>::type
        segmented_scan_seq(ExPolicy const& policy, SegIter first,
            SegIter last, OutIter dest, T const& init, Op && op, std::true_type)
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

            // 1. Step: scan on each partition, push last T of scan into results
            if (sit_in == send_in)
            {
                // all elements on the same partition
                local_iterator_type_in beg = traits_in::local(first);
                local_iterator_type_in end = traits_in::end(sit_in);
                if (beg != end)
                {
                    results.push_back(dispatch(traits_in::get_id(sit_in),
                        segmented_scan_T<T>(), policy,
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
                        segmented_scan_T<T>(), policy,
                        std::true_type(), beg, end, op));
                    in_iters.push_back(hpx::util::make_tuple(beg, end));
                    out_iters.push_back(sit_out);
                }

                // handle all partitions
                for(++sit_in, ++sit_out; sit_in != send_in; ++sit_in, ++sit_out)
                {
                    beg = traits_in::begin(sit_in);
                    end = traits_in::end(sit_in);
                    if (beg != end)
                    {
                        results.push_back(dispatch(traits_in::get_id(sit_in),
                            segmented_scan_T<T>(), policy,
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
                        segmented_scan_T<T>(), policy,
                        std::true_type(), beg, end, op));
                    in_iters.push_back(hpx::util::make_tuple(beg, end));
                    out_iters.push_back(sit_out);
                }
            }

            // first init value is the given init value
            T last_value = init;
            for (std::size_t i = 0; i < results.size(); ++i)
            {
                using hpx::util::get;
                local_iterator_type_out out = traits_out::begin(out_iters[i]);

                // 2. Step: use the init values to dispatch final scan for each
                // segment
                dispatch(traits_out::get_id(out_iters[i]),
                    segmented_scan_void<Algo>(),
                    policy, std::true_type(),
                    get<0>(in_iters[i]), get<1>(in_iters[i]),
                    out, last_value, op);

                // 3. Step: compute new init values for the next segment
                last_value = op(results[i], last_value);
            }

            OutIter final_dest = dest;
            std::advance(final_dest, std::distance(first,last));

            return result::get(std::move(final_dest));

        }


        // sequential non segmented OutIter implementation
        template <typename Algo, typename ExPolicy, typename SegIter,
            typename OutIter, typename T, typename Op, typename F1, typename F2>
        static typename util::detail::algorithm_result<ExPolicy, OutIter>::type
        segmented_scan_seq_non(ExPolicy const& policy, SegIter first,
            SegIter last, OutIter dest, T const& init, Op && op, F1 && f1, F2 && f2)
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

            // scan on each partition, push whole result vector into results
            if (sit == send)
            {
                // all elements on the same partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::end(sit);
                if (beg != end)
                {
                    results.push_back(dispatch(traits::get_id(sit),
                        Algo(), policy, std::true_type(), beg, end, op));
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
                        Algo(), policy, std::true_type(), beg, end, op));
                }

                // handle all partitions
                for(++sit; sit != send; ++sit)
                {
                    beg = traits::begin(sit);
                    end = traits::end(sit);
                    if (beg != end)
                    {
                        results.push_back(dispatch(traits::get_id(sit),
                            Algo(), policy, std::true_type(), beg, end, op));
                    }
                }

                // handle the beginning of the last partition
                beg = traits::begin(sit);
                end = traits::local(last);
                if (beg != end)
                {
                    results.push_back(dispatch(traits::get_id(sit),
                        Algo(), policy, std::true_type(), beg, end, op));
                }
            }

            // merge results with given merge algorithm f1
            // update init value with function f2
            T last_value = init;
            for (auto res : results)
            {
                dest = f1(res.begin(), res.end(),
                    dest, last_value, op);
                last_value = f2(res, last_value);
            }
            return result::get(std::move(dest));
        }

        ///////////////////////////////////////////////////////////////////////
        // parallel implementation

        // parallel segmented OutIter implementation
        template <typename Algo, typename ExPolicy, typename SegIter,
            typename OutIter, typename T, typename Op>
        static typename util::detail::algorithm_result<ExPolicy, OutIter>::type
        segmented_scan_par(ExPolicy const& policy, SegIter first,
            SegIter last, OutIter dest, T const& init, Op && op, std::true_type)
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

            std::vector<hpx::shared_future<T> > results;
            std::vector<local_iterator_in_tuple> in_iters;
            std::vector<segment_iterator_out> out_iters;

            results.reserve(count);
            in_iters.reserve(count);
            out_iters.reserve(count);

            // 1. Step: scan on each partition, push last T of scan into results
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
                        segmented_scan_T<T>(), policy,
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
                        segmented_scan_T<T>(), policy,
                        forced_seq(), beg, end, op));
                }

                // handle all partitions
                for(++sit_in, ++sit_out; sit_in != send_in; ++sit_in, ++sit_out)
                {
                    beg = traits_in::begin(sit_in);
                    end = traits_in::end(sit_in);
                    if (beg != end)
                    {
                        in_iters.push_back(hpx::util::make_tuple(beg, end));
                        out_iters.push_back(sit_out);
                        results.push_back(dispatch_async(traits_in::get_id(sit_in),
                            segmented_scan_T<T>(), policy,
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
                        segmented_scan_T<T>(), policy,
                        forced_seq(), beg, end, op));
                }
            }

            std::vector<hpx::shared_future<T> > workitems;
            workitems.reserve(results.size()+1);

            std::vector<hpx::future<void>> finalitems;
            finalitems.reserve(results.size());

            // first init value is the given init value
            workitems.push_back(make_ready_future(init));

            std::size_t i = 0;

            for (auto const& res : results)
            {
                using hpx::util::get;
                segment_iterator_out out_it = out_iters[i];
                local_iterator_type_out out = traits_out::begin(out_it);
                local_iterator_in_tuple in_tuple = in_iters[i];

                // 2. Step: use the init values to dispatch final scan for each
                // segment performed as soon as the init values are ready
                // wait for 1. step of current partition to prevent race condition
                // when used in place
                finalitems.push_back(
                    hpx::dataflow(
                        policy.executor(),
                        hpx::util::unwrapped(
                            [=, &op](T last_value, T)
                            {
                                dispatch(traits_out::get_id(out_it),
                                    segmented_scan_void<Algo>(),
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
                    hpx::dataflow(
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
                hpx::dataflow(
                    [final_dest](
                        std::vector<hpx::shared_future<T> > &&r,
                        std::vector<hpx::shared_future<T> > &&wi,
                        std::vector<hpx::future<void> > &&fi
                    ) mutable -> OutIter
                    {
                        return final_dest;
                    },
                    std::move(results), std::move(workitems),
                    std::move(finalitems)));
        }

        // parallel non-segmented OutIter implementation
        template <typename Algo, typename ExPolicy, typename SegIter,
            typename OutIter, typename T, typename Op, typename F1, typename F2>
        static typename util::detail::algorithm_result<ExPolicy, OutIter>::type
        segmented_scan_par_non(ExPolicy const& policy, SegIter first,
            SegIter last, OutIter dest, T const& init, Op && op, F1 && f1, F2 && f2)
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
            std::vector<hpx::shared_future<vector_type> > results;
            results.reserve(count);

            std::vector<std::size_t> segment_sizes;
            segment_sizes.reserve(count);

            OutIter final_dest = dest;
            std::advance(final_dest, std::distance(first, last));

            // scan on each partition, push whole result vector into results
            if (sit == send)
            {
                // all elements on the same partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::end(sit);
                if (beg != end)
                {
                    results.push_back(dispatch_async(traits::get_id(sit),
                        Algo(), policy, forced_seq(), beg, end, op));
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
                        Algo(), policy, forced_seq(), beg, end, op));
                    segment_sizes.push_back(std::distance(beg, end));
                }

                // handle all partitions
                for(++sit; sit != send; ++sit)
                {
                    beg = traits::begin(sit);
                    end = traits::end(sit);
                    if (beg != end)
                    {
                        results.push_back(dispatch_async(traits::get_id(sit),
                        Algo(), policy, forced_seq(), beg, end, op));
                        segment_sizes.push_back(std::distance(beg, end));
                    }
                }

                // handle the beginning of the last partition
                beg = traits::begin(sit);
                end = traits::local(last);
                if (beg != end)
                {
                    results.push_back(dispatch_async(traits::get_id(sit),
                        Algo(), policy, forced_seq(), beg, end, op));
                    segment_sizes.push_back(std::distance(beg, end));
                }
            }

            std::vector<hpx::shared_future<T> > workitems;
            workitems.reserve(results.size()+1);

            std::vector<hpx::future<void>> finalitems;
            finalitems.reserve(results.size());

            typedef typename hpx::util::decay<ExPolicy>::type::executor_type
                executor_type;
            typedef typename hpx::parallel::executor_traits<executor_type>
                executor_traits;

            workitems.push_back(make_ready_future(init));

            std::size_t segment_index = 0;

            for (auto const& res : results)
            {
                // collect all results with updated init values
                finalitems.push_back(
                    hpx::dataflow(
                        policy.executor(),
                        hpx::util::unwrapped(
                            [&, dest](T last_value, vector_type && r)
                            {
                                // merge function
                                f1(r.begin(), r.end(), dest, last_value, op);
                            }
                        ), workitems.back(), res
                    )
                );

                std::advance(dest, segment_sizes[segment_index++]);

                // propagate results from left to right
                // new init value is most right value combined with old init
                workitems.push_back(
                    hpx::dataflow(
                        policy.executor(),
                        hpx::util::unwrapped(op),
                        workitems.back(),
                        executor_traits::async_execute(
                            policy.executor(),
                            hpx::util::unwrapped(f2),
                            res
                        )
                    )
                );
            }

            // wait for all tasks to finish
            return result::get(
                hpx::dataflow(
                    [final_dest](
                        std::vector<hpx::shared_future<vector_type> > &&r,
                        std::vector<hpx::shared_future<T> > &&wi,
                        std::vector<hpx::future<void> > &&fi
                    ) mutable -> OutIter
                    {
                        return final_dest;
                    },
                    std::move(results), std::move(workitems),
                    std::move(finalitems)));
        }
        /// \endcond
    }
}}}

#endif
