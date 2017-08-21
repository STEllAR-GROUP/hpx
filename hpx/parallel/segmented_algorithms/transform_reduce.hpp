//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_SEGMENTED_ALGORITHM_TRANSFORM_REDUCE_DEC_17_2014_1157AM)
#define HPX_PARALLEL_SEGMENTED_ALGORITHM_TRANSFORM_REDUCE_DEC_17_2014_1157AM

#include <hpx/config.hpp>
#include <hpx/traits/segmented_iterator_traits.hpp>
#include <hpx/util/invoke.hpp>

#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/transform_reduce.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/segmented_algorithms/detail/dispatch.hpp>
#include <hpx/parallel/segmented_algorithms/detail/reduce.hpp>
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

namespace hpx { namespace parallel { inline namespace v1
{
    ///////////////////////////////////////////////////////////////////////////
    // segmented_transform_reduce
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        /// \cond NOINTERNAL

        // sequential remote implementation
        template <typename Algo, typename ExPolicy, typename FwdIter,
            typename T, typename Reduce, typename Convert>
        static typename util::detail::algorithm_result<ExPolicy, T>::type
        segmented_transform_reduce(Algo && algo, ExPolicy const& policy,
            FwdIter first, FwdIter last, T && init,
            Reduce && red_op, Convert && conv_op, std::true_type)
        {
            typedef hpx::traits::segmented_iterator_traits<FwdIter> traits;
            typedef typename traits::segment_iterator segment_iterator;
            typedef typename traits::local_iterator local_iterator_type;
            typedef util::detail::algorithm_result<ExPolicy, T> result;

            segment_iterator sit = traits::segment(first);
            segment_iterator send = traits::segment(last);

            T overall_result = init;

            if (sit == send)
            {
                // all elements are on the same partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::local(last);
                if (beg != end)
                {
                    overall_result = hpx::util::invoke(red_op, overall_result,
                        dispatch(traits::get_id(sit), algo, policy,
                            std::true_type(), beg, end, red_op, conv_op));
                }
            }
            else {
                // handle the remaining part of the first partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::end(sit);
                if (beg != end)
                {
                    overall_result = hpx::util::invoke(red_op, overall_result,
                        dispatch(traits::get_id(sit), algo, policy,
                            std::true_type(), beg, end, red_op, conv_op)
                    );
                }

                // handle all of the full partitions
                for (++sit; sit != send; ++sit)
                {
                    beg = traits::begin(sit);
                    end = traits::end(sit);
                    if (beg != end)
                    {
                        overall_result = hpx::util::invoke(red_op, overall_result,
                            dispatch(traits::get_id(sit), algo, policy,
                                std::true_type(), beg, end, red_op, conv_op)
                        );
                    }
                }

                // handle the beginning of the last partition
                beg = traits::begin(sit);
                end = traits::local(last);
                if (beg != end)
                {
                    overall_result = hpx::util::invoke(red_op, overall_result,
                        dispatch(traits::get_id(sit), algo, policy,
                            std::true_type(), beg, end, red_op, conv_op)
                    );
                }
            }

            return result::get(std::move(overall_result));
        }

        // parallel remote implementation
        template <typename Algo, typename ExPolicy, typename FwdIter,
            typename T, typename Reduce, typename Convert>
        static typename util::detail::algorithm_result<ExPolicy, T>::type
        segmented_transform_reduce(Algo && algo, ExPolicy const& policy,
            FwdIter first, FwdIter last, T && init,
            Reduce && red_op, Convert && conv_op, std::false_type)
        {
            typedef hpx::traits::segmented_iterator_traits<FwdIter> traits;
            typedef typename traits::segment_iterator segment_iterator;
            typedef typename traits::local_iterator local_iterator_type;
            typedef util::detail::algorithm_result<ExPolicy, T> result;

            typedef std::integral_constant<bool,
                    !hpx::traits::is_forward_iterator<FwdIter>::value
                > forced_seq;

            segment_iterator sit = traits::segment(first);
            segment_iterator send = traits::segment(last);

            std::vector<shared_future<T> > segments;
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
                            beg, end, red_op, conv_op)
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
                            beg, end, red_op, conv_op)
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
                                beg, end, red_op, conv_op)
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
                            beg, end, red_op, conv_op)
                    );
                }
            }

            return result::get(
                dataflow(
                    [=](std::vector<shared_future<T> > && r) -> T
                    {
                        // handle any remote exceptions, will throw on error
                        std::list<std::exception_ptr> errors;
                        parallel::util::detail::handle_remote_exceptions<
                            ExPolicy
                        >::call(r, errors);

                        // VS2015RC bails out if red_op is capture by ref
                        return std::accumulate(
                            r.begin(), r.end(), init,
                            [=](T const& val, shared_future<T>& curr)
                            {
                                return red_op(val, curr.get());
                            });
                    },
                    std::move(segments)));
        }

        ///////////////////////////////////////////////////////////////////////
        // segmented implementation
        template <typename ExPolicy, typename InIter, typename T, typename Reduce,
            typename Convert>
        typename util::detail::algorithm_result<
            ExPolicy, typename hpx::util::decay<T>::type
        >::type
        transform_reduce_(ExPolicy&& policy, InIter first, InIter last, T && init,
            Reduce && red_op, Convert && conv_op, std::true_type)
        {
            typedef parallel::execution::is_sequenced_execution_policy<
                    ExPolicy
                > is_seq;
            typedef typename hpx::util::decay<T>::type init_type;

            if (first == last)
            {
                return util::detail::algorithm_result<
                        ExPolicy, init_type
                    >::get(std::forward<T>(init));
            }

            return segmented_transform_reduce(
                seg_transform_reduce<init_type>(),
                std::forward<ExPolicy>(policy), first, last,
                std::forward<T>(init), std::forward<Reduce>(red_op),
                std::forward<Convert>(conv_op), is_seq());
        }

        // forward declare the non-segmented version of this algorithm
        template <typename ExPolicy, typename InIter, typename T, typename Reduce,
            typename Convert>
        typename util::detail::algorithm_result<
            ExPolicy, typename hpx::util::decay<T>::type
        >::type
        transform_reduce_(ExPolicy&& policy, InIter first, InIter last, T && init,
            Reduce && red_op, Convert && conv_op, std::false_type);

        // sequential remote implementation
        template <typename Algo, typename ExPolicy, typename FwdIter1,
            typename FwdIter2, typename T, typename Reduce, typename Convert>
        static typename util::detail::algorithm_result<ExPolicy, T>::type
        segmented_transform_reduce(Algo && algo, ExPolicy const& policy,
            FwdIter1 first1, FwdIter1 last1, FwdIter2 first2, T && init,
            Reduce && red_op, Convert && conv_op, std::true_type)
        {
            typedef hpx::traits::segmented_iterator_traits<FwdIter1> traits1;
            typedef typename traits1::segment_iterator segment_iterator1;
            typedef typename traits1::local_iterator local_iterator_type1;
            typedef hpx::traits::segmented_iterator_traits<FwdIter2> traits2;
            typedef typename traits2::segment_iterator segment_iterator2;
            typedef typename traits2::local_iterator local_iterator_type2;
            typedef util::detail::algorithm_result<ExPolicy, T> result;

            auto last2 = first2;
            detail::advance(last2,std::distance(first1,last1));

            segment_iterator1 sit1 = traits1::segment(first1);
            segment_iterator1 send1 = traits1::segment(last1);
            segment_iterator2 sit2 = traits1::segment(first2);
            segment_iterator2 send2 = traits1::segment(last2);

            T overall_result = init;

            if (sit1 == send1)
            {
                // all elements are on the same partition
                local_iterator_type1 beg1 = traits1::local(first1);
                local_iterator_type2 beg2 = traits2::local(first2);
                local_iterator_type1 end1 = traits1::local(last1);
                if (beg1 != end1)
                {
                    overall_result = hpx::util::invoke(red_op, overall_result,
                        dispatch(traits1::get_id(sit1), algo, policy,
                            std::true_type(), beg1, end1, beg2, red_op, conv_op));
                }
            }
            else {
                // handle the remaining part of the first1 partition
                local_iterator_type1 beg1 = traits1::local(first1);
                local_iterator_type2 beg2 = traits2::local(first1);
                local_iterator_type1 end1 = traits1::end(sit1);
                if (beg1 != end1)
                {
                    overall_result = hpx::util::invoke(red_op, overall_result,
                        dispatch(traits1::get_id(sit1), algo, policy,
                            std::true_type(), beg1, end1, beg2, red_op, conv_op)
                    );
                }

                // handle all of the full partitions
                for (++sit1, ++sit2; sit1 != send1; ++sit1, ++sit2)
                {
                    beg1 = traits1::begin(sit1);
                    beg2 = traits2::begin(sit2);
                    end1 = traits1::end(sit1);
                    if (beg1 != end1)
                    {
                        overall_result = hpx::util::invoke(red_op, overall_result,
                            dispatch(traits1::get_id(sit1), algo, policy,
                                std::true_type(), beg1, end1, beg2, red_op, conv_op)
                        );
                    }
                }

                // handle the beginning of the last1 partition
                beg1 = traits1::begin(sit1);
                beg2 = traits2::begin(sit2);
                end1 = traits1::local(last1);
                if (beg1 != end1)
                {
                    overall_result = hpx::util::invoke(red_op, overall_result,
                        dispatch(traits1::get_id(sit1), algo, policy,
                            std::true_type(), beg1, end1, beg2, red_op, conv_op)
                    );
                }
            }

            return result::get(std::move(overall_result));
        }

        // parallel remote implementation
        template <typename Algo, typename ExPolicy, typename FwdIter1,
            typename FwdIter2, typename T, typename Reduce, typename Convert>
        static typename util::detail::algorithm_result<ExPolicy, T>::type
        segmented_transform_reduce(Algo && algo, ExPolicy const& policy,
            FwdIter1 first1, FwdIter1 last1, FwdIter2 first2, T && init,
            Reduce && red_op, Convert && conv_op, std::false_type)
        {
            typedef hpx::traits::segmented_iterator_traits<FwdIter1> traits1;
            typedef typename traits1::segment_iterator segment_iterator1;
            typedef typename traits1::local_iterator local_iterator_type1;
            typedef hpx::traits::segmented_iterator_traits<FwdIter2> traits2;
            typedef typename traits2::segment_iterator segment_iterator2;
            typedef typename traits2::local_iterator local_iterator_type2;
            typedef util::detail::algorithm_result<ExPolicy, T> result;

            auto last2 = first2;
            detail::advance(last2,std::distance(first1,last1));

            segment_iterator1 sit1 = traits1::segment(first1);
            segment_iterator1 send1 = traits1::segment(last1);
            segment_iterator2 sit2 = traits1::segment(first2);
            segment_iterator2 send2 = traits1::segment(last2);

            typedef std::integral_constant<bool,
                    !hpx::traits::is_forward_iterator<FwdIter1>::value
                > forced_seq;


            std::vector<shared_future<T> > segments;
            segments.reserve(std::distance(sit1, send1));

            if (sit1 == send1)
            {
                // all elements are on the same partition
                local_iterator_type1 beg1 = traits1::local(first1);
                local_iterator_type2 beg2 = traits2::local(first2);
                local_iterator_type1 end1 = traits1::local(last1);
                if (beg1 != end1)
                {
                    segments.push_back(
                        dispatch_async(traits1::get_id(sit1),
                            algo, policy, forced_seq(),
                            beg1, end1, beg2, red_op, conv_op)
                    );
                }
            }
            else {
                // handle the remaining part of the first1 partition
                local_iterator_type1 beg1 = traits1::local(first1);
                local_iterator_type2 beg2 = traits2::local(first2);
                local_iterator_type1 end1 = traits1::end(sit1);
                if (beg1 != end1)
                {
                    segments.push_back(
                        dispatch_async(traits1::get_id(sit1),
                            algo, policy, forced_seq(),
                            beg1, end1, beg2, red_op, conv_op)
                    );
                }

                // handle all of the full partitions
                for (++sit1, ++sit2; sit1 != send1; ++sit1, ++sit2)
                {
                    beg1 = traits1::begin(sit1);
                    beg2 = traits2::begin(sit2);
                    end1 = traits1::end(sit1);
                    if (beg1 != end1)
                    {
                        segments.push_back(
                            dispatch_async(traits1::get_id(sit1),
                                algo, policy, forced_seq(),
                                beg1, end1, beg2, red_op, conv_op)
                        );
                    }
                }

                // handle the beginning of the last1 partition
                beg1 = traits1::begin(sit1);
                beg2 = traits2::begin(sit2);
                end1= traits1::local(last1);
                if (beg1 != end1)
                {
                    segments.push_back(
                        dispatch_async(traits1::get_id(sit1),
                            algo, policy, forced_seq(),
                            beg1, end1, beg2, red_op, conv_op)
                    );
                }
            }

            return result::get(
                dataflow(
                    [=](std::vector<shared_future<T> > && r) -> T
                    {
                        // handle any remote exceptions, will throw on error
                        std::list<std::exception_ptr> errors;
                        parallel::util::detail::handle_remote_exceptions<
                            ExPolicy
                        >::call(r, errors);

                        // VS2015RC bails out if red_op is capture by ref
                        return std::accumulate(
                            r.begin(), r.end(), init,
                            [=](T const& val, shared_future<T>& curr)
                            {
                                return red_op(val, curr.get());
                            });
                    },
                    std::move(segments)));
        }

        template <typename ExPolicy, typename FwdIter1, typename FwdIter2, typename T,
            typename Reduce, typename Convert>
        typename util::detail::algorithm_result<ExPolicy, T>::type
        transform_reduce_(ExPolicy && policy, FwdIter1 first1, FwdIter1 last1,
            FwdIter2 first2, T init, Reduce && red_op, Convert && conv_op, std::true_type)
        {
            typedef parallel::execution::is_sequenced_execution_policy<
                    ExPolicy
                > is_seq;
            typedef typename hpx::util::decay<T>::type init_type;

            if (first1 == last1)
            {
                return util::detail::algorithm_result<
                        ExPolicy, init_type
                    >::get(std::forward<T>(init));
            }

            return segmented_transform_reduce(
                seg_transform_reduce_binary<init_type>(),
                std::forward<ExPolicy>(policy), first1, last1, first2,
                std::forward<T>(init), std::forward<Reduce>(red_op),
                std::forward<Convert>(conv_op), is_seq());
        }

        template <typename ExPolicy, typename FwdIter1, typename FwdIter2, typename T,
            typename Reduce, typename Convert>
        typename util::detail::algorithm_result<ExPolicy, T>::type
        transform_reduce_(ExPolicy && policy, FwdIter1 first1, FwdIter1 last1,
            FwdIter2 first2, T init, Reduce && red_op, Convert && conv_op, std::false_type);
        /// \endcond
    }
}}}

#endif
