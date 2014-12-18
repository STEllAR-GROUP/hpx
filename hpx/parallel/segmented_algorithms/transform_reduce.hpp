//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_SEGMENTED_ALGORITHM_TRANSFORM_REDUCE_DEC_17_2014_1157AM)
#define HPX_PARALLEL_SEGMENTED_ALGORITHM_TRANSFORM_REDUCE_DEC_17_2014_1157AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/void_guard.hpp>
#include <hpx/util/move.hpp>
#include <hpx/traits/segmented_iterator_traits.hpp>

#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/algorithm_result.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/detail/is_negative.hpp>
#include <hpx/parallel/algorithms/remote/dispatch.hpp>
#include <hpx/parallel/algorithms/transform_reduce.hpp>

#include <algorithm>
#include <iterator>

#include <boost/type_traits/is_same.hpp>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
    ///////////////////////////////////////////////////////////////////////////
    // segmented_transform_reduce
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        /// \cond NOINTERNAL

        // sequential remote implementation
        template <typename Algo, typename ExPolicy, typename SegIter,
            typename T, typename Reduce, typename Convert>
        static typename detail::algorithm_result<ExPolicy, T>::type
        segmented_transform_reduce(Algo && algo, ExPolicy const& policy,
            SegIter first, SegIter last, T && init,
            Reduce && red_op, Convert && conv_op, boost::mpl::true_)
        {
            typedef hpx::traits::segmented_iterator_traits<SegIter> traits;
            typedef typename traits::segment_iterator segment_iterator;
            typedef typename traits::local_iterator local_iterator_type;
            typedef detail::algorithm_result<ExPolicy, T> result;

            using boost::mpl::true_;

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
                    overall_result =
                        util::remote::dispatch(traits::get_id(sit),
                            std::forward<Algo>(algo), policy, true_(),
                            beg, end, std::forward<T>(init),
                            std::forward<Reduce>(red_op),
                            std::forward<Convert>(conv_op));
                }
            }
            else {
                // handle the remaining part of the first partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::end(sit);
                if (beg != end)
                {
                    overall_result = red_op(
                        overall_result,
                        util::remote::dispatch(traits::get_id(sit),
                            std::forward<Algo>(algo), policy, true_(),
                            beg, end, std::forward<T>(init),
                            std::forward<Reduce>(red_op),
                            std::forward<Convert>(conv_op))
                    );
                }

                // handle all of the full partitions
                for (++sit; sit != send; ++sit)
                {
                    beg = traits::begin(sit);
                    end = traits::end(sit);
                    if (beg != end)
                    {
                        overall_result = red_op(
                            overall_result,
                            util::remote::dispatch(traits::get_id(sit),
                                std::forward<Algo>(algo), policy, true_(),
                                beg, end, std::forward<T>(init),
                                std::forward<Reduce>(red_op),
                                std::forward<Convert>(conv_op))
                        );
                    }
                }

                // handle the beginning of the last partition
                beg = traits::begin(sit);
                end = traits::local(last);
                if (beg != end)
                {
                    overall_result = red_op(
                        overall_result,
                        util::remote::dispatch(traits::get_id(sit),
                            std::forward<Algo>(algo), policy, true_(),
                            beg, end, std::forward<T>(init),
                            std::forward<Reduce>(red_op),
                            std::forward<Convert>(conv_op))
                    );
                }
            }

            return result::get(std::move(overall_result));
        }

        // parallel remote implementation
        template <typename Algo, typename ExPolicy, typename SegIter,
            typename T, typename Reduce, typename Convert>
        static typename detail::algorithm_result<ExPolicy, T>::type
        segmented_transform_reduce(Algo && algo, ExPolicy const& policy,
            SegIter first, SegIter last, T && init,
            Reduce && red_op, Convert && conv_op, boost::mpl::false_)
        {
            typedef hpx::traits::segmented_iterator_traits<SegIter> traits;
            typedef typename traits::segment_iterator segment_iterator;
            typedef typename traits::local_iterator local_iterator_type;
            typedef detail::algorithm_result<ExPolicy> result;

            typedef typename std::iterator_traits<SegIter>::iterator_category
                iterator_category;
            typedef typename boost::mpl::bool_<boost::is_same<
                    iterator_category, std::input_iterator_tag
                >::value> forced_seq;

            segment_iterator sit = traits::segment(first);
            segment_iterator send = traits::segment(last);

            std::vector<future<T> > segments;
            segments.reserve(std::distance(sit, send));

            if (sit == send)
            {
                // all elements are on the same partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::local(last);
                if (beg != end)
                {
                    segments.push_back(
                        util::remote::dispatch_async(traits::get_id(sit),
                            std::forward<Algo>(algo), policy, forced_seq(),
                            beg, end, std::forward<T>(init),
                            std::forward<Reduce>(red_op),
                            std::forward<Convert>(conv_op))
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
                        util::remote::dispatch_async(traits::get_id(sit),
                            std::forward<Algo>(algo), policy, forced_seq(),
                            beg, end, std::forward<T>(init),
                            std::forward<Reduce>(red_op),
                            std::forward<Convert>(conv_op))
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
                            util::remote::dispatch_async(traits::get_id(sit),
                                std::forward<Algo>(algo), policy, forced_seq(),
                                beg, end, std::forward<T>(init),
                                std::forward<Reduce>(red_op),
                                std::forward<Convert>(conv_op))
                        );
                    }
                }

                // handle the beginning of the last partition
                beg = traits::begin(sit);
                end = traits::local(last);
                if (beg != end)
                {
                    segments.push_back(
                        util::remote::dispatch_async(traits::get_id(sit),
                            std::forward<Algo>(algo), policy, forced_seq(),
                            beg, end, std::forward<T>(init),
                            std::forward<Reduce>(red_op),
                            std::forward<Convert>(conv_op))
                    );
                }
            }

            return result::get(
                dataflow(
                    hpx::util::unwrapped([=](std::vector<T> && r)
                    {
                        return std::accumulate(r.begin(), r.end(), init, red_op);
                    }),
                    segments));
        }

        ///////////////////////////////////////////////////////////////////////
        // segmented implementation
        template <typename ExPolicy, typename InIter, typename T, typename Reduce,
            typename Convert>
        typename detail::algorithm_result<ExPolicy, T>::type
        transform_reduce_(ExPolicy&& policy, InIter first, InIter last, T init,
            Reduce && red_op, Convert && conv_op, boost::mpl::true_)
        {
            typedef typename parallel::is_sequential_execution_policy<
                    ExPolicy
                >::type is_seq;

            if (first == last)
                return detail::algorithm_result<ExPolicy, T>::get(std::move(init));

            return segmented_transform_reduce(
                transform_reduce<T>(), std::forward<ExPolicy>(policy),
                first, last, std::move(init),
                std::forward<Reduce>(red_op), std::forward<Convert>(conv_op),
                is_seq());
        }

        // forward declare the non-segmented version of this algorithm
        template <typename ExPolicy, typename InIter, typename T, typename Reduce,
            typename Convert>
        typename detail::algorithm_result<ExPolicy, T>::type
        transform_reduce_(ExPolicy&& policy, InIter first, InIter last, T init,
            Reduce && red_op, Convert && conv_op, boost::mpl::false_);

        /// \endcond
    }
}}}

#endif
