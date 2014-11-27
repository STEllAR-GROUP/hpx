//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_SEGMENTED_ALGORITHM_FOR_EACH_OCT_15_2014_0839PM)
#define HPX_PARALLEL_SEGMENTED_ALGORITHM_FOR_EACH_OCT_15_2014_0839PM

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
#include <hpx/parallel/algorithms/for_each.hpp>

#include <algorithm>
#include <iterator>

#include <boost/type_traits/is_same.hpp>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
    ///////////////////////////////////////////////////////////////////////////
    // for_each_n_segmented
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        /// \cond NOINTERNAL

        // sequential remote implementation
        template <typename Algo, typename ExPolicy, typename SegIter, typename F>
        static typename detail::algorithm_result<ExPolicy>::type
        segmented_for_each(Algo && algo, ExPolicy const& policy,
            SegIter first, SegIter last, F && f, boost::mpl::true_)
        {
            typedef hpx::traits::segmented_iterator_traits<SegIter> traits;
            typedef typename traits::segment_iterator segment_iterator;
            typedef typename traits::local_iterator local_iterator_type;
            typedef typename local_iterator_type::base_iterator_type
                local_base_iterator_type;
            typedef detail::algorithm_result<ExPolicy> result;

            using boost::mpl::true_;

            segment_iterator sit = traits::segment(first);
            segment_iterator send = traits::segment(last);

            if (sit == send)
            {
                // all elements are on the same partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::local(last);
                if (beg != end)
                {
                    util::remote::dispatch(sit->get_id(),
                        std::forward<Algo>(algo), policy, true_(),
                        beg, end, std::forward<F>(f));
                }
            }
            else {
                // handle the remaining part of the first partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::end(sit);
                if (beg != end)
                {
                    util::remote::dispatch(sit->get_id(),
                        std::forward<Algo>(algo), policy, true_(),
                        beg, end, std::forward<F>(f));
                }

                // handle all of the full partitions
                for (++sit; sit != send; ++sit)
                {
                    beg = traits::begin(sit);
                    end = traits::end(sit);
                    if (beg != end)
                    {
                        util::remote::dispatch(sit->get_id(),
                            std::forward<Algo>(algo), policy, true_(),
                            beg, end, std::forward<F>(f));
                    }
                }

                // handle the beginning of the last partition
                beg = traits::begin(sit);
                end = traits::local(last);
                if (beg != end)
                {
                    util::remote::dispatch(sit->get_id(),
                        std::forward<Algo>(algo), policy, true_(),
                        beg, end, std::forward<F>(f));
                }
            }

            return result::get();
        }

        // parallel remote implementation
        template <typename Algo, typename ExPolicy, typename SegIter, typename F>
        static typename detail::algorithm_result<ExPolicy>::type
        segmented_for_each(Algo && algo, ExPolicy const& policy,
            SegIter first, SegIter last, F && f, boost::mpl::false_)
        {
            typedef hpx::traits::segmented_iterator_traits<SegIter> traits;
            typedef typename traits::segment_iterator segment_iterator;
            typedef typename traits::local_iterator local_iterator_type;
            typedef typename local_iterator_type::base_iterator_type
                local_base_iterator_type;
            typedef detail::algorithm_result<ExPolicy> result;

            typedef typename std::iterator_traits<SegIter>::iterator_category
                iterator_category;
            typedef typename boost::mpl::bool_<boost::is_same<
                    iterator_category, std::input_iterator_tag
                >::value> forced_seq;

            segment_iterator sit = traits::segment(first);
            segment_iterator send = traits::segment(last);

            std::vector<future<void> > segments;
            segments.reserve(std::distance(sit, send));

            if (sit == send)
            {
                // all elements are on the same partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::local(last);
                if (beg != end)
                {
                    segments.push_back(
                        util::remote::dispatch_async(sit->get_id(),
                            std::forward<Algo>(algo), policy, forced_seq(),
                            beg, end, std::forward<F>(f))
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
                        util::remote::dispatch_async(sit->get_id(),
                            std::forward<Algo>(algo), policy, forced_seq(),
                            beg, end, std::forward<F>(f))
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
                            util::remote::dispatch_async(sit->get_id(),
                                std::forward<Algo>(algo), policy, forced_seq(),
                                beg, end, std::forward<F>(f))
                        );
                    }
                }

                // handle the beginning of the last partition
                beg = traits::begin(sit);
                end = traits::local(last);
                if (beg != end)
                {
                    segments.push_back(
                        util::remote::dispatch_async(sit->get_id(),
                            std::forward<Algo>(algo), policy, forced_seq(),
                            beg, end, std::forward<F>(f))
                    );
                }
            }

            return result::get(future<void>(when_all(segments)));
        }

        ///////////////////////////////////////////////////////////////////////
        // segmented implementation
        template <typename ExPolicy, typename SegIter, typename F>
        inline typename detail::algorithm_result<ExPolicy>::type
        for_each_(ExPolicy && policy, SegIter first, SegIter last, F && f,
            boost::mpl::true_)
        {
            typedef typename parallel::is_sequential_execution_policy<
                    ExPolicy
                >::type is_seq;

            if (first == last)
                return detail::algorithm_result<ExPolicy>::get();

            return segmented_for_each(
                for_each(), std::forward<ExPolicy>(policy),
                first, last, std::forward<F>(f), is_seq());
        }

        // forward declare the non-segmented version of this algorithm
        template <typename ExPolicy, typename InIter, typename F>
        inline typename detail::algorithm_result<ExPolicy>::type
        for_each_(ExPolicy && policy, InIter first, InIter last, F && f,
            boost::mpl::false_);

        /// \endcond
    }
}}}

#endif
