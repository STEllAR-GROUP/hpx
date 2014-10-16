//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_SEGMENTED_ALGORITHM_FOR_EACH_OCT_15_2014_0839PM)
#define HPX_PARALLEL_SEGMENTED_ALGORITHM_FOR_EACH_OCT_15_2014_0839PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/void_guard.hpp>
#include <hpx/util/move.hpp>

#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/algorithm_result.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/detail/is_negative.hpp>
#include <hpx/parallel/util/remote/loop.hpp>

#include <algorithm>
#include <iterator>

#include <boost/static_assert.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_same.hpp>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
    ///////////////////////////////////////////////////////////////////////////
    // for_each_n_segmented
    namespace detail
    {
        /// \cond NOINTERNAL
        struct for_each_segmented
          : public detail::algorithm<for_each_segmented>
        {
            for_each_segmented()
              : for_each_segmented::algorithm("for_each_segmented")
            {}

            template <typename F, typename Iter>
            static void invoke(F const& f, Iter it)
            {
                f(*it);
            }

            template <typename ExPolicy, typename SegIter, typename F>
            static void
            sequential(ExPolicy const&, SegIter first, SegIter last, F && f)
            {
                using hpx::util:placeholders::_1;
                using traits::local;
                using traits::begin;
                using traits::end;

                typedef segmented_iterator_traits<SegIter> traits;
                typedef typename traits::segment_iterator segment_iterator;
                typedef typename traits::local_iterator local_iterator;

                segment_iterator sfirst = traits::segment(first);
                segment_iterator slast = traits::segment(last);

                void (invoke_)(F const&, local_iterator) =
                    &for_each_segmented::invoke;
                auto f_ = hpx::util::bind(invoke_, std::forward<F>(f), _1);

                id_type id = sfirst.get_id();

                if (sfirst == slast)
                    util::remote::loop(id, local(first), local(last), f_);
                else
                {
                    util::remote::loop(id, local(first), end(sfirst), f_);
                    for (++sfirst; sfirst != slast; ++sfirst)
                        util::remote::loop(id, begin(sfirst), end(sfirst), f_);
                    util::remote::loop(id, begin(sfirst), local(last), f_);
                }
                return hpx::util::unused;
            }

            template <typename ExPolicy, typename F>
            static typename detail::algorithm_result<ExPolicy>::type
            parallel(ExPolicy const& policy, SegIter first, SegIter last,
                F && f)
            {
//                 if (count != 0)
//                 {
//                     return util::foreach_n_partitioner<ExPolicy, Iter>::call(
//                         policy, first, count,
//                         [f](Iter part_begin, std::size_t part_size)
//                         {
//                             util::loop_n(part_begin, part_size,
//                                 [&f](Iter const& curr)
//                                 {
//                                     f(*curr);
//                                 });
//                         });
//                 }

                return detail::algorithm_result<ExPolicy>::get();
            }
        };

        ///////////////////////////////////////////////////////////////////////
        // segmented implementation
        template <typename ExPolicy, typename SegIter, typename F>
        inline typename boost::enable_if<
            is_execution_policy<ExPolicy>,
            typename detail::algorithm_result<ExPolicy, void>::type
        >::type
        for_each(ExPolicy && policy, SegIter first, SegIter last, F && f,
            boost::mpl::true_)
        {
            typedef typename std::iterator_traits<SegIter>::iterator_category
                iterator_category;

            typedef typename boost::mpl::or_<
                is_sequential_execution_policy<ExPolicy>,
                boost::is_same<std::input_iterator_tag, iterator_category>
            >::type is_seq;

            return for_each_segmented().call(
                std::forward<ExPolicy>(policy),
                first, last, std::forward<F>(f), is_seq());
        }
        /// \endcond
    }
}}}

#endif
