//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_SEGMENTED_ALGORITHM_FOR_EACH_OCT_15_2014_0839PM)
#define HPX_PARALLEL_SEGMENTED_ALGORITHM_FOR_EACH_OCT_15_2014_0839PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/void_guard.hpp>
#include <hpx/util/move.hpp>
#include <hpx/traits/segemented_iterator_traits.hpp>

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
#include <boost/serialization/serialization.hpp>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
    ///////////////////////////////////////////////////////////////////////////
    // for_each_n_segmented
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        /// \cond NOINTERNAL
//         template <typename F>
//         struct remotable_invoke
//         {
//             remotable_invoke() {}
//
//             template <typename F_>
//             explicit remotable_invoke(F_ && f)
//               : f_(std::forward<F_>(f))
//             {}
//
//             template <typename Iter>
//             void operator()(Iter it) const
//             {
//                 f_(*it);
//             }
//
//             operator F& () { return f_; }
//             operator F const& () const { return f_; }
//
//         private:
//             friend class boost::serialization::access;
//
//             F f_;
//
//             template <typename Archive>
//             void serialize(Archive& ar, unsigned int)
//             {
//                 ar & f_;
//             }
//         };
//
//         ///////////////////////////////////////////////////////////////////////
//         struct for_each_segmented
//           : public detail::algorithm<for_each_segmented>
//         {
//             for_each_segmented()
//               : for_each_segmented::algorithm("for_each_segmented")
//             {}
//
//             template <typename ExPolicy, typename SegIter, typename F>
//             static hpx::util::unused_type
//             sequential(ExPolicy const&, SegIter first, SegIter last, F && f)
//             {
//                 typedef hpx::traits::segmented_iterator_traits<SegIter> traits;
//                 typedef typename traits::segment_iterator segment_iterator;
//                 typedef typename traits::local_iterator::base_iterator_type
//                     local_base_iterator_type;
//                 typedef typename hpx::util::decay<F>::type function_type;
//
//                 segment_iterator sit = traits::segment(first);
//                 segment_iterator send = traits::segment(last);
//                 remotable_invoke<function_type> f_(std::forward<F>(f));
//
//                 using util::remote::segmented_sequential_loop;
//                 if (sit == send)
//                 {
//                     // all elements are on the same partition
//                     segmented_sequential_loop(sit, traits::local(first),
//                         traits::local(last), f_);
//                 }
//                 else
//                 {
//                     // handle the remaining part of the first partition
//                     segmented_sequential_loop(sit, traits::local(first),
//                         traits::end(sit), f_);
//
//                     // handle all of the full partitions
//                     for (++sit; sit != send; ++sit)
//                     {
//                         segmented_sequential_loop(sit, traits::begin(sit),
//                             traits::end(sit), f_);
//                     }
//
//                     // handle the beginning of the last partition
//                     segmented_sequential_loop(sit, traits::begin(sit),
//                         traits::local(last), f_);
//                 }
//
//                 return hpx::util::unused;
//             }
//
//             template <typename ExPolicy, typename SegIter, typename F>
//             static typename detail::algorithm_result<ExPolicy>::type
//             parallel(ExPolicy const& policy, SegIter first, SegIter last,
//                 F && f)
//             {
// //                 if (count != 0)
// //                 {
// //                     return util::foreach_n_partitioner<ExPolicy, Iter>::call(
// //                         policy, first, count,
// //                         [f](Iter part_begin, std::size_t part_size)
// //                         {
// //                             util::loop_n(part_begin, part_size,
// //                                 [&f](Iter const& curr)
// //                                 {
// //                                     f(*curr);
// //                                 });
// //                         });
// //                 }
//
//                 return detail::algorithm_result<ExPolicy>::get();
//             }
//         };

        struct for_each_segmented : public detail::algorithm<for_each_segmented>
        {
            for_each_segmented()
              : for_each_segmented::algorithm("for_each_segmented")
            {}

            template <typename ExPolicy, typename InIter1, typename InIter2, typename F>
            static hpx::util::unused_type
            sequential(ExPolicy const&, InIter1 const& first,
                InIter2 const& last, F const& f)
            {
                std::for_each(first, last, f);
                return hpx::util::unused;
            }

            template <typename ExPolicy, typename FwdIter, typename F>
            static typename detail::algorithm_result<ExPolicy>::type
            parallel(ExPolicy const& policy, FwdIter first, FwdIter last, F && f)
            {
                typedef
                    typename detail::algorithm_result<ExPolicy>::type
                result_type;

                return hpx::util::void_guard<result_type>(),
                    detail::for_each_n<FwdIter>().call(
                        policy, first, std::distance(first, last),
                        std::forward<F>(f), boost::mpl::false_());
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Algo, typename ExPolicy, typename Arg0, typename Arg1, typename Arg2>
        typename parallel::v1::detail::algorithm_result<
            ExPolicy, typename Algo::result_type
        >::type
        call_sequential(Algo && algo, ExPolicy const& policy,
            Arg0 && arg0, Arg1 && arg1, Arg2 && arg2)
        {
            return algo.call(policy, std::forward<Arg0>(arg0),
                std::forward<Arg1>(arg1), std::forward<Arg2>(arg2),
                boost::mpl::true_());
        }

        template <typename Algo, typename ExPolicy, typename Arg0, typename Arg1, typename Arg2>
        typename parallel::v1::detail::algorithm_result<
            ExPolicy, typename Algo::result_type
        >::type
        call_parallel(Algo && algo, ExPolicy const& policy,
            Arg0 && arg0, Arg1 && arg1, Arg2 && arg2)
        {
            return algo.call(policy, std::forward<Arg0>(arg0),
                std::forward<Arg1>(arg1), std::forward<Arg2>(arg2),
                boost::mpl::false_());
        }

        ///////////////////////////////////////////////////////////////////////
        template <
            typename Algo, typename ExPolicy, typename IsSeq,
            typename Arg0, typename Arg1, typename Arg2,
            typename R =
                typename parallel::v1::detail::algorithm_result<
                    ExPolicy, typename Algo::result_type
                >::type>
        struct algorithm_invoker_action2;

        // sequential
        template <
            typename Algo, typename ExPolicy,
            typename Arg0, typename Arg1, typename Arg2,
            typename R>
        struct algorithm_invoker_action2<Algo, ExPolicy, boost::mpl::true_, Arg0, Arg1, Arg2, R>
          : hpx::actions::make_action<
                R (*)(Algo &&, ExPolicy const&, Arg0 &&, Arg1 &&, Arg2 &&),
                &call_sequential<Algo, ExPolicy, Arg0, Arg1, Arg2>,
                algorithm_invoker_action2<Algo, ExPolicy, boost::mpl::true_, Arg0, Arg1, Arg2, R>
            >
        {};

        // parallel
        template <
            typename Algo, typename ExPolicy,
            typename Arg0, typename Arg1, typename Arg2,
            typename R>
        struct algorithm_invoker_action2<Algo, ExPolicy, boost::mpl::false_, Arg0, Arg1, Arg2, R>
          : hpx::actions::make_action<
                R (*)(Algo &&, ExPolicy const&, Arg0 &&, Arg1 &&, Arg2 &&),
                &call_parallel<Algo, ExPolicy, Arg0, Arg1, Arg2>,
                algorithm_invoker_action2<Algo, ExPolicy, boost::mpl::false_, Arg0, Arg1, Arg2, R>
            >
        {};

        ///////////////////////////////////////////////////////////////////////
        template <typename Algo, typename ExPolicy,
            typename Arg0, typename Arg1, typename Arg2,
            typename IsSeq>
        BOOST_FORCEINLINE
        hpx::future<typename detail::algorithm_result<ExPolicy>::type>
        remote_call_async(id_type const& id, Algo && algo,
            ExPolicy const& policy,
            Arg0 && arg0, Arg1 && arg1, Arg2 && arg2,
            IsSeq)
        {
            typename algorithm_invoker_action2<
                    typename hpx::util::decay<Algo>::type,
                    ExPolicy, IsSeq,
                    typename hpx::util::decay<Arg0>::type,
                    typename hpx::util::decay<Arg1>::type,
                    typename hpx::util::decay<Arg2>::type
                > act;

            return hpx::async_colocated(act, id, algo, policy,
                std::forward<Arg0>(arg0), std::forward<Arg1>(arg1),
                std::forward<Arg2>(arg2));
        }

        template <typename Algo, typename ExPolicy,
            typename Arg0, typename Arg1, typename Arg2,
            typename IsSeq>
        BOOST_FORCEINLINE
        typename detail::algorithm_result<ExPolicy>::type
        remote_call(id_type const& id, Algo && algo, ExPolicy const& policy,
            Arg0 && arg0, Arg1 && arg1, Arg2 && arg2,
            IsSeq is_seq)
        {
            return remote_call_async(id, std::forward<Algo>(algo), policy,
                std::forward<Arg0>(arg0), std::forward<Arg1>(arg1),
                std::forward<Arg2>(arg2), is_seq).get();
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename Algo, typename ExPolicy, typename SegIter, typename F>
        static typename detail::algorithm_result<ExPolicy>::type
        segmented_for_each(Algo && algo, ExPolicy const& policy,
            SegIter first, SegIter last, F && f)
        {
            typedef hpx::traits::segmented_iterator_traits<SegIter> traits;
            typedef typename traits::segment_iterator segment_iterator;
            typedef typename traits::local_iterator local_iterator_type;
            typedef typename local_iterator_type::base_iterator_type
                local_base_iterator_type;
            typedef detail::algorithm_result<ExPolicy> result;

            segment_iterator sit = traits::segment(first);
            segment_iterator send = traits::segment(last);

            if (sit == send)
            {
                // all elements are on the same partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::local(last);
                if (beg == end)
                    return result::get();

                return remote_call(sit->get_id(),
                    std::forward<Algo>(algo), policy, beg, end,
                    std::forward<F>(f), boost::mpl::true_());
            }

            // handle the remaining part of the first partition
            local_iterator_type beg = traits::local(first);
            local_iterator_type end = traits::end(sit);
            if (beg != end)
            {
                remote_call(sit->get_id(),
                    std::forward<Algo>(algo), policy, beg, end,
                    std::forward<F>(f), boost::mpl::true_());
            }

            // handle all of the full partitions
            for (++sit; sit != send; ++sit)
            {
                beg = traits::begin(sit);
                end = traits::end(sit);
                if (beg != end)
                {
                    remote_call(sit->get_id(),
                        std::forward<Algo>(algo), policy, beg, end,
                        std::forward<F>(f), boost::mpl::true_());
                }
            }

            // handle the beginning of the last partition
            beg = traits::begin(sit);
            end = traits::local(last);
            if (beg != end)
            {
                remote_call(sit->get_id(),
                    std::forward<Algo>(algo), policy, beg, end,
                    std::forward<F>(f), boost::mpl::true_());
            }

            return result::get();
        }

        ///////////////////////////////////////////////////////////////////////
        // segmented implementation
        template <typename ExPolicy, typename SegIter, typename F>
        inline typename detail::algorithm_result<ExPolicy>::type
        for_each_(ExPolicy && policy, SegIter first, SegIter last, F && f,
            boost::mpl::true_)
        {
            typedef typename std::iterator_traits<SegIter>::iterator_category
                iterator_category;

            typedef typename boost::mpl::or_<
                parallel::is_sequential_execution_policy<ExPolicy>,
                boost::is_same<std::input_iterator_tag, iterator_category>
            >::type is_seq;

            if (first == last)
                return;

            return segmented_for_each(
                for_each_segmented(), std::forward<ExPolicy>(policy),
                first, last, std::forward<F>(f));
        }
        /// \endcond
    }
}}}

HPX_REGISTER_PLAIN_ACTION_TEMPLATE(
    (template <typename Algo, typename ExPolicy, typename IsSeq, typename Arg0, typename Arg1, typename Arg2, typename R>),
    (hpx::parallel::v1::detail::algorithm_invoker_action2<Algo, ExPolicy, IsSeq, Arg0, Arg1, Arg2, R>))

#endif
