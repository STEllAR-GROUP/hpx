//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/mismatch.hpp

#if !defined(HPX_PARALLEL_DETAIL_MISMATCH_JUL_13_2014_0142PM)
#define HPX_PARALLEL_DETAIL_MISMATCH_JUL_13_2014_0142PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/detail/algorithm_result.hpp>
#include <hpx/parallel/util/partitioner.hpp>
#include <hpx/parallel/util/loop.hpp>
#include <hpx/parallel/util/zip_iterator.hpp>

#include <algorithm>
#include <iterator>

#include <boost/static_assert.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_base_of.hpp>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
//     ///////////////////////////////////////////////////////////////////////////
//     // mismatch (binary)
//     namespace detail
//     {
//         /// \cond NOINTERNAL
//         template <typename InIter1, typename InIter2, typename F>
//         std::pair<InIter1, InIter1>
//         sequential_mismatch_binary(InIter1 first1, InIter1 last1,
//             InIter2 first2, InIter2 last2, F && f)
//         {
//             while (first1 != last1 && first2 != last2 && f(*first1, *first2))
//             {
//                 ++first1, ++first2;
//             }
//             return std::make_pair(first1, first2);
//         }
//
//         template <typename ExPolicy, typename InIter1, typename InIter2,
//             typename F>
//         typename detail::algorithm_result<ExPolicy, bool>::type
//         mismatch_binary(ExPolicy const&, InIter1 first1, InIter1 last1,
//             InIter2 first2, InIter2 last2, F && f, boost::mpl::true_)
//         {
//             try {
//                 return detail::algorithm_result<ExPolicy, bool>::get(
//                     sequential_mismatch_binary(first1, last1,
//                         first2, last2, std::forward<F>(f)));
//             }
//             catch (...) {
//                 detail::handle_exception<ExPolicy>::call();
//             }
//         }
//
//         template <typename ExPolicy, typename FwdIter1, typename FwdIter2, typename F>
//         typename detail::algorithm_result<ExPolicy, bool>::type
//         mismatch_binary(ExPolicy const& policy, FwdIter1 first1, FwdIter1 last1,
//             FwdIter2 first2, FwdIter2 last2, F && f, boost::mpl::false_)
//         {
//             if (first1 == last1)
//                 return detail::algorithm_result<ExPolicy, bool>::get(first2 == last2);
//
//             if (first2 == last2)
//                 return detail::algorithm_result<ExPolicy, bool>::get(false);
//
//             std::size_t count1 = std::distance(first1, last1);
//
//             // The specifcation of std::mismatch(_binary) states that if InIter1
//             // and InIter2 meet the requirements of RandomAccessIterator and
//             // last1 - first1 != last2 - first2 then no applications of the
//             // predicate p are made.
//             //
//             // We perform this check for any iterator type better than input
//             // iterators. This could turn into a QoI issue.
//             std::size_t count2 = std::distance(first2, last2);
//             if (count1 != count2)
//                 return detail::algorithm_result<ExPolicy, bool>::get(false);
//
//             typedef hpx::util::zip_iterator<FwdIter1, FwdIter2> zip_iterator;
//             typedef typename zip_iterator::reference reference;
//
//             util::cancellation_token<> tok;
//             return util::partitioner<ExPolicy, bool>::call(policy,
//                 hpx::util::make_zip_iterator(first1, first2), count1,
//                 [f, tok](zip_iterator it, std::size_t part_count) mutable
//                 {
//                     util::loop_n(
//                         it, part_count, tok,
//                         [&f, &tok](reference t)
//                         {
//                             if (!f(hpx::util::get<0>(t), hpx::util::get<1>(t)))
//                                 tok.cancel();
//                         });
//                     return !tok.was_cancelled();
//                 },
//                 [](std::vector<hpx::future<bool> > && results)
//                 {
//                     return std::all_of(
//                         boost::begin(results), boost::end(results),
//                         [](hpx::future<bool>& val)
//                         {
//                             return val.get();
//                         });
//                 });
//         }
//
//         template <typename InIter1, typename InIter2, typename F>
//         bool mismatch_binary(execution_policy const& policy, InIter1 first1,
//             InIter1 last1, InIter2 first2, InIter2 last2, F && f,
//             boost::mpl::false_)
//         {
//             HPX_PARALLEL_DISPATCH(policy, detail::mismatch_binary, first1, last1,
//                 first2, last2, std::forward<F>(f));
//         }
//
//         template <typename InIter1, typename InIter2, typename F>
//         bool mismatch_binary(execution_policy const& policy, InIter1 first1,
//             InIter1 last1, InIter2 first2, InIter2 last2, F && f,
//             boost::mpl::true_)
//         {
//             return detail::mismatch_binary(sequential_execution_policy(),
//                 first1, last1, first2, last2,
//                 std::forward<F>(f), boost::mpl::true_());
//         }
//         /// \endcond
//     }
//
//     /// Returns true if the range [first1, last1) is mismatch to the range
//     /// [first2, last2), and false otherwise.
//     ///
//     /// \note   Complexity: At most min(last1 - first1, last2 - first2)
//     ///         applications of the operator==().
//     ///
//     /// \tparam ExPolicy    The type of the execution policy to use (deduced).
//     ///                     It describes the manner in which the execution
//     ///                     of the algorithm may be parallelized and the manner
//     ///                     in which it executes the assignments.
//     /// \tparam InIter1     The type of the source iterators used for the
//     ///                     first range (deduced).
//     ///                     This iterator type must meet the requirements of an
//     ///                     input iterator.
//     /// \tparam InIter2     The type of the source iterators used for the
//     ///                     second range (deduced).
//     ///                     This iterator type must meet the requirements of an
//     ///                     input iterator.
//     ///
//     /// \param policy       The execution policy to use for the scheduling of
//     ///                     the iterations.
//     /// \param first1       Refers to the beginning of the sequence of elements
//     ///                     of the first range the algorithm will be applied to.
//     /// \param last1        Refers to the end of the sequence of elements of
//     ///                     the first range the algorithm will be applied to.
//     /// \param first2       Refers to the beginning of the sequence of elements
//     ///                     of the second range the algorithm will be applied to.
//     /// \param last2        Refers to the end of the sequence of elements of
//     ///                     the second range the algorithm will be applied to.
//     ///
//     /// The comparison operations in the parallel \a mismatch algorithm invoked
//     /// with an execution policy object of type \a sequential_execution_policy
//     /// execute in sequential order in the calling thread.
//     ///
//     /// The comparison operations in the parallel \a mismatch algorithm invoked
//     /// with an execution policy object of type \a parallel_execution_policy
//     /// or \a task_execution_policy are permitted to execute in an unordered
//     /// fashion in unspecified threads, and indeterminately sequenced
//     /// within each thread.
//     ///
//     /// \note     The two ranges are considered mismatch if, for every iterator
//     ///           i in the range [first1,last1), *i mismatchs *(first2 + (i - first1)).
//     ///           This overload of mismatch uses operator== to determine if two
//     ///           elements are mismatch.
//     ///
//     /// \returns  The \a mismatch algorithm returns a \a hpx::future<bool> if the
//     ///           execution policy is of type \a task_execution_policy and
//     ///           returns \a bool otherwise.
//     ///           The \a mismatch algorithm returns true if the elements in the
//     ///           two ranges are mismatch, otherwise it returns false.
//     ///           If the length of the range [first1, last1) does not mismatch
//     ///           the length of the range [first2, last2), it returns false.
//     ///
//     /// This overload of \a reduce is available only if the compiler
//     /// supports default function template arguments.
//     ///
//     /// The difference between \a reduce and \a accumulate is
//     /// that the behavior of reduce may be non-deterministic for
//     /// non-associative or non-commutative binary predicate.
//     ///
//     template <typename ExPolicy, typename InIter1, typename InIter2>
//     inline typename boost::enable_if<
//         is_execution_policy<ExPolicy>,
//         typename detail::algorithm_result<ExPolicy, bool>::type
//     >::type
//     mismatch(ExPolicy&& policy, InIter1 first1, InIter1 last1,
//         InIter2 first2, InIter2 last2)
//     {
//         typedef typename std::iterator_traits<InIter1>::iterator_category
//             iterator_category1;
//         typedef typename std::iterator_traits<InIter2>::iterator_category
//             iterator_category2;
//
//         BOOST_STATIC_ASSERT_MSG(
//             (boost::is_base_of<std::input_iterator_tag, iterator_category1>::value),
//             "Requires at least input iterator.");
//         BOOST_STATIC_ASSERT_MSG(
//             (boost::is_base_of<std::input_iterator_tag, iterator_category2>::value),
//             "Requires at least input iterator.");
//
//         typedef typename boost::mpl::or_<
//             is_sequential_execution_policy<ExPolicy>,
//             boost::is_same<std::input_iterator_tag, iterator_category1>,
//             boost::is_same<std::input_iterator_tag, iterator_category2>
//         >::type is_seq;
//
//         return detail::mismatch_binary(std::forward<ExPolicy>(policy), first1,
//             last1, first2, last2, detail::equal_to(), is_seq());
//     }
//
//     /// Returns true if the range [first1, last1) is mismatch to the range
//     /// [first2, last2), and false otherwise.
//     ///
//     /// \note   Complexity: At most min(last1 - first1, last2 - first2)
//     ///         applications of the predicate \a f.
//     ///
//     /// \tparam ExPolicy    The type of the execution policy to use (deduced).
//     ///                     It describes the manner in which the execution
//     ///                     of the algorithm may be parallelized and the manner
//     ///                     in which it executes the assignments.
//     /// \tparam InIter1     The type of the source iterators used for the
//     ///                     first range (deduced).
//     ///                     This iterator type must meet the requirements of an
//     ///                     input iterator.
//     /// \tparam InIter2     The type of the source iterators used for the
//     ///                     second range (deduced).
//     ///                     This iterator type must meet the requirements of an
//     ///                     input iterator.
//     /// \tparam F           The type of the function/function object to use
//     ///                     (deduced). Unlike its sequential form, the parallel
//     ///                     overload of \a mismatch requires \a F to meet the
//     ///                     requirements of \a CopyConstructible.
//     ///
//     /// \param policy       The execution policy to use for the scheduling of
//     ///                     the iterations.
//     /// \param first1       Refers to the beginning of the sequence of elements
//     ///                     of the first range the algorithm will be applied to.
//     /// \param last1        Refers to the end of the sequence of elements of
//     ///                     the first range the algorithm will be applied to.
//     /// \param first2       Refers to the beginning of the sequence of elements
//     ///                     of the second range the algorithm will be applied to.
//     /// \param last2        Refers to the end of the sequence of elements of
//     ///                     the second range the algorithm will be applied to.
//     /// \param f            The binary predicate which returns true if the
//     ///                     elements should be treated as mismatch. The signature
//     ///                     of the predicate function should be equivalent to
//     ///                     the following:
//     ///                     \code
//     ///                     bool pred(const Type1 &a, const Type2 &b);
//     ///                     \endcode \n
//     ///                     The signature does not need to have const &, but
//     ///                     the function must not modify the objects passed to
//     ///                     it. The types \a Type1 and \a Type2 must be such
//     ///                     that objects of types \a InIter1 and \a InIter2 can
//     ///                     be dereferenced and then implicitly converted to
//     ///                     \a Type1 and \a Type2 respectively
//     ///
//     /// The comparison operations in the parallel \a mismatch algorithm invoked
//     /// with an execution policy object of type \a sequential_execution_policy
//     /// execute in sequential order in the calling thread.
//     ///
//     /// The comparison operations in the parallel \a mismatch algorithm invoked
//     /// with an execution policy object of type \a parallel_execution_policy
//     /// or \a task_execution_policy are permitted to execute in an unordered
//     /// fashion in unspecified threads, and indeterminately sequenced
//     /// within each thread.
//     ///
//     /// \note     The two ranges are considered mismatch if, for every iterator
//     ///           i in the range [first1,last1), *i mismatchs *(first2 + (i - first1)).
//     ///           This overload of mismatch uses operator== to determine if two
//     ///           elements are mismatch.
//     ///
//     /// \returns  The \a mismatch algorithm returns a \a hpx::future<bool> if the
//     ///           execution policy is of type \a task_execution_policy and
//     ///           returns \a bool otherwise.
//     ///           The \a mismatch algorithm returns true if the elements in the
//     ///           two ranges are mismatch, otherwise it returns false.
//     ///           If the length of the range [first1, last1) does not mismatch
//     ///           the length of the range [first2, last2), it returns false.
//     ///
//     /// This overload of \a reduce is available only if the compiler
//     /// supports default function template arguments.
//     ///
//     /// The difference between \a reduce and \a accumulate is
//     /// that the behavior of reduce may be non-deterministic for
//     /// non-associative or non-commutative binary predicate.
//     ///
//     template <typename ExPolicy, typename InIter1, typename InIter2, typename F>
//     inline typename boost::enable_if<
//         is_execution_policy<ExPolicy>,
//         typename detail::algorithm_result<ExPolicy, bool>::type
//     >::type
//     mismatch(ExPolicy&& policy, InIter1 first1, InIter1 last1,
//         InIter2 first2, InIter2 last2, F && f)
//     {
//         typedef typename std::iterator_traits<InIter1>::iterator_category
//             iterator_category1;
//         typedef typename std::iterator_traits<InIter2>::iterator_category
//             iterator_category2;
//
//         BOOST_STATIC_ASSERT_MSG(
//             (boost::is_base_of<std::input_iterator_tag, iterator_category1>::value),
//             "Requires at least input iterator.");
//         BOOST_STATIC_ASSERT_MSG(
//             (boost::is_base_of<std::input_iterator_tag, iterator_category2>::value),
//             "Requires at least input iterator.");
//
//         typedef typename boost::mpl::or_<
//             is_sequential_execution_policy<ExPolicy>,
//             boost::is_same<std::input_iterator_tag, iterator_category1>,
//             boost::is_same<std::input_iterator_tag, iterator_category2>
//         >::type is_seq;
//
//         return detail::mismatch_binary(std::forward<ExPolicy>(policy), first1,
//             last1, first2, last2, std::forward<F>(f), is_seq());
//     }

    ///////////////////////////////////////////////////////////////////////////
    // mismatch
    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename ExPolicy, typename InIter1, typename InIter2,
            typename F>
        typename detail::algorithm_result<ExPolicy, bool>::type
        mismatch(ExPolicy const&, InIter1 first1, InIter1 last1,
            InIter2 first2, F && f, boost::mpl::true_)
        {
            try {
                typedef std::pair<InIter1, InIter2> return_type;
                return detail::algorithm_result<ExPolicy, return_type>::get(
                    std::mismatch(first1, last1, first2, std::forward<F>(f)));
            }
            catch (...) {
                detail::handle_exception<ExPolicy>::call();
            }
        }

        template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
            typename F>
        typename detail::algorithm_result<
            ExPolicy, std::pair<FwdIter1, FwdIter2>
        >::type
        mismatch(ExPolicy const& policy, FwdIter1 first1, FwdIter1 last1,
            FwdIter2 first2, F && f, boost::mpl::false_)
        {
            if (first1 == last1)
                return detail::algorithm_result<ExPolicy, bool>::get(true);

            std::size_t count = std::distance(first1, last1);

            typedef hpx::util::zip_iterator<FwdIter1, FwdIter2> zip_iterator;
            typedef typename zip_iterator::reference reference;

            util::cancellation_token<std::size_t> tok(count);

            util::partitioner<ExPolicy>::call(policy,
                hpx::util::make_zip_iterator(first1, first2), count,
                [f, tok](zip_iterator it, std::size_t part_count) mutable
                {
                    std::size_t base_idx =
                        std::distance(first1, hpx::util::get<0>(*it));

                    util::loop_idx_n(
                        base_idx, it, part_count, tok,
                        [&f, &tok](reference t, std::size_t i)
                        {
                            if (!f(hpx::util::get<0>(t), hpx::util::get<1>(t)))
                                tok.cancel(i);
                        });
                });

                std::size_t mismatched = tok.get_data();
                if (mismatched != count)
                    std::advance(first1, mismatched);
                else
                    first1 = last1;

                std::advance(first2, mismatched);
                return std::make_pair(first1, first2);
        }

        template <typename InIter1, typename InIter2, typename F>
        std::pair<InIter1, InIter2> mismatch(execution_policy const& policy,
            InIter1 first1, InIter1 last1, InIter2 first2, F && f,
            boost::mpl::false_)
        {
            HPX_PARALLEL_DISPATCH(policy, detail::mismatch, first1, last1,
                first2, std::forward<F>(f));
        }

        template <typename InIter1, typename InIter2, typename F>
        std::pair<InIter1, InIter2> mismatch(execution_policy const& policy,
            InIter1 first1, InIter1 last1, InIter2 first2, F && f,
            boost::mpl::true_)
        {
            return detail::mismatch(sequential_execution_policy(),
                first1, last1, first2, std::forward<F>(f), boost::mpl::true_());
        }
        /// \endcond
    }

    /// Returns std::pair with iterators to the first two non-equivalent
    /// elements.
    ///
    /// \note   Complexity: At most \a last1 - \a first1 applications of the
    ///         operator==().
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam InIter1     The type of the source iterators used for the
    ///                     first range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
    /// \tparam InIter2     The type of the source iterators used for the
    ///                     second range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first1       Refers to the beginning of the sequence of elements
    ///                     of the first range the algorithm will be applied to.
    /// \param last1        Refers to the end of the sequence of elements of
    ///                     the first range the algorithm will be applied to.
    /// \param first2       Refers to the beginning of the sequence of elements
    ///                     of the second range the algorithm will be applied to.
    ///
    /// The comparison operations in the parallel \a mismatch algorithm invoked
    /// with an execution policy object of type \a sequential_execution_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The comparison operations in the parallel \a mismatch algorithm invoked
    /// with an execution policy object of type \a parallel_execution_policy
    /// or \a task_execution_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a mismatch algorithm returns a
    ///           \a hpx::future<std::pair<InIter1, InIter2> > if the
    ///           execution policy is of type \a task_execution_policy and
    ///           returns \a std::pair<InIter1, InIter2> otherwise.
    ///           The \a mismatch algorithm returns the first mismatching pair
    ///           of elements from two ranges: one defined by [first1, last1)
    ///           and another defined by [first2, last2).
    ///
    template <typename ExPolicy, typename InIter1, typename InIter2>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename detail::algorithm_result<
            ExPolicy, std::pair<InIter1, InIter2>
        >::type
    >::type
    mismatch(ExPolicy&& policy, InIter1 first1, InIter1 last1,
        InIter2 first2)
    {
        typedef typename std::iterator_traits<InIter1>::iterator_category
            iterator_category1;
        typedef typename std::iterator_traits<InIter2>::iterator_category
            iterator_category2;

        BOOST_STATIC_ASSERT_MSG(
            (boost::is_base_of<std::input_iterator_tag, iterator_category1>::value),
            "Requires at least input iterator.");
        BOOST_STATIC_ASSERT_MSG(
            (boost::is_base_of<std::input_iterator_tag, iterator_category2>::value),
            "Requires at least input iterator.");

        typedef typename boost::mpl::or_<
            is_sequential_execution_policy<ExPolicy>,
            boost::is_same<std::input_iterator_tag, iterator_category1>,
            boost::is_same<std::input_iterator_tag, iterator_category2>
        >::type is_seq;

        return detail::mismatch(std::forward<ExPolicy>(policy), first1, last1,
            first2, detail::equal_to(), is_seq());
    }

    /// Returns std::pair with iterators to the first two non-equivalent
    /// elements.
    ///
    /// \note   Complexity: At most \a last1 - \a first1 applications of the
    ///         predicate \a f.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam InIter1     The type of the source iterators used for the
    ///                     first range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
    /// \tparam InIter2     The type of the source iterators used for the
    ///                     second range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a mismatch requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first1       Refers to the beginning of the sequence of elements
    ///                     of the first range the algorithm will be applied to.
    /// \param last1        Refers to the end of the sequence of elements of
    ///                     the first range the algorithm will be applied to.
    /// \param first2       Refers to the beginning of the sequence of elements
    ///                     of the second range the algorithm will be applied to.
    /// \param f            The binary predicate which returns true if the
    ///                     elements should be treated as mismatch. The signature
    ///                     of the predicate function should be equivalent to
    ///                     the following:
    ///                     \code
    ///                     bool pred(const Type1 &a, const Type2 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The types \a Type1 and \a Type2 must be such
    ///                     that objects of types \a InIter1 and \a InIter2 can
    ///                     be dereferenced and then implicitly converted to
    ///                     \a Type1 and \a Type2 respectively
    ///
    /// The comparison operations in the parallel \a mismatch algorithm invoked
    /// with an execution policy object of type \a sequential_execution_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The comparison operations in the parallel \a mismatch algorithm invoked
    /// with an execution policy object of type \a parallel_execution_policy
    /// or \a task_execution_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a mismatch algorithm returns a
    ///           \a hpx::future<std::pair<InIter1, InIter2> > if the
    ///           execution policy is of type \a task_execution_policy and
    ///           returns \a std::pair<InIter1, InIter2> otherwise.
    ///           The \a mismatch algorithm returns the first mismatching pair
    ///           of elements from two ranges: one defined by [first1, last1)
    ///           and another defined by [first2, last2).
    ///
    template <typename ExPolicy, typename InIter1, typename InIter2, typename F>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename detail::algorithm_result<
            ExPolicy, std::pair<InIter1, InIter2>
        >::type
    >::type
    mismatch(ExPolicy&& policy, InIter1 first1, InIter1 last1, InIter2 first2,
        F && f)
    {
        typedef typename std::iterator_traits<InIter1>::iterator_category
            iterator_category1;
        typedef typename std::iterator_traits<InIter2>::iterator_category
            iterator_category2;

        BOOST_STATIC_ASSERT_MSG(
            (boost::is_base_of<std::input_iterator_tag, iterator_category1>::value),
            "Requires at least input iterator.");
        BOOST_STATIC_ASSERT_MSG(
            (boost::is_base_of<std::input_iterator_tag, iterator_category2>::value),
            "Requires at least input iterator.");

        typedef typename boost::mpl::or_<
            is_sequential_execution_policy<ExPolicy>,
            boost::is_same<std::input_iterator_tag, iterator_category1>,
            boost::is_same<std::input_iterator_tag, iterator_category2>
        >::type is_seq;

        return detail::mismatch(std::forward<ExPolicy>(policy), first1, last1,
            first2, std::forward<F>(f), is_seq());
    }
}}}

#endif
