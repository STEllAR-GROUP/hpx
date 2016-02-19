//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/set_symmetric_difference.hpp

#if !defined(HPX_PARALLEL_ALGORITHM_SET_SYMMETRIC_DIFFERENCE_MAR_10_2015_0204PM)
#define HPX_PARALLEL_ALGORITHM_SET_SYMMETRIC_DIFFERENCE_MAR_10_2015_0204PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/traits/segmented_iterator_traits.hpp>
#include <hpx/util/move.hpp>
#include <hpx/util/decay.hpp>

#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/detail/is_negative.hpp>
#include <hpx/parallel/algorithms/detail/set_operation.hpp>
#include <hpx/parallel/algorithms/copy.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/loop.hpp>

#include <algorithm>
#include <iterator>

#include <boost/shared_array.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/type_traits/is_base_of.hpp>
#include <boost/mpl/or.hpp>
#include <boost/mpl/not.hpp>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
    ///////////////////////////////////////////////////////////////////////////
    // set_symmetric_difference
    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename OutIter>
        struct set_symmetric_difference
          : public detail::algorithm<set_symmetric_difference<OutIter>, OutIter>
        {
            set_symmetric_difference()
              : set_symmetric_difference::algorithm("set_symmetric_difference")
            {}

            template <typename ExPolicy, typename InIter1, typename InIter2,
                typename F>
            static OutIter
            sequential(ExPolicy, InIter1 first1, InIter1 last1,
                InIter2 first2, InIter2 last2, OutIter dest, F && f)
            {
                return std::set_symmetric_difference(first1, last1,
                    first2, last2, dest, std::forward<F>(f));
            }

            template <typename ExPolicy, typename RanIter1, typename RanIter2,
                typename F>
            static typename util::detail::algorithm_result<
                ExPolicy, OutIter
            >::type
            parallel(ExPolicy policy, RanIter1 first1, RanIter1 last1,
                RanIter2 first2, RanIter2 last2, OutIter dest, F && f)
            {
                typedef typename std::iterator_traits<RanIter1>::difference_type
                    difference_type1;
                typedef typename std::iterator_traits<RanIter2>::difference_type
                    difference_type2;

                if (first1 == last1)
                {
                    return util::detail::convert_to_result(
                        detail::copy<std::pair<RanIter2, OutIter> >()
                            .call(
                                policy, boost::mpl::false_(), first2, last2,
                                dest
                            ),
                            [](std::pair<RanIter2, OutIter> const& p) -> OutIter
                            {
                                return p.second;
                            });
                }

                if (first2 == last2)
                {
                    return util::detail::convert_to_result(
                        detail::copy<std::pair<RanIter1, OutIter> >()
                            .call(
                                policy, boost::mpl::false_(), first1, last1,
                                dest
                            ),
                            [](std::pair<RanIter1, OutIter> const& p) -> OutIter
                            {
                                return p.second;
                            });
                }

                typedef typename set_operations_buffer<OutIter>::type buffer_type;
                typedef typename hpx::util::decay<F>::type func_type;

                return set_operation(policy,
                    first1, last1, first2, last2, dest, std::forward<F>(f),
                    // calculate approximate destination index
                    [](difference_type1 idx1, difference_type2 idx2)
                    {
                        return idx1 + idx2;
                    },
                    // perform required set operation for one chunk
                    [](RanIter1 part_first1, RanIter1 part_last1,
                        RanIter2 part_first2, RanIter2 part_last2,
                        buffer_type* dest, func_type const& f)
                    {
                        return std::set_symmetric_difference(
                            part_first1, part_last1,
                            part_first2, part_last2, dest, f);
                    });
            }
        };
        /// \endcond
    }

    /// Constructs a sorted range beginning at dest consisting of all elements
    /// present in either of the sorted ranges [first1, last1) and
    /// [first2, last2), but not in both of them are copied to the range
    /// beginning at \a dest. The resulting range is also sorted. This
    /// algorithm expects both input ranges to be sorted with the given binary
    /// predicate \a f.
    ///
    /// \note   Complexity: At most 2*(N1 + N2 - 1) comparisons, where \a N1 is
    ///         the length of the first sequence and \a N2 is the length of the
    ///         second sequence.
    ///
    /// If some element is found \a m times in [first1, last1) and \a n times
    /// in [first2, last2), it will be copied to \a dest exactly std::abs(m-n)
    /// times. If m>n, then the last m-n of those elements are copied from
    /// [first1,last1), otherwise the last n-m elements are copied from
    /// [first2,last2). The resulting range cannot overlap with either of the
    /// input ranges.
    ///
    /// The resulting range cannot overlap with either of the input ranges.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it applies user-provided function objects.
    /// \tparam InIter      The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
    /// \tparam OutIter     The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     output iterator.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a set_symmetric_difference requires
    ///                     \a F to meet the
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
    /// \param last2        Refers to the end of the sequence of elements of
    ///                     the second range the algorithm will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
    /// \param f            The binary predicate which returns true if the
    ///                     elements should be treated as equal. The signature
    ///                     of the predicate function should be equivalent to
    ///                     the following:
    ///                     \code
    ///                     bool pred(const Type1 &a, const Type1 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type1 must be such
    ///                     that objects of type \a InIter can
    ///                     be dereferenced and then implicitly converted to
    ///                     \a Type1
    ///
    /// The application of function objects in parallel algorithm
    /// invoked with a sequential execution policy object execute in sequential
    /// order in the calling thread (\a sequential_execution_policy) or in a
    /// single new thread spawned from the current thread
    /// (for \a sequential_task_execution_policy).
    ///
    /// The application of function objects in parallel algorithm
    /// invoked with an execution policy object of type
    /// \a parallel_execution_policy or \a parallel_task_execution_policy are
    /// permitted to execute in an unordered fashion in unspecified
    /// threads, and indeterminately sequenced within each thread.
    ///
    /// \returns  The \a set_symmetric_difference algorithm returns a
    ///           \a hpx::future<OutIter>
    ///           if the execution policy is of type
    ///           \a sequential_task_execution_policy or
    ///           \a parallel_task_execution_policy and
    ///           returns \a OutIter otherwise.
    ///           The \a set_symmetric_difference algorithm returns the output
    ///           iterator to the
    ///           element in the destination range, one past the last element
    ///           copied.
    ///
    template <typename ExPolicy, typename InIter1, typename InIter2,
        typename OutIter, typename F>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename util::detail::algorithm_result<ExPolicy, OutIter>::type
    >::type
    set_symmetric_difference(ExPolicy && policy, InIter1 first1, InIter1 last1,
        InIter2 first2, InIter2 last2, OutIter dest, F && f)
    {
        typedef typename std::iterator_traits<InIter1>::iterator_category
            input_iterator_category1;
        typedef typename std::iterator_traits<InIter2>::iterator_category
            input_iterator_category2;
        typedef typename std::iterator_traits<OutIter>::iterator_category
            output_iterator_category;

        static_assert(
            (boost::is_base_of<
                std::input_iterator_tag, input_iterator_category1>::value),
            "Requires at least input iterator.");
        static_assert(
            (boost::is_base_of<
                std::input_iterator_tag, input_iterator_category2>::value),
            "Requires at least input iterator.");

        static_assert(
            (boost::mpl::or_<
                boost::is_base_of<
                    std::forward_iterator_tag, output_iterator_category>,
                boost::is_same<
                    std::output_iterator_tag, output_iterator_category>
            >::value),
            "Requires at least output iterator.");

        typedef typename boost::mpl::or_<
            parallel::is_sequential_execution_policy<ExPolicy>,
            boost::mpl::not_<boost::is_same<
                std::random_access_iterator_tag, input_iterator_category1
            > >,
            boost::mpl::not_<boost::is_same<
                std::random_access_iterator_tag, input_iterator_category2
            > >,
            boost::mpl::not_<boost::is_same<
                std::random_access_iterator_tag, output_iterator_category
            > >
        >::type is_seq;

        return detail::set_symmetric_difference<OutIter>().call(
            std::forward<ExPolicy>(policy), is_seq(),
            first1, last1, first2, last2, dest, std::forward<F>(f));
    }

    /// Constructs a sorted range beginning at dest consisting of all elements
    /// present in either of the sorted ranges [first1, last1) and
    /// [first2, last2), but not in both of them are copied to the range
    /// beginning at \a dest. This algorithm expects both input ranges to be
    /// sorted with operator<.
    ///
    /// \note   Complexity: At most 2*(N1 + N2 - 1) comparisons, where \a N1 is
    ///         the length of the first sequence and \a N2 is the length of the
    ///         second sequence.
    ///
    /// If some element is found \a m times in [first1, last1) and \a n times
    /// in [first2, last2), it will be copied to \a dest exactly std::abs(m-n)
    /// times. If m>n, then the last m-n of those elements are copied from
    /// [first1,last1), otherwise the last n-m elements are copied from
    /// [first2,last2). The resulting range cannot overlap with either of the
    /// input ranges.
    ///
    /// The resulting range cannot overlap with either of the input ranges.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it applies user-provided function objects.
    /// \tparam InIter      The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
    /// \tparam OutIter     The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     output iterator.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first1       Refers to the beginning of the sequence of elements
    ///                     of the first range the algorithm will be applied to.
    /// \param last1        Refers to the end of the sequence of elements of
    ///                     the first range the algorithm will be applied to.
    /// \param first2       Refers to the beginning of the sequence of elements
    ///                     of the second range the algorithm will be applied to.
    /// \param last2        Refers to the end of the sequence of elements of
    ///                     the second range the algorithm will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
    ///
    /// The application of function objects in parallel algorithm
    /// invoked with a sequential execution policy object execute in sequential
    /// order in the calling thread (\a sequential_execution_policy) or in a
    /// single new thread spawned from the current thread
    /// (for \a sequential_task_execution_policy).
    ///
    /// The application of function objects in parallel algorithm
    /// invoked with an execution policy object of type
    /// \a parallel_execution_policy or \a parallel_task_execution_policy are
    /// permitted to execute in an unordered fashion in unspecified
    /// threads, and indeterminately sequenced within each thread.
    ///
    /// \returns  The \a set_symmetric_difference algorithm returns a
    ///           \a hpx::future<OutIter>
    ///           if the execution policy is of type
    ///           \a sequential_task_execution_policy or
    ///           \a parallel_task_execution_policy and
    ///           returns \a OutIter otherwise.
    ///           The \a set_symmetric_difference algorithm returns the output
    ///           iterator to the
    ///           element in the destination range, one past the last element
    ///           copied.
    ///
    template <typename ExPolicy, typename InIter1, typename InIter2,
        typename OutIter>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename util::detail::algorithm_result<ExPolicy, OutIter>::type
    >::type
    set_symmetric_difference(ExPolicy && policy, InIter1 first1, InIter1 last1,
        InIter2 first2, InIter2 last2, OutIter dest)
    {
        typedef typename std::iterator_traits<InIter1>::iterator_category
            input_iterator_category1;
        typedef typename std::iterator_traits<InIter2>::iterator_category
            input_iterator_category2;
        typedef typename std::iterator_traits<OutIter>::iterator_category
            output_iterator_category;

        static_assert(
            (boost::is_base_of<
                std::input_iterator_tag, input_iterator_category1>::value),
            "Requires at least input iterator.");
        static_assert(
            (boost::is_base_of<
                std::input_iterator_tag, input_iterator_category2>::value),
            "Requires at least input iterator.");

        static_assert(
            (boost::mpl::or_<
                boost::is_base_of<
                    std::forward_iterator_tag, output_iterator_category>,
                boost::is_same<
                    std::output_iterator_tag, output_iterator_category>
            >::value),
            "Requires at least output iterator.");

        typedef typename boost::mpl::or_<
            parallel::is_sequential_execution_policy<ExPolicy>,
            boost::mpl::not_<boost::is_same<
                std::random_access_iterator_tag, input_iterator_category1
            > >,
            boost::mpl::not_<boost::is_same<
                std::random_access_iterator_tag, input_iterator_category2
            > >,
            boost::mpl::not_<boost::is_same<
                std::random_access_iterator_tag, output_iterator_category
            > >
        >::type is_seq;

        return detail::set_symmetric_difference<OutIter>().call(
            std::forward<ExPolicy>(policy), is_seq(),
            first1, last1, first2, last2, dest,
            std::less<typename std::iterator_traits<InIter1>::value_type>());
    }
}}}

#endif
