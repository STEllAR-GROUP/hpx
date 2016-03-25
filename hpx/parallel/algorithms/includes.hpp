//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/includes.hpp

#if !defined(HPX_PARALLEL_ALGORITH_INCLUDES_MAR_10_2015_0737PM)
#define HPX_PARALLEL_ALGORITH_INCLUDES_MAR_10_2015_0737PM

#include <hpx/config.hpp>

#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/detail/predicates.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/cancellation_token.hpp>
#include <hpx/parallel/util/partitioner.hpp>

#include <boost/mpl/or.hpp>
#include <boost/range/functions.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/type_traits/is_base_of.hpp>
#include <boost/utility/enable_if.hpp>

#include <algorithm>
#include <iterator>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
    ///////////////////////////////////////////////////////////////////////////
    // includes
    namespace detail
    {
        /// \cond NOINTERNAL

        ///////////////////////////////////////////////////////////////////////
        template <typename FwdIter, typename T, typename F, typename CancelToken>
        FwdIter lower_bound(FwdIter first, FwdIter last, T const& value,
            F && f, CancelToken& tok)
        {
            typedef typename std::iterator_traits<FwdIter>::difference_type
                difference_type;

            difference_type count = std::distance(first, last);
            while (count > 0)
            {
                if (tok.was_cancelled())
                    break;

                difference_type step = count / 2;
                FwdIter it = detail::next(first, step);

                if (f(*it, value))
                {
                    first = ++it;
                    count -= step + 1;
                }
                else
                {
                    count = step;
                }
            }
            return first;
        }

        template <typename FwdIter, typename T, typename F, typename CancelToken>
        FwdIter upper_bound(FwdIter first, FwdIter last, T const& value,
            F && f, CancelToken& tok)
        {
            typedef typename std::iterator_traits<FwdIter>::difference_type
                difference_type;

            difference_type count = std::distance(first, last);
            while (count > 0)
            {
                if (tok.was_cancelled())
                    break;

                difference_type step = count / 2;
                FwdIter it = detail::next(first, step);

                if (!f(*it, value))
                {
                    first = ++it;
                    count -= step + 1;
                }
                else
                {
                    count = step;
                }
            }
            return first;
        }

        template <typename InIter1, typename InIter2, typename F,
            typename CancelToken>
        bool sequential_includes(InIter1 first1, InIter1 last1, InIter2 first2,
            InIter2 last2, F && f, CancelToken& tok)
        {
            while (first2 != last2)
            {
                if (tok.was_cancelled())
                    return false;

                if (first1 == last1 || f(*first2, *first1))
                    return false;
                if (!f(*first1, *first2))
                    ++first2;

                ++first1;
            }
            return true;
        }

        ///////////////////////////////////////////////////////////////////////
        struct includes : public detail::algorithm<includes, bool>
        {
            includes()
              : includes::algorithm("includes")
            {}

            template <typename ExPolicy, typename InIter1, typename InIter2,
                typename F>
            static bool
            sequential(ExPolicy, InIter1 first1, InIter1 last1,
                InIter2 first2, InIter2 last2, F && f)
            {
                return std::includes(first1, last1, first2, last2,
                    std::forward<F>(f));
            }

            template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
                typename F>
            static typename util::detail::algorithm_result<ExPolicy, bool>::type
            parallel(ExPolicy && policy, FwdIter1 first1, FwdIter1 last1,
                FwdIter2 first2, FwdIter2 last2, F && f)
            {
                if (first1 == last1)
                {
                    return util::detail::algorithm_result<ExPolicy, bool>::get(
                        false);
                }

                if (first2 == last2)
                {
                    return util::detail::algorithm_result<ExPolicy, bool>::get(
                        true);
                }

                util::cancellation_token<> tok;
                return util::partitioner<ExPolicy, bool>::call(
                    std::forward<ExPolicy>(policy),
                    first2, std::distance(first2, last2),
                    [first1, last1, first2, last2, f, tok](
                            FwdIter2 part_begin, std::size_t part_count
                        ) mutable -> bool
                    {
                        FwdIter2 part_end = detail::next(part_begin, part_count);
                        if (first2 != part_begin)
                        {
                            part_begin = upper_bound(part_begin, part_end,
                                *part_begin, f, tok);
                            if (tok.was_cancelled())
                                return false;
                            if (part_begin == part_end)
                                return true;
                        }

                        FwdIter1 low = lower_bound(first1, last1,
                            *part_begin, f, tok);
                        if (tok.was_cancelled())
                            return false;

                        if (low == last1 || f(*part_begin, *low))
                        {
                            tok.cancel();
                            return false;
                        }

                        FwdIter1 high = last1;
                        if (part_end != last2)
                        {
                            high = upper_bound(low, last1, *part_end, f, tok);
                            part_end = upper_bound(part_end, last2,
                                *part_end, f, tok);
                            if (tok.was_cancelled())
                                return false;
                        }

                        if (!sequential_includes(low, high, part_begin,
                            part_end, f, tok))
                        {
                            tok.cancel();
                        }
                        return !tok.was_cancelled();
                    },
                    [](std::vector<hpx::future<bool> > && results)
                    {
                        return std::all_of(
                            boost::begin(results), boost::end(results),
                            [](hpx::future<bool>& val)
                            {
                                return val.get();
                            });
                    });
            }
        };
        /// \endcond
    }

    /// Returns true if every element from the sorted range [first2, last2) is
    /// found within the sorted range [first1, last1). Also returns true if
    /// [first2, last2) is empty. The version expects both ranges to be sorted
    /// with operator<().
    ///
    /// \note   At most 2*(N1+N2-1) comparisons, where
    ///         N1 = std::distance(first1, last1) and
    ///         N2 = std::distance(first2, last2).
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
    /// \param last2        Refers to the end of the sequence of elements of
    ///                     the second range the algorithm will be applied to.
    ///
    /// The comparison operations in the parallel \a includes algorithm invoked
    /// with an execution policy object of type \a sequential_execution_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The comparison operations in the parallel \a includes algorithm invoked
    /// with an execution policy object of type \a parallel_execution_policy
    /// or \a parallel_task_execution_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a includes algorithm returns a \a hpx::future<bool> if the
    ///           execution policy is of type
    ///           \a sequential_task_execution_policy or
    ///           \a parallel_task_execution_policy and
    ///           returns \a bool otherwise.
    ///           The \a includes algorithm returns true every element from the
    ///           sorted range [first2, last2) is found within the sorted range
    ///           [first1, last1). Also returns true if [first2, last2) is empty.
    ///
    template <typename ExPolicy, typename InIter1, typename InIter2>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename util::detail::algorithm_result<ExPolicy, bool>::type
    >::type
    includes(ExPolicy&& policy, InIter1 first1, InIter1 last1,
        InIter2 first2, InIter2 last2)
    {
        typedef typename std::iterator_traits<InIter1>::iterator_category
            iterator_category1;
        typedef typename std::iterator_traits<InIter2>::iterator_category
            iterator_category2;

        static_assert(
            (boost::is_base_of<std::input_iterator_tag, iterator_category1>::value),
            "Requires at least input iterator.");
        static_assert(
            (boost::is_base_of<std::input_iterator_tag, iterator_category2>::value),
            "Requires at least input iterator.");

        typedef typename boost::mpl::or_<
            is_sequential_execution_policy<ExPolicy>,
            boost::is_same<std::input_iterator_tag, iterator_category1>,
            boost::is_same<std::input_iterator_tag, iterator_category2>
        >::type is_seq;

        return detail::includes().call(
            std::forward<ExPolicy>(policy), is_seq(),
            first1, last1, first2, last2,
            std::less<typename std::iterator_traits<InIter1>::value_type>());
    }

    /// Returns true if every element from the sorted range [first2, last2) is
    /// found within the sorted range [first1, last1). Also returns true if
    /// [first2, last2) is empty. The version expects both ranges to be sorted
    /// with the user supplied binary predicate \a f.
    ///
    /// \note   At most 2*(N1+N2-1) comparisons, where
    ///         N1 = std::distance(first1, last1) and
    ///         N2 = std::distance(first2, last2).
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
    ///                     overload of \a includes requires \a F to meet the
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
    /// \param f            The binary predicate which returns true if the
    ///                     elements should be treated as includes. The signature
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
    /// The comparison operations in the parallel \a includes algorithm invoked
    /// with an execution policy object of type \a sequential_execution_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The comparison operations in the parallel \a includes algorithm invoked
    /// with an execution policy object of type \a parallel_execution_policy
    /// or \a parallel_task_execution_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a includes algorithm returns a \a hpx::future<bool> if the
    ///           execution policy is of type
    ///           \a sequential_task_execution_policy or
    ///           \a parallel_task_execution_policy and
    ///           returns \a bool otherwise.
    ///           The \a includes algorithm returns true every element from the
    ///           sorted range [first2, last2) is found within the sorted range
    ///           [first1, last1). Also returns true if [first2, last2) is empty.
    ///
    template <typename ExPolicy, typename InIter1, typename InIter2, typename F>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename util::detail::algorithm_result<ExPolicy, bool>::type
    >::type
    includes(ExPolicy&& policy, InIter1 first1, InIter1 last1,
        InIter2 first2, InIter2 last2, F && f)
    {
        typedef typename std::iterator_traits<InIter1>::iterator_category
            iterator_category1;
        typedef typename std::iterator_traits<InIter2>::iterator_category
            iterator_category2;

        static_assert(
            (boost::is_base_of<std::input_iterator_tag, iterator_category1>::value),
            "Requires at least input iterator.");
        static_assert(
            (boost::is_base_of<std::input_iterator_tag, iterator_category2>::value),
            "Requires at least input iterator.");

        typedef typename boost::mpl::or_<
            is_sequential_execution_policy<ExPolicy>,
            boost::is_same<std::input_iterator_tag, iterator_category1>,
            boost::is_same<std::input_iterator_tag, iterator_category2>
        >::type is_seq;

        return detail::includes().call(
            std::forward<ExPolicy>(policy), is_seq(),
            first1, last1, first2, last2, std::forward<F>(f));
    }
}}}

#endif
