//  Copyright (c) 2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/replace.hpp

#if !defined(HPX_PARALLEL_DETAIL_REPLACE_AUG_18_2014_0136PM)
#define HPX_PARALLEL_DETAIL_REPLACE_AUG_18_2014_0136PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/move.hpp>
#include <hpx/util/unused.hpp>
#include <hpx/util/void_guard.hpp>

#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/for_each.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/zip_iterator.hpp>

#include <algorithm>
#include <iterator>

#include <boost/static_assert.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_base_of.hpp>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
    ///////////////////////////////////////////////////////////////////////////
    // replace
    namespace detail
    {
        /// \cond NOINTERNAL
        struct replace : public detail::algorithm<replace>
        {
            replace()
              : replace::algorithm("replace")
            {}

            template <typename ExPolicy, typename FwdIter, typename T>
            static hpx::util::unused_type
            sequential(ExPolicy, FwdIter first, FwdIter last,
                T const& old_value, T const& new_value)
            {
                std::replace(first, last, old_value, new_value);
                return hpx::util::unused;
            }

            template <typename ExPolicy, typename FwdIter, typename T>
            static typename util::detail::algorithm_result<ExPolicy>::type
            parallel(ExPolicy policy, FwdIter first, FwdIter last,
                T const& old_value, T const& new_value)
            {
                typedef typename util::detail::algorithm_result<ExPolicy>::type
                    result_type;
                typedef typename std::iterator_traits<FwdIter>::value_type type;

                return hpx::util::void_guard<result_type>(),
                    for_each_n<FwdIter>().call(
                        policy, boost::mpl::false_(),
                        first, std::distance(first, last),
                        [old_value, new_value](type& t) {
                            if (t == old_value)
                                t = new_value;
                        });
            }
        };
        /// \endcond
    }

    /// Replaces all elements satisfying specific criteria with \a new_value
    /// in the range [first, last).
    ///
    /// \note   Complexity: Performs exactly \a last - \a first assignments.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter     The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     forward iterator.
    /// \tparam T           The type of the old and new values to replace
    ///                     (deduced).
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param old_value    Refers to the old value of the elements to replace.
    /// \param new_value    Refers to the new value to use as the replacement.
    ///
    /// The assignments in the parallel \a replace algorithm invoked with an
    /// execution policy object of type \a sequential_execution_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a replace algorithm invoked with
    /// an execution policy object of type \a parallel_execution_policy or
    /// \a parallel_task_execution_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a replace algorithm returns a \a hpx::future<void> if
    ///           the execution policy is of type
    ///           \a sequential_task_execution_policy or
    ///           \a parallel_task_execution_policy and
    ///           returns \a void otherwise.
    ///
    template <typename ExPolicy, typename FwdIter, typename T>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename util::detail::algorithm_result<ExPolicy>::type
    >::type
    replace(ExPolicy && policy, FwdIter first, FwdIter last,
        T const& old_value, T const& new_value)
    {
        typedef typename std::iterator_traits<FwdIter>::iterator_category
            iterator_category;

        BOOST_STATIC_ASSERT_MSG(
            (boost::is_base_of<
                std::forward_iterator_tag, iterator_category>::value),
            "Required at least forward iterator.");

        typedef typename is_sequential_execution_policy<ExPolicy>::type is_seq;

        return detail::replace().call(
            std::forward<ExPolicy>(policy), is_seq(),
            first, last, old_value, new_value);
    }

    ///////////////////////////////////////////////////////////////////////////
    // replace_if
    namespace detail
    {
        /// \cond NOINTERNAL
        struct replace_if : public detail::algorithm<replace_if>
        {
            replace_if()
              : replace_if::algorithm("replace_if")
            {}

            template <typename ExPolicy, typename FwdIter, typename F, typename T>
            static hpx::util::unused_type
            sequential(ExPolicy, FwdIter first, FwdIter last, F && f,
                T const& new_value)
            {
                std::replace_if(first, last, std::forward<F>(f), new_value);
                return hpx::util::unused;
            }

            template <typename ExPolicy, typename FwdIter, typename F, typename T>
            static typename util::detail::algorithm_result<ExPolicy>::type
            parallel(ExPolicy policy, FwdIter first, FwdIter last,
                F && f, T const& new_value)
            {
                typedef typename util::detail::algorithm_result<ExPolicy>::type
                    result_type;
                typedef typename std::iterator_traits<FwdIter>::value_type type;

                return hpx::util::void_guard<result_type>(),
                    for_each_n<FwdIter>().call(
                        policy, boost::mpl::false_(),
                        first, std::distance(first, last),
                        [f, new_value](type& t) {
                            if (f(t))
                                t = new_value;
                        });
            }
        };
        /// \endcond
    }

    /// Replaces all elements satisfying specific criteria (for which predicate
    /// \a f returns true) with \a new_value in the range [first, last).
    ///
    /// \note   Complexity: Performs exactly \a last - \a first applications of
    ///         the predicate.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter     The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     forward iterator.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a equal requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    ///                     (deduced).
    /// \tparam T           The type of the new values to replace (deduced).
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param f            Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last).This is an
    ///                     unary predicate which returns \a true for the
    ///                     elements which need to replaced. The
    ///                     signature of this predicate should be equivalent
    ///                     to:
    ///                     \code
    ///                     bool pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that an object of
    ///                     type \a FwdIter can be dereferenced and then
    ///                     implicitly converted to \a Type.
    /// \param new_value    Refers to the new value to use as the replacement.
    ///
    /// The assignments in the parallel \a replace_if algorithm invoked with an
    /// execution policy object of type \a sequential_execution_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a replace_if algorithm invoked with
    /// an execution policy object of type \a parallel_execution_policy or
    /// \a parallel_task_execution_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a replace_if algorithm returns a \a hpx::future<void>
    ///           if the execution policy is of type
    ///           \a sequential_task_execution_policy or
    ///           \a parallel_task_execution_policy
    ///           and returns \a void otherwise.
    ///
    template <typename ExPolicy, typename FwdIter, typename F, typename T>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename util::detail::algorithm_result<ExPolicy>::type
    >::type
    replace_if(ExPolicy && policy, FwdIter first, FwdIter last,
        F && f, T const& new_value)
    {
        typedef typename std::iterator_traits<FwdIter>::iterator_category
            iterator_category;

        BOOST_STATIC_ASSERT_MSG(
            (boost::is_base_of<
                std::forward_iterator_tag, iterator_category>::value),
            "Required at least forward iterator.");

        typedef typename is_sequential_execution_policy<ExPolicy>::type is_seq;

        return detail::replace_if().call(
            std::forward<ExPolicy>(policy), is_seq(),
            first, last, std::forward<F>(f), new_value);
    }

    ///////////////////////////////////////////////////////////////////////////
    // replace_copy
    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename OutIter>
        struct replace_copy
          : public detail::algorithm<replace_copy<OutIter>, OutIter>
        {
            replace_copy()
              : replace_copy::algorithm("replace_copy")
            {}

            template <typename ExPolicy, typename InIter, typename T>
            static OutIter
            sequential(ExPolicy, InIter first, InIter last,
                OutIter dest, T const& old_value, T const& new_value)
            {
                return std::replace_copy(first, last, dest, old_value, new_value);
            }

            template <typename ExPolicy, typename FwdIter, typename T>
            static typename util::detail::algorithm_result<
                ExPolicy, OutIter
            >::type
            parallel(ExPolicy policy, FwdIter first, FwdIter last,
                OutIter dest, T const& old_value, T const& new_value)
            {
                typedef hpx::util::zip_iterator<FwdIter, OutIter> zip_iterator;
                typedef typename zip_iterator::reference reference;
                typedef typename util::detail::algorithm_result<
                        ExPolicy, OutIter
                    >::type result_type;

                return get_iter<1, result_type>(
                    for_each_n<zip_iterator>().call(
                        policy, boost::mpl::false_(),
                        hpx::util::make_zip_iterator(first, dest),
                        std::distance(first, last),
                        [old_value, new_value](reference t) {
                            using hpx::util::get;
                            if (old_value == get<0>(t))
                                get<1>(t) = new_value;
                            else
                                get<1>(t) = get<0>(t); //-V573
                        }));
            }
        };
        /// \endcond
    }

    /// Copies the all elements from the range [first, last) to another range
    /// beginning at \a dest replacing all elements satisfying a specific
    /// criteria with \a new_value.
    ///
    /// \note   Complexity: Performs exactly \a last - \a first applications of
    ///         the predicate.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam InIter      The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
    /// \tparam OutIter     The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     output iterator.
    /// \tparam T           The type of the old and new values to replace
    ///                     (deduced).
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
    /// \param old_value    Refers to the old value of the elements to replace.
    /// \param new_value    Refers to the new value to use as the replacement.
    ///
    /// The assignments in the parallel \a replace_copy algorithm invoked
    /// with an execution policy object of type \a sequential_execution_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a replace_copy algorithm invoked
    /// with an execution policy object of type \a parallel_execution_policy or
    /// \a parallel_task_execution_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a replace_copy algorithm returns a \a hpx::future<OutIter>
    ///           if the execution policy is of type
    ///           \a sequential_task_execution_policy or
    ///           \a parallel_task_execution_policy
    ///           and returns \a OutIter otherwise.
    ///           The \a replace_copy algorithm returns the output iterator
    ///           to the element in the destination range, one past the last
    ///           element copied.
    ///
    template <typename ExPolicy, typename InIter, typename OutIter, typename T>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename util::detail::algorithm_result<ExPolicy, OutIter>::type
    >::type
    replace_copy(ExPolicy && policy, InIter first, InIter last, OutIter dest,
        T const& old_value, T const& new_value)
    {
        typedef typename std::iterator_traits<InIter>::iterator_category
            input_iterator_category;
        typedef typename std::iterator_traits<OutIter>::iterator_category
            output_iterator_category;

        BOOST_STATIC_ASSERT_MSG(
            (boost::is_base_of<
                std::input_iterator_tag, input_iterator_category>::value),
            "Required at least forward iterator.");

        BOOST_STATIC_ASSERT_MSG(
            (boost::mpl::or_<
                boost::is_base_of<
                    std::forward_iterator_tag, output_iterator_category>,
                boost::is_same<
                    std::output_iterator_tag, output_iterator_category>
            >::value),
            "Requires at least output iterator.");

        typedef typename boost::mpl::or_<
            is_sequential_execution_policy<ExPolicy>,
            boost::is_same<std::input_iterator_tag, input_iterator_category>,
            boost::is_same<std::output_iterator_tag, output_iterator_category>
        >::type is_seq;

        return detail::replace_copy<OutIter>().call(
            std::forward<ExPolicy>(policy), is_seq(),
            first, last, dest, old_value, new_value);
    }

    ///////////////////////////////////////////////////////////////////////////
    // replace_copy_if
    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename OutIter>
        struct replace_copy_if
          : public detail::algorithm<replace_copy_if<OutIter>, OutIter>
        {
            replace_copy_if()
              : replace_copy_if::algorithm("replace_copy_if")
            {}

            template <typename ExPolicy, typename InIter, typename F, typename T>
            static OutIter
            sequential(ExPolicy, InIter first, InIter last,
                OutIter dest, F && f, T const& new_value)
            {
                return std::replace_copy_if(first, last, dest,
                    std::forward<F>(f), new_value);
            }

            template <typename ExPolicy, typename FwdIter, typename F, typename T>
            static typename util::detail::algorithm_result<
                ExPolicy, OutIter
            >::type
            parallel(ExPolicy policy, FwdIter first, FwdIter last,
                OutIter dest, F && f, T const& new_value)
            {
                typedef hpx::util::zip_iterator<FwdIter, OutIter> zip_iterator;
                typedef typename zip_iterator::reference reference;
                typedef typename util::detail::algorithm_result<
                        ExPolicy, OutIter
                    >::type result_type;

                return get_iter<1, result_type>(
                    for_each_n<zip_iterator>().call(
                        policy, boost::mpl::false_(),
                        hpx::util::make_zip_iterator(first, dest),
                        std::distance(first, last),
                        [f, new_value](reference t) {
                            using hpx::util::get;
                            if (f(get<0>(t)))
                                get<1>(t) = new_value;
                            else
                                get<1>(t) = get<0>(t); //-V573
                        }));
            }
        };
        /// \endcond
    }

    /// Copies the all elements from the range [first, last) to another range
    /// beginning at \a dest replacing all elements satisfying a specific
    /// criteria with \a new_value.
    ///
    /// \note   Complexity: Performs exactly \a last - \a first applications of
    ///         the predicate.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam InIter      The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
    /// \tparam OutIter     The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     output iterator.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a equal requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    ///                     (deduced).
    /// \tparam T           The type of the new values to replace (deduced).
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
    /// \param f            Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last).This is an
    ///                     unary predicate which returns \a true for the
    ///                     elements which need to replaced. The
    ///                     signature of this predicate should be equivalent
    ///                     to:
    ///                     \code
    ///                     bool pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that an object of
    ///                     type \a FwdIter can be dereferenced and then
    ///                     implicitly converted to \a Type.
    /// \param new_value    Refers to the new value to use as the replacement.
    ///
    /// The assignments in the parallel \a replace_copy_if algorithm invoked
    /// with an execution policy object of type \a sequential_execution_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a replace_copy_if algorithm invoked
    /// with an execution policy object of type \a parallel_execution_policy or
    /// \a parallel_task_execution_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a replace_copy_if algorithm returns a \a hpx::future<OutIter>
    ///           if the execution policy is of type
    ///           \a sequential_task_execution_policy or
    ///           \a parallel_task_execution_policy
    ///           and returns \a OutIter otherwise.
    ///           The \a replace_copy_if algorithm returns the output iterator
    ///           to the element in the destination range, one past the last
    ///           element copied.
    ///
    template <typename ExPolicy, typename InIter, typename OutIter, typename F,
        typename T>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename util::detail::algorithm_result<ExPolicy, OutIter>::type
    >::type
    replace_copy_if(ExPolicy && policy, InIter first, InIter last, OutIter dest,
        F && f, T const& new_value)
    {
        typedef typename std::iterator_traits<InIter>::iterator_category
            input_iterator_category;
        typedef typename std::iterator_traits<OutIter>::iterator_category
            output_iterator_category;

        BOOST_STATIC_ASSERT_MSG(
            (boost::is_base_of<
                std::input_iterator_tag, input_iterator_category>::value),
            "Required at least forward iterator.");

        BOOST_STATIC_ASSERT_MSG(
            (boost::mpl::or_<
                boost::is_base_of<
                    std::forward_iterator_tag, output_iterator_category>,
                boost::is_same<
                    std::output_iterator_tag, output_iterator_category>
            >::value),
            "Requires at least output iterator.");

        typedef typename boost::mpl::or_<
            is_sequential_execution_policy<ExPolicy>,
            boost::is_same<std::input_iterator_tag, input_iterator_category>,
            boost::is_same<std::output_iterator_tag, output_iterator_category>
        >::type is_seq;

        return detail::replace_copy_if<OutIter>().call(
            std::forward<ExPolicy>(policy), is_seq(),
            first, last, dest, std::forward<F>(f), new_value);
    }
}}}

#endif
