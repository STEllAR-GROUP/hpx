//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/all_any_none.hpp

#if !defined(HPX_PARALLEL_DETAIL_ALL_ANY_NONE_JUL_05_2014_0940PM)
#define HPX_PARALLEL_DETAIL_ALL_ANY_NONE_JUL_05_2014_0940PM

#include <hpx/config.hpp>
#include <hpx/util/void_guard.hpp>

#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/partitioner.hpp>
#include <hpx/parallel/util/loop.hpp>

#include <boost/range/functions.hpp>
#include <boost/type_traits/is_base_of.hpp>
#include <boost/utility/enable_if.hpp>

#include <algorithm>
#include <iterator>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
    ///////////////////////////////////////////////////////////////////////////
    // none_of
    namespace detail
    {
        /// \cond NOINTERNAL
        struct none_of : public detail::algorithm<none_of, bool>
        {
            none_of()
              : none_of::algorithm("none_of")
            {}

            template <typename ExPolicy, typename InIter, typename F>
            static bool
            sequential(ExPolicy, InIter first, InIter last, F && f)
            {
                return std::none_of(first, last, std::forward<F>(f));
            }

            template <typename ExPolicy, typename FwdIter, typename F>
            static typename util::detail::algorithm_result<
                ExPolicy, bool
            >::type
            parallel(ExPolicy && policy, FwdIter first, FwdIter last,
                F && op)
            {
                if (first == last)
                {
                    return util::detail::algorithm_result<
                            ExPolicy, bool
                        >::get(true);
                }

                util::cancellation_token<> tok;
                return util::partitioner<ExPolicy, bool>::call(
                    std::forward<ExPolicy>(policy),
                    first, std::distance(first, last),
                    [op, tok](FwdIter part_begin, std::size_t part_count)
                        mutable -> bool
                    {
                        util::loop_n(
                            part_begin, part_count, tok,
                            [&op, &tok](FwdIter const& curr)
                            {
                                if (op(*curr))
                                    tok.cancel();
                            });
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

    ///  Checks if unary predicate \a f returns true for no elements in the
    ///  range [first, last).
    ///
    /// \note   Complexity: At most \a last - \a first applications of the
    ///         predicate \a f
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it applies user-provided function objects.
    /// \tparam InIter      The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a none_of requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param f            Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last).
    ///                     The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     bool pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed
    ///                     to it. The type \a Type must be such that an object
    ///                     of type \a InIter can be dereferenced and then
    ///                     implicitly converted to Type.
    ///
    /// The application of function objects in parallel algorithm
    /// invoked with an execution policy object of type
    /// \a sequential_execution_policy execute in sequential order in the
    /// calling thread.
    ///
    /// The application of function objects in parallel algorithm
    /// invoked with an execution policy object of type
    /// \a parallel_execution_policy or \a parallel_task_execution_policy are
    /// permitted to execute in an unordered fashion in unspecified
    /// threads, and indeterminately sequenced within each thread.
    ///
    /// \returns  The \a none_of algorithm returns a \a hpx::future<bool> if
    ///           the execution policy is of type
    ///           \a sequential_task_execution_policy or
    ///           \a parallel_task_execution_policy and returns \a bool
    ///           otherwise.
    ///           The \a none_of algorithm returns true if the unary predicate
    ///           \a f returns true for no elements in the range, false
    ///           otherwise. It returns true if the range is empty.
    ///
    template <typename ExPolicy, typename InIter, typename F>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename util::detail::algorithm_result<ExPolicy, bool>::type
    >::type
    none_of(ExPolicy && policy, InIter first, InIter last, F && f)
    {
        typedef typename std::iterator_traits<InIter>::iterator_category
            iterator_category;

        static_assert(
            (boost::is_base_of<std::input_iterator_tag, iterator_category>::value),
            "Requires at least input iterator.");

        typedef typename boost::mpl::or_<
            is_sequential_execution_policy<ExPolicy>,
            boost::is_same<std::input_iterator_tag, iterator_category>
        >::type is_seq;

        return detail::none_of().call(
            std::forward<ExPolicy>(policy), is_seq(),
            first, last, std::forward<F>(f));
    }

    ///////////////////////////////////////////////////////////////////////////
    // any_of
    namespace detail
    {
        /// \cond NOINTERNAL
        struct any_of : public detail::algorithm<any_of, bool>
        {
            any_of()
              : any_of::algorithm("any_of")
            {}

            template <typename ExPolicy, typename InIter, typename F>
            static bool
            sequential(ExPolicy, InIter first, InIter last, F && f)
            {
                return std::any_of(first, last, std::forward<F>(f));
            }

            template <typename ExPolicy, typename FwdIter, typename F>
            static typename util::detail::algorithm_result<
                ExPolicy, bool
            >::type
            parallel(ExPolicy && policy, FwdIter first, FwdIter last,
                F && op)
            {
                if (first == last)
                {
                    return util::detail::algorithm_result<
                            ExPolicy, bool
                        >::get(false);
                }

                util::cancellation_token<> tok;
                return util::partitioner<ExPolicy, bool>::call(
                    std::forward<ExPolicy>(policy),
                    first, std::distance(first, last),
                    [op, tok](FwdIter part_begin, std::size_t part_count)
                        mutable -> bool
                    {
                        util::loop_n(
                            part_begin, part_count, tok,
                            [&op, &tok](FwdIter const& curr)
                            {
                                if (op(*curr))
                                    tok.cancel();
                            });
                        return tok.was_cancelled();
                    },
                    [](std::vector<hpx::future<bool> > && results)
                    {
                        return std::any_of(
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

    ///  Checks if unary predicate \a f returns true for at least one element
    ///  in the range [first, last).
    ///
    /// \note   Complexity: At most \a last - \a first applications of the
    ///         predicate \a f
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it applies user-provided function objects.
    /// \tparam InIter      The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a any_of requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param f            Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last).
    ///                     The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     bool pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed
    ///                     to it. The type \a Type must be such that an object
    ///                     of type \a InIter can be dereferenced and then
    ///                     implicitly converted to Type.
    ///
    /// The application of function objects in parallel algorithm
    /// invoked with an execution policy object of type
    /// \a sequential_execution_policy execute in sequential order in the
    /// calling thread.
    ///
    /// The application of function objects in parallel algorithm
    /// invoked with an execution policy object of type
    /// \a parallel_execution_policy or \a parallel_task_execution_policy are
    /// permitted to execute in an unordered fashion in unspecified
    /// threads, and indeterminately sequenced within each thread.
    ///
    /// \returns  The \a any_of algorithm returns a \a hpx::future<bool> if
    ///           the execution policy is of type
    ///           \a sequential_task_execution_policy or
    ///           \a parallel_task_execution_policy and
    ///           returns \a bool otherwise.
    ///           The \a any_of algorithm returns true if the unary predicate
    ///           \a f returns true for at least one element in the range,
    ///           false otherwise. It returns false if the range is empty.
    ///
    template <typename ExPolicy, typename InIter, typename F>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename util::detail::algorithm_result<ExPolicy, bool>::type
    >::type
    any_of(ExPolicy && policy, InIter first, InIter last, F && f)
    {
        typedef typename std::iterator_traits<InIter>::iterator_category
            iterator_category;

        static_assert(
            (boost::is_base_of<std::input_iterator_tag, iterator_category>::value),
            "Requires at least input iterator.");

        typedef typename boost::mpl::or_<
            is_sequential_execution_policy<ExPolicy>,
            boost::is_same<std::input_iterator_tag, iterator_category>
        >::type is_seq;

        return detail::any_of().call(
            std::forward<ExPolicy>(policy), is_seq(),
            first, last, std::forward<F>(f));
    }

    ///////////////////////////////////////////////////////////////////////////
    // all_of
    namespace detail
    {
        /// \cond NOINTERNAL
        struct all_of : public detail::algorithm<all_of, bool>
        {
            all_of()
              : all_of::algorithm("all_of")
            {}

            template <typename ExPolicy, typename InIter, typename F>
            static bool
            sequential(ExPolicy, InIter first, InIter last, F && f)
            {
                return std::all_of(first, last, std::forward<F>(f));
            }

            template <typename ExPolicy, typename FwdIter, typename F>
            static typename util::detail::algorithm_result<
                ExPolicy, bool
            >::type
            parallel(ExPolicy && policy, FwdIter first, FwdIter last, F && op)
            {
                if (first == last)
                {
                    return util::detail::algorithm_result<
                            ExPolicy, bool
                        >::get(true);
                }

                util::cancellation_token<> tok;
                return util::partitioner<ExPolicy, bool>::call(
                    std::forward<ExPolicy>(policy),
                    first, std::distance(first, last),
                    [op, tok](FwdIter part_begin, std::size_t part_count)
                        mutable -> bool
                    {
                        util::loop_n(
                            part_begin, part_count, tok,
                            [&op, &tok](FwdIter const& curr)
                            {
                                if (!op(*curr))
                                    tok.cancel();
                            });
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

    /// Checks if unary predicate \a f returns true for all elements in the
    /// range [first, last).
    ///
    /// \note   Complexity: At most \a last - \a first applications of the
    ///         predicate \a f
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it applies user-provided function objects.
    /// \tparam InIter      The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a all_of requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param f            Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last).
    ///                     The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     bool pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed
    ///                     to it. The type \a Type must be such that an object
    ///                     of type \a InIter can be dereferenced and then
    ///                     implicitly converted to Type.
    ///
    /// The application of function objects in parallel algorithm
    /// invoked with an execution policy object of type
    /// \a sequential_execution_policy execute in sequential order in the
    /// calling thread.
    ///
    /// The application of function objects in parallel algorithm
    /// invoked with an execution policy object of type
    /// \a parallel_execution_policy or \a parallel_task_execution_policy are
    /// permitted to execute in an unordered fashion in unspecified
    /// threads, and indeterminately sequenced within each thread.
    ///
    /// \returns  The \a all_of algorithm returns a \a hpx::future<bool> if
    ///           the execution policy is of type
    ///           \a sequential_task_execution_policy or
    ///           \a parallel_task_execution_policy and
    ///           returns \a bool otherwise.
    ///           The \a all_of algorithm returns true if the unary predicate
    ///           \a f returns true for all elements in the range, false
    ///           otherwise. It returns true if the range is empty.
    ///
    template <typename ExPolicy, typename InIter, typename F>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename util::detail::algorithm_result<ExPolicy, bool>::type
    >::type
    all_of(ExPolicy && policy, InIter first, InIter last, F && f)
    {
        typedef typename std::iterator_traits<InIter>::iterator_category
            iterator_category;

        static_assert(
            (boost::is_base_of<std::input_iterator_tag, iterator_category>::value),
            "Requires at least input iterator.");

        typedef typename boost::mpl::or_<
            is_sequential_execution_policy<ExPolicy>,
            boost::is_same<std::input_iterator_tag, iterator_category>
        >::type is_seq;

        return detail::all_of().call(
            std::forward<ExPolicy>(policy), is_seq(),
            first, last, std::forward<F>(f));
    }
}}}

#endif
