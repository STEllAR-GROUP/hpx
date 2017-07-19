//  Copyright (c) 2014 Grant Mercer
//  Copyright (c) 2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/find.hpp

#if !defined(HPX_PARALLEL_DETAIL_FIND_JULY_16_2014_0213PM)
#define HPX_PARALLEL_DETAIL_FIND_JULY_16_2014_0213PM

#include <hpx/config.hpp>
#include <hpx/traits/is_iterator.hpp>
#include <hpx/util/invoke.hpp>

#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/detail/predicates.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/loop.hpp>
#include <hpx/parallel/util/partitioner.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace parallel { inline namespace v1
{
    ///////////////////////////////////////////////////////////////////////////
    // find
    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename FwdIter>
        struct find : public detail::algorithm<find<FwdIter>, FwdIter>
        {
            find()
                : find::algorithm("find")
            {}

            template <typename ExPolicy, typename InIter, typename T>
            static InIter
            sequential(ExPolicy, InIter first, InIter last, T const& val)
            {
                return std::find(first, last, val);
            }

            template <typename ExPolicy, typename InIter, typename T>
            static typename util::detail::algorithm_result<
                ExPolicy, FwdIter
            >::type
            parallel(ExPolicy && policy, FwdIter first, FwdIter last,
                T const& val)
            {
                typedef util::detail::algorithm_result<ExPolicy, FwdIter> result;
                typedef typename std::iterator_traits<FwdIter>::value_type type;
                typedef typename std::iterator_traits<FwdIter>::difference_type
                    difference_type;

                difference_type count = std::distance(first, last);
                if (count <= 0)
                    return result::get(std::move(last));

                util::cancellation_token<std::size_t> tok(count);

                return util::partitioner<ExPolicy, FwdIter, void>::
                    call_with_index(
                        std::forward<ExPolicy>(policy), first, count, 1,
                        [val, tok](FwdIter it, std::size_t part_size,
                            std::size_t base_idx) mutable -> void
                        {
                            util::loop_idx_n(
                                base_idx, it, part_size, tok,
                                [&val, &tok](type& v, std::size_t i) -> void
                                {
                                    if (v == val)
                                        tok.cancel(i);
                                });
                        },
                        [=](std::vector<hpx::future<void> > &&) mutable -> FwdIter
                        {
                            difference_type find_res =
                                static_cast<difference_type>(tok.get_data());

                            if (find_res != count)
                                std::advance(first, find_res);
                            else
                                first = last;

                            return std::move(first);
                        });
            }
        };

        template <typename ExPolicy, typename FwdIter, typename T>
        inline typename std::enable_if<
            execution::is_execution_policy<ExPolicy>::value,
            typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
        >::type
        find_(ExPolicy && policy, FwdIter first, FwdIter last, T const& val,
            std::false_type)
        {
#if defined(HPX_HAVE_ALGORITHM_INPUT_ITERATOR_SUPPORT)
            static_assert(
                (hpx::traits::is_input_iterator<FwdIter>::value),
                "Requires at least input iterator.");
            typedef std::integral_constant<bool,
                    execution::is_sequenced_execution_policy<ExPolicy>::value ||
                   !hpx::traits::is_forward_iterator<FwdIter>::value
                > is_seq;
#else
            static_assert(
                (hpx::traits::is_forward_iterator<FwdIter>::value),
                "Requires at least forward iterator.");

            typedef execution::is_sequenced_execution_policy<ExPolicy> is_seq;
#endif
            return detail::find<FwdIter>().call(
                std::forward<ExPolicy>(policy), is_seq(),
                first, last, std::move(val));
        }

        template <typename ExPolicy, typename FwdIter, typename T>
        inline typename std::enable_if<
            execution::is_execution_policy<ExPolicy>::value,
            typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
        >::type
        find_(ExPolicy && policy, FwdIter first, FwdIter last, T const& val,
            std::true_type);
        /// \endcond
    }

    /// Returns the first element in the range [first, last) that is equal
    /// to value
    ///
    /// \note   Complexity: At most last - first
    ///         applications of the operator==().
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter     The type of the source iterators used for the
    ///                     first range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam T           The type of the value to find (deduced).
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     of the first range the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements of
    ///                     the first range the algorithm will be applied to.
    /// \param val          the value to compare the elements to
    ///
    /// The comparison operations in the parallel \a find algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The comparison operations in the parallel \a find algorithm invoked
    /// with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a find algorithm returns a \a hpx::future<FwdIter> if the
    ///           execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a FwdIter otherwise.
    ///           The \a find algorithm returns the first element in the range
    ///           [first,last) that is equal to \a val.
    ///           If no such element in the range of [first,last) is equal to
    ///           \a val, then the algorithm returns \a last.
    ///
    template <typename ExPolicy, typename FwdIter, typename T>
    inline typename std::enable_if<
        execution::is_execution_policy<ExPolicy>::value,
        typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
    >::type
    find(ExPolicy && policy, FwdIter first, FwdIter last, T const& val)
    {
        typedef hpx::traits::is_segmented_iterator<FwdIter> is_segmented;

        return detail::find_(std::forward<ExPolicy>(policy), first, last,
            std::move(val), is_segmented());

    }

    ///////////////////////////////////////////////////////////////////////////
    // find_if
    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename Iter>
        struct find_if : public detail::algorithm<find_if<Iter>, Iter>
        {
            find_if()
              : find_if::algorithm("find_if")
            {}

            template <typename ExPolicy, typename InIter, typename F>
            static InIter
            sequential(ExPolicy, InIter first, InIter last, F && f)
            {
                return std::find_if(first, last, f);
            }

            template <typename ExPolicy, typename FwdIter, typename F>
            static typename util::detail::algorithm_result<
                ExPolicy, FwdIter
            >::type
            parallel(ExPolicy && policy, FwdIter first, FwdIter last, F && f)
            {
                typedef util::detail::algorithm_result<ExPolicy, FwdIter> result;
                typedef typename std::iterator_traits<Iter>::value_type type;
                typedef typename std::iterator_traits<Iter>::difference_type
                    difference_type;

                difference_type count = std::distance(first, last);
                if (count <= 0)
                    return result::get(std::move(last));

                util::cancellation_token<std::size_t> tok(count);

                return util::partitioner<ExPolicy, FwdIter, void>::
                    call_with_index(
                        std::forward<ExPolicy>(policy), first, count, 1,
                        [f, tok](FwdIter it, std::size_t part_size,
                            std::size_t base_idx) mutable -> void
                        {
                            util::loop_idx_n(
                                base_idx, it, part_size, tok,
                                [&f, &tok](type& v, std::size_t i) -> void
                                {
                                    if (hpx::util::invoke(f, v))
                                        tok.cancel(i);
                                });
                        },
                        [=](std::vector<hpx::future<void> > &&) mutable -> FwdIter
                        {
                            difference_type find_res =
                                static_cast<difference_type>(tok.get_data());

                            if (find_res != count)
                                std::advance(first, find_res);
                            else
                                first = last;

                            return std::move(first);
                        });
            }
        };

        template <typename ExPolicy, typename FwdIter, typename F>
        inline typename std::enable_if<
            execution::is_execution_policy<ExPolicy>::value,
            typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
        >::type
        find_if_(ExPolicy && policy, FwdIter first, FwdIter last, F && f,
            std::false_type)
        {
#if defined(HPX_HAVE_ALGORITHM_INPUT_ITERATOR_SUPPORT)
            static_assert(
                (hpx::traits::is_input_iterator<FwdIter>::value),
                "Requires at least input iterator.");
            typedef std::integral_constant<bool,
                    execution::is_sequenced_execution_policy<ExPolicy>::value ||
                   !hpx::traits::is_forward_iterator<FwdIter>::value
                > is_seq;
#else
            static_assert(
                (hpx::traits::is_forward_iterator<FwdIter>::value),
                "Requires at least forward iterator.");

            typedef execution::is_sequenced_execution_policy<ExPolicy> is_seq;
#endif
            return detail::find_if<FwdIter>().call(
                std::forward<ExPolicy>(policy), is_seq(),
                first, last, std::forward<F>(f));
        }

        template <typename ExPolicy, typename FwdIter, typename F>
        inline typename std::enable_if<
            execution::is_execution_policy<ExPolicy>::value,
            typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
        >::type
        find_if_(ExPolicy && policy, FwdIter first, FwdIter last, F && f,
            std::true_type);
        /// \endcond
    }

    /// Returns the first element in the range [first, last) for which
    /// predicate \a f returns true
    ///
    /// \note   Complexity: At most last - first
    ///         applications of the predicate.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter     The type of the source iterators used for the
    ///                     first range (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     forward iterator.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a equal requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     of the first range the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements of
    ///                     the first range the algorithm will be applied to.
    /// \param f            The unary predicate which returns true for the
    ///                     required element. The signature of the predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     bool pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such
    ///                     that objects of type \a FwdIter can
    ///                     be dereferenced and then implicitly converted to
    ///                     \a Type.
    ///
    /// The comparison operations in the parallel \a find_if algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The comparison operations in the parallel \a find_if algorithm invoked
    /// with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a find_if algorithm returns a \a hpx::future<FwdIter> if the
    ///           execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a FwdIter otherwise.
    ///           The \a find_if algorithm returns the first element in the range
    ///           [first,last) that satisfies the predicate \a f.
    ///           If no such element exists that satisfies the predicate f, the
    ///           algorithm returns \a last.
    ///
    template <typename ExPolicy, typename FwdIter, typename F>
    inline typename std::enable_if<
        execution::is_execution_policy<ExPolicy>::value,
        typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
    >::type
    find_if(ExPolicy && policy, FwdIter first, FwdIter last, F && f)
    {
        typedef hpx::traits::is_segmented_iterator<FwdIter> is_segmented;

        return detail::find_if_(std::forward<ExPolicy>(policy), first, last,
            std::forward<F>(f), is_segmented());
    }

    ///////////////////////////////////////////////////////////////////////////
    // find_if_not
    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename Iter>
        struct find_if_not
          : public detail::algorithm<find_if_not<Iter>, Iter>
        {
            find_if_not()
              : find_if_not::algorithm("find_if_not")
            {}

            template <typename ExPolicy, typename InIter, typename F>
            static InIter
            sequential(ExPolicy, InIter first, InIter last, F && f)
            {
                for (; first != last; ++first)
                {
                    if (!hpx::util::invoke(f, *first))
                        return first;
                }
                return last;
            }

            template <typename ExPolicy, typename FwdIter, typename F>
            static typename util::detail::algorithm_result<
                ExPolicy, FwdIter
            >::type
            parallel(ExPolicy && policy, FwdIter first, FwdIter last, F && f)
            {
                typedef util::detail::algorithm_result<ExPolicy, FwdIter> result;
                typedef typename std::iterator_traits<Iter>::value_type type;
                typedef typename std::iterator_traits<Iter>::difference_type
                    difference_type;

                difference_type count = std::distance(first, last);
                if (count <= 0)
                    return result::get(std::move(last));

                util::cancellation_token<std::size_t> tok(count);

                return util::partitioner<ExPolicy, FwdIter, void>::
                    call_with_index(
                        std::forward<ExPolicy>(policy), first, count, 1,
                        [f, tok](FwdIter it, std::size_t part_size,
                            std::size_t base_idx) mutable -> void
                        {
                            util::loop_idx_n(
                                base_idx, it, part_size, tok,
                                [&f, &tok](type& v, std::size_t i) -> void
                            {
                                if (!hpx::util::invoke(f, v))
                                    tok.cancel(i);
                            });
                        },
                        [=](std::vector<hpx::future<void> > &&) mutable -> FwdIter
                        {
                            difference_type find_res =
                                static_cast<difference_type>(tok.get_data());

                            if (find_res != count)
                                std::advance(first, find_res);
                            else
                                first = last;

                            return std::move(first);
                        });
            }
        };

        template <typename ExPolicy, typename FwdIter, typename F>
        inline typename std::enable_if<
            execution::is_execution_policy<ExPolicy>::value,
            typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
        >::type
        find_if_not_(ExPolicy && policy, FwdIter first, FwdIter last, F && f,
            std::false_type)
        {
#if defined(HPX_HAVE_ALGORITHM_INPUT_ITERATOR_SUPPORT)
            static_assert(
                (hpx::traits::is_input_iterator<FwdIter>::value),
                "Requires at least input iterator.");
            typedef std::integral_constant<bool,
                    execution::is_sequenced_execution_policy<ExPolicy>::value ||
                   !hpx::traits::is_forward_iterator<FwdIter>::value
                > is_seq;
#else
            static_assert(
                (hpx::traits::is_forward_iterator<FwdIter>::value),
                "Requires at least forward iterator.");

            typedef execution::is_sequenced_execution_policy<ExPolicy> is_seq;
#endif
            return detail::find_if_not<FwdIter>().call(
                std::forward<ExPolicy>(policy), is_seq(),
                first, last, std::forward<F>(f));
        }

        template <typename ExPolicy, typename FwdIter, typename F>
        inline typename std::enable_if<
            execution::is_execution_policy<ExPolicy>::value,
            typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
        >::type
        find_if_not_(ExPolicy && policy, FwdIter first, FwdIter last, F && f,
            std::true_type);
        /// \endcond
    }

    /// Returns the first element in the range [first, last) for which
    /// predicate \a f returns false
    ///
    /// \note   Complexity: At most last - first
    ///         applications of the predicate.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter     The type of the source iterators used for the
    ///                     first range (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     forward iterator.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a equal requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     of the first range the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements of
    ///                     the first range the algorithm will be applied to.
    /// \param f            The unary predicate which returns false for the
    ///                     required element. The signature of the predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     bool pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such
    ///                     that objects of type \a FwdIter can
    ///                     be dereferenced and then implicitly converted to
    ///                     \a Type.
    ///
    /// The comparison operations in the parallel \a find_if_not algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The comparison operations in the parallel \a find_if_not algorithm invoked
    /// with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a find_if_not algorithm returns a \a hpx::future<FwdIter> if the
    ///           execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a FwdIter otherwise.
    ///           The \a find_if_not algorithm returns the first element in the range
    ///           [first, last) that does \b not satisfy the predicate \a f.
    ///           If no such element exists that does not satisfy the predicate f, the
    ///           algorithm returns \a last.
    ///
    template <typename ExPolicy, typename FwdIter, typename F>
    inline typename std::enable_if<
        execution::is_execution_policy<ExPolicy>::value,
        typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
    >::type
    find_if_not(ExPolicy && policy, FwdIter first, FwdIter last, F && f)
    {
        typedef hpx::traits::is_segmented_iterator<FwdIter> is_segmented;

        return detail::find_if_not_(std::forward<ExPolicy>(policy), first, last,
            std::forward<F>(f), is_segmented());
    }

    ///////////////////////////////////////////////////////////////////////////
    // find_end
    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename FwdIter>
        struct find_end : public detail::algorithm<find_end<FwdIter>, FwdIter>
        {
            find_end()
              : find_end::algorithm("find_end")
            {}

            template <typename ExPolicy, typename InIter1, typename InIter2,
                typename Pred>
            static InIter1
            sequential(ExPolicy, InIter1 first1, InIter1 last1,
                InIter2 first2, InIter2 last2, Pred && op)
            {
                return std::find_end(first1, last1, first2, last2, op);
            }

            template <typename ExPolicy, typename FwdIter2, typename Pred>
            static typename util::detail::algorithm_result<
                ExPolicy, FwdIter
            >::type
            parallel(ExPolicy && policy, FwdIter first1, FwdIter last1,
                FwdIter2 first2, FwdIter2 last2, Pred && op)
            {
                typedef util::detail::algorithm_result<ExPolicy, FwdIter> result;
                typedef typename std::iterator_traits<FwdIter>::reference reference;
                typedef typename std::iterator_traits<FwdIter>::difference_type
                    difference_type;

                difference_type diff = std::distance(first2, last2);
                if (diff <= 0)
                    return result::get(std::move(last1));

                difference_type count = std::distance(first1, last1);
                if (diff > count)
                    return result::get(std::move(last1));

                util::cancellation_token<
                    difference_type, std::greater<difference_type>
                > tok(-1);

                return util::partitioner<ExPolicy, FwdIter, void>::
                    call_with_index(
                        std::forward<ExPolicy>(policy), first1, count-(diff-1), 1,
                        [=](FwdIter it, std::size_t part_size,
                            std::size_t base_idx) mutable -> void
                        {
                            FwdIter curr = it;

                            util::loop_idx_n(
                                base_idx, it, part_size, tok,
                                [=, &tok, &curr](
                                    reference t, std::size_t i
                                ) -> void
                                {
                                    ++curr;
                                    if (hpx::util::invoke(op, t, *first2))
                                    {
                                        difference_type local_count = 1;
                                        FwdIter2 needle = first2;
                                        FwdIter mid = curr;

                                        for (difference_type len = 0;
                                             local_count != diff && len != count;
                                             (void) ++local_count, ++len, ++mid)
                                        {
                                            if (*mid != *++needle)
                                                break;
                                        }

                                        if (local_count == diff)
                                            tok.cancel(i);
                                    }
                                });
                        },
                        [=](std::vector<hpx::future<void> > &&) mutable -> FwdIter
                        {
                            difference_type find_end_res = tok.get_data();

                            if (find_end_res != count)
                                std::advance(first1, find_end_res);
                            else
                                first1 = last1;

                            return std::move(first1);
                        });
            }
        };
        /// \endcond
    }

    /// Returns the last subsequence of elements [first2, last2) found in the range
    /// [first, last) using the given predicate \a f to compare elements.
    ///
    /// \note   Complexity: at most S*(N-S+1) comparisons where
    ///         \a S = distance(first2, last2) and
    ///         \a N = distance(first1, last1).
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter1    The type of the source iterators used for the
    ///                     first range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam FwdIter2    The type of the source iterators used for the
    ///                     second range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Pred        The type of an optional function/function object to use.
    ///                     Unlike its sequential form, the parallel
    ///                     overload of \a replace requires \a Pred to meet the
    ///                     requirements of \a CopyConstructible. This defaults
    ///                     to std::equal_to<>
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first1       Refers to the beginning of the sequence of elements
    ///                     of the first range the algorithm will be applied to.
    /// \param last1        Refers to the end of the sequence of elements of
    ///                     the first range the algorithm will be applied to.
    /// \param first2       Refers to the beginning of the sequence of elements
    ///                     the algorithm will be searching for.
    /// \param last2        Refers to the end of the sequence of elements of
    ///                     the algorithm will be searching for.
    /// \param op           The binary predicate which returns \a true
    ///                     if the elements should be treated as equal. The signature
    ///                     should be equivalent to the following:
    ///                     \code
    ///                     bool pred(const Type1 &a, const Type2 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The types \a Type1 and \a Type2 must be such
    ///                     that objects of types \a FwdIter1 and \a FwdIter2
    ///                     can be dereferenced and then implicitly converted
    ///                     to \a Type1 and \a Type2 respectively.
    ///
    /// The comparison operations in the parallel \a find_end algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The comparison operations in the parallel \a find_end algorithm invoked
    /// with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a find_end algorithm returns a \a hpx::future<FwdIter> if the
    ///           execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a FwdIter otherwise.
    ///           The \a find_end algorithm returns an iterator to the beginning of
    ///           the last subsequence [first2, last2) in range [first, last).
    ///           If the length of the subsequence [first2, last2) is greater
    ///           than the length of the range [first1, last1), \a last1 is returned.
    ///           Additionally if the size of the subsequence is empty or no subsequence
    ///           is found, \a last1 is also returned.
    ///
    /// This overload of \a find_end is available if the user decides to provide the
    /// algorithm their own predicate \a f.
    ///
    template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
        typename Pred = detail::equal_to>
    inline typename std::enable_if<
        execution::is_execution_policy<ExPolicy>::value,
        typename util::detail::algorithm_result<ExPolicy, FwdIter1>::type
    >::type
    find_end(ExPolicy && policy, FwdIter1 first1, FwdIter1 last1,
        FwdIter2 first2, FwdIter2 last2, Pred && op = Pred())
    {
        static_assert(
            (hpx::traits::is_forward_iterator<FwdIter1>::value),
            "Requires at least forward iterator.");
        static_assert(
            (hpx::traits::is_forward_iterator<FwdIter2>::value),
            "Requires at least forward iterator.");

        typedef execution::is_sequenced_execution_policy<ExPolicy> is_seq;

        return detail::find_end<FwdIter1>().call(
            std::forward<ExPolicy>(policy), is_seq(),
            first1, last1, first2, last2, std::forward<Pred>(op));
    }

    ///////////////////////////////////////////////////////////////////////////
    // find_first_of
    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename FwdIter>
        struct find_first_of
          : public detail::algorithm<find_first_of<FwdIter>, FwdIter>
        {
            find_first_of()
              : find_first_of::algorithm("find_first_of")
            {}

            template <typename ExPolicy, typename InIter1, typename InIter2,
                typename Pred>
            static InIter1
            sequential(ExPolicy, InIter1 first, InIter1 last, InIter2 s_first,
                InIter2 s_last, Pred && op)
            {
                if (first == last)
                    return last;

                for (/* */; first != last; ++first)
                {
                    for (InIter2 iter = s_first; iter != s_last; ++iter)
                    {
                        if (hpx::util::invoke(op, *first, *iter))
                            return first;
                    }
                }
                return last;
            }

            template <typename ExPolicy, typename FwdIter2, typename Pred>
            static typename util::detail::algorithm_result<
                ExPolicy, FwdIter
            >::type
            parallel(ExPolicy && policy, FwdIter first, FwdIter last,
                FwdIter2 s_first, FwdIter2 s_last, Pred && op)
            {
                typedef util::detail::algorithm_result<ExPolicy, FwdIter> result;
                typedef typename std::iterator_traits<FwdIter>::reference reference;
                typedef typename std::iterator_traits<FwdIter>::difference_type
                    difference_type;
                typedef typename std::iterator_traits<FwdIter2>::difference_type
                    s_difference_type;

                s_difference_type diff = std::distance(s_first, s_last);
                if(diff <= 0)
                    return result::get(std::move(last));

                difference_type count = std::distance(first, last);
                if(diff > count)
                    return result::get(std::move(last));

                util::cancellation_token<difference_type> tok(count);

                return util::partitioner<ExPolicy, FwdIter, void>::
                    call_with_index(
                        std::forward<ExPolicy>(policy), first, count, 1,
                        [s_first, s_last, tok, op](
                            FwdIter it, std::size_t part_size,
                            std::size_t base_idx
                        ) mutable -> void
                        {
                            util::loop_idx_n(
                                base_idx, it, part_size, tok,
                                [&tok, &s_first, &s_last, &op](
                                    reference v, std::size_t i
                                ) -> void
                                {
                                    for(FwdIter2 iter = s_first; iter != s_last;
                                        ++iter)
                                    {
                                        if (hpx::util::invoke(op, v, *iter))
                                            tok.cancel(i);
                                    }
                                });
                        },
                        [=](std::vector<hpx::future<void> > &&) mutable -> FwdIter
                        {
                            difference_type find_first_of_res = tok.get_data();

                            if (find_first_of_res != count)
                                std::advance(first, find_first_of_res);
                            else
                                first = last;

                            return std::move(first);
                        });
            }
        };
        /// \endcond
    }

    /// Searches the range [first, last) for any elements in the range [s_first, s_last).
    /// Uses binary predicate p to compare elements
    ///
    /// \note   Complexity: at most (S*N) comparisons where
    ///         \a S = distance(s_first, s_last) and
    ///         \a N = distance(first, last).
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter1    The type of the source iterators used for the
    ///                     first range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward  iterator.
    /// \tparam FwdIter2    The type of the source iterators used for the
    ///                     second range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Pred        The type of an optional function/function object to use.
    ///                     Unlike its sequential form, the parallel
    ///                     overload of \a equal requires \a Pred to meet the
    ///                     requirements of \a CopyConstructible. This defaults
    ///                     to std::equal_to<>
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     of the first range the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements of
    ///                     the first range the algorithm will be applied to.
    /// \param s_first      Refers to the beginning of the sequence of elements
    ///                     the algorithm will be searching for.
    /// \param s_last       Refers to the end of the sequence of elements of
    ///                     the algorithm will be searching for.
    /// \param op           The binary predicate which returns \a true
    ///                     if the elements should be treated as equal. The signature
    ///                     should be equivalent to the following:
    ///                     \code
    ///                     bool pred(const Type1 &a, const Type2 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The types \a Type1 and \a Type2 must be such
    ///                     that objects of types \a FwdIter1 and \a FwdIter2
    ///                     can be dereferenced and then implicitly converted
    ///                     to \a Type1 and \a Type2 respectively.
    ///
    /// The comparison operations in the parallel \a find_first_of algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The comparison operations in the parallel \a find_first_of algorithm invoked
    /// with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a find_first_of algorithm returns a \a hpx::future<FwdIter1> if the
    ///           execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a FwdIter1 otherwise.
    ///           The \a find_first_of algorithm returns an iterator to the first element
    ///           in the range [first, last) that is equal to an element from the range
    ///           [s_first, s_last).
    ///           If the length of the subsequence [s_first, s_last) is
    ///           greater than the length of the range [first, last),
    ///           \a last is returned.
    ///           Additionally if the size of the subsequence is empty or no subsequence
    ///           is found, \a last is also returned.
    ///           This overload of \a find_end is available if
    ///           the user decides to provide the
    ///           algorithm their own predicate \a f.
    ///
    template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
        typename Pred = detail::equal_to>
    inline typename std::enable_if<
        execution::is_execution_policy<ExPolicy>::value,
        typename util::detail::algorithm_result<ExPolicy, FwdIter1>::type
    >::type
    find_first_of(ExPolicy && policy, FwdIter1 first, FwdIter1 last,
        FwdIter2 s_first, FwdIter2 s_last, Pred && op = Pred())
    {
#if defined(HPX_HAVE_ALGORITHM_INPUT_ITERATOR_SUPPORT)
        static_assert(
            (hpx::traits::is_input_iterator<FwdIter1>::value),
            "Requires at least input iterator.");

        typedef std::integral_constant<bool,
                execution::is_sequenced_execution_policy<ExPolicy>::value ||
               !hpx::traits::is_forward_iterator<FwdIter1>::value
            > is_seq;
#else
        static_assert(
            (hpx::traits::is_forward_iterator<FwdIter1>::value),
            "Requires at least forward iterator.");

        typedef execution::is_sequenced_execution_policy<ExPolicy> is_seq;
#endif

        static_assert(
            (hpx::traits::is_forward_iterator<FwdIter2>::value),
            "Subsequence requires at least forward iterator.");

        return detail::find_first_of<FwdIter1>().call(
            std::forward<ExPolicy>(policy), is_seq(),
            first, last, s_first, s_last, std::forward<Pred>(op));
    }
}}}

#endif
