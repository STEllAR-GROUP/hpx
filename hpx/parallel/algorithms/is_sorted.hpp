//  Copyright (c) 2015 Daniel Bourgeois
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/is_sorted.hpp

#if !defined(HPX_PARALLEL_ALGORITHMS_IS_SORTED_FEB_9_2015_0331PM)
#define HPX_PARALLEL_ALGORITHMS_IS_SORTED_FEB_9_2015_0331PM

#include <hpx/config.hpp>
#include <hpx/traits/is_iterator.hpp>
#include <hpx/util/invoke.hpp>

#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/util/cancellation_token.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/loop.hpp>
#include <hpx/parallel/util/partitioner.hpp>
#include <hpx/util/unused.hpp>

#include <boost/range/functions.hpp>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace parallel { inline namespace v1
{
    ////////////////////////////////////////////////////////////////////////////
    // is_sorted
    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename FwdIter>
        struct is_sorted: public detail::algorithm<is_sorted<FwdIter>, bool>
        {
            is_sorted()
              : is_sorted::algorithm("is_sorted")
            {}

            template<typename ExPolicy, typename Pred>
            static bool
            sequential(ExPolicy, FwdIter first, FwdIter last,
                Pred && pred)
            {
                return std::is_sorted(first, last, std::forward<Pred>(pred));
            }

            template <typename ExPolicy, typename Pred>
            static typename util::detail::algorithm_result<ExPolicy, bool>::type
            parallel(ExPolicy && policy, FwdIter first, FwdIter last, Pred && pred)
            {
                typedef typename std::iterator_traits<FwdIter>::difference_type
                    difference_type;
                typedef typename util::detail::algorithm_result<ExPolicy, bool>
                    result;

                difference_type count = std::distance(first, last);
                if (count <= 1)
                    return result::get(true);

                util::cancellation_token<> tok;
                auto f1 =
                    [tok, pred, last, policy](
                        FwdIter part_begin, std::size_t part_size
                    ) mutable -> bool
                    {
                        HPX_UNUSED(policy);

                        FwdIter trail = part_begin++;
                        util::loop_n<ExPolicy>(
                            part_begin, part_size - 1,
                            [&trail, &tok, &pred](FwdIter it)
                            {
                                if (hpx::util::invoke(pred, *it, *trail++))
                                {
                                    tok.cancel();
                                }
                            });

                        FwdIter i = trail++;
                        // trail now points one past the current grouping
                        // unless canceled

                        if (!tok.was_cancelled() && trail != last)
                        {
                            return !pred(*trail, *i);
                        }
                        return !tok.was_cancelled();
                    };

                return util::partitioner<ExPolicy, bool>::call(
                    std::forward<ExPolicy>(policy), first, count,
                    std::move(f1),
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

    /// Determines if the range [first, last) is sorted. Uses pred to
    /// compare elements.
    ///
    /// \note   Complexity: at most (N+S-1) comparisons where
    ///         \a N = distance(first, last).
    ///         \a S = number of partitions
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter     The type of the source iterators used for the
    ///                     This iterator type must meet the requirements of a
    ///                     forward iterator.
    /// \tparam Pred        The type of an optional function/function object to use.
    ///                     Unlike its sequential form, the parallel
    ///                     overload of \a is_sorted requires \a Pred to meet the
    ///                     requirements of \a CopyConstructible. This defaults
    ///                     to std::less<>
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     of that the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements of
    ///                     that the algorithm will be applied to.
    /// \param pred         Refers to the binary predicate which returns true
    ///                     if the first argument should be treated as less than
    ///                     the second argument. The signature of the function
    ///                     should be equivalent to
    ///                     \code
    ///                     bool pred(const Type &a, const Type &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that objects of
    ///                     types \a FwdIter can be dereferenced and then
    ///                     implicitly converted to Type.
    ///
    /// The comparison operations in the parallel \a is_sorted algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// executes in sequential order in the calling thread.
    ///
    /// The comparison operations in the parallel \a is_sorted algorithm invoked
    /// with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a is_sorted algorithm returns a \a hpx::future<bool>
    ///           if the execution policy is of type \a task_execution_policy
    ///           and returns \a bool otherwise.
    ///           The \a is_sorted algorithm returns a bool if each element in
    ///           the sequence [first, last) satisfies the predicate passed.
    ///           If the range [first, last) contains less than two elements,
    ///           the function always returns true.
    ///
    template <typename ExPolicy, typename FwdIter, typename Pred = detail::less>
    inline typename std::enable_if<
        execution::is_execution_policy<ExPolicy>::value,
        typename util::detail::algorithm_result<ExPolicy, bool>::type
    >::type
    is_sorted(ExPolicy && policy, FwdIter first, FwdIter last,
        Pred && pred = Pred())
    {
        static_assert(
            (hpx::traits::is_forward_iterator<FwdIter>::value),
            "Requires at least forward iterator.");

        typedef execution::is_sequential_execution_policy<ExPolicy> is_seq;

        return detail::is_sorted<FwdIter>().call(
            std::forward<ExPolicy>(policy), is_seq(), first, last,
            std::forward<Pred>(pred));
    }

    ////////////////////////////////////////////////////////////////////////////
    // is_sorted_until
    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename FwdIter>
        struct is_sorted_until:
            public detail::algorithm<is_sorted_until<FwdIter>, FwdIter>
        {
            is_sorted_until()
              : is_sorted_until::algorithm("is_sorted_until")
            {}

            template<typename ExPolicy, typename Pred>
            static FwdIter
            sequential(ExPolicy, FwdIter first, FwdIter last,
                Pred && pred)
            {
                return std::is_sorted_until(first, last,
                    std::forward<Pred>(pred));
            }

            template <typename ExPolicy, typename Pred>
            static typename util::detail::algorithm_result<
                ExPolicy, FwdIter
            >::type
            parallel(ExPolicy && policy, FwdIter first, FwdIter last, Pred && pred)
            {
                typedef typename std::iterator_traits<FwdIter>::reference
                    reference;
                typedef typename std::iterator_traits<FwdIter>::difference_type
                        difference_type;
                typedef typename util::detail::algorithm_result<
                        ExPolicy, FwdIter
                    > result;

                difference_type count = std::distance(first, last);
                if (count <= 1)
                    return result::get(std::move(last));


                util::cancellation_token<difference_type> tok(count);
                return util::partitioner<ExPolicy, FwdIter, void>::
                call_with_index(
                    std::forward<ExPolicy>(policy), first, count, 1,
                    [tok, pred, last](FwdIter part_begin, std::size_t part_size,
                        std::size_t base_idx) mutable
                    {
                        FwdIter trail = part_begin++;
                        util::loop_idx_n(++base_idx, part_begin,
                            part_size - 1, tok,
                            [&trail, &tok, &pred](reference& v, std::size_t ind)
                            {
                                if (pred(v, *trail++))
                                {
                                    tok.cancel(ind);
                                }
                            });

                        FwdIter i = trail++;

                        //trail now points one past the current grouping
                        //unless canceled
                        if (!tok.was_cancelled(base_idx + part_size)
                            && trail != last)
                        {
                            if (pred(*trail, *i))
                            {
                                tok.cancel(base_idx + part_size);
                            }
                        }
                    },
                    [first, tok](std::vector<hpx::future<void> > &&) mutable
                    ->  FwdIter
                    {
                        difference_type loc = tok.get_data();
                        std::advance(first, loc);
                        return std::move(first);
                    });
            }
        };
        /// \endcond
    }

    /// Returns the first element in the range [first, last) that is not sorted.
    /// Uses a predicate to compare elements or the less than operator.
    ///
    /// \note   Complexity: at most (N+S-1) comparisons where
    ///         \a N = distance(first, last).
    ///         \a S = number of partitions
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter     The type of the source iterators used for the
    ///                     This iterator type must meet the requirements of a
    ///                     forward iterator.
    /// \tparam Pred        The type of an optional function/function object to use.
    ///                     Unlike its sequential form, the parallel
    ///                     overload of \a is_sorted_until requires \a Pred to meet
    ///                     the requirements of \a CopyConstructible. This defaults
    ///                     to std::less<>
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     of that the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements of
    ///                     that the algorithm will be applied to.
    /// \param pred         Refers to the binary predicate which returns true
    ///                     if the first argument should be treated as less than
    ///                     the second argument. The signature of the function
    ///                     should be equivalent to
    ///                     \code
    ///                     bool pred(const Type &a, const Type &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that objects of
    ///                     types \a FwdIter can be dereferenced and then
    ///                     implicitly converted to Type.
    ///
    /// The comparison operations in the parallel \a is_sorted_until algorithm
    /// invoked with an execution policy object of type
    /// \a sequenced_policy executes in sequential order in the
    /// calling thread.
    ///
    /// The comparison operations in the parallel \a is_sorted_until algorithm
    /// invoked with an execution policy object of type
    /// \a parallel_policy or \a parallel_task_policy are
    /// permitted to execute in an unordered fashion in unspecified threads,
    /// and indeterminately sequenced within each thread.
    ///
    /// \returns  The \a is_sorted_until algorithm returns a \a hpx::future<FwdIter>
    ///           if the execution policy is of type \a task_execution_policy
    ///           and returns \a FwdIter otherwise.
    ///           The \a is_sorted_until algorithm returns the first unsorted
    ///           element. If the sequence has less than two elements or the
    ///           sequence is sorted, last is returned.
    ///
    template <typename ExPolicy, typename FwdIter, typename Pred = detail::less>
    inline typename std::enable_if<
        execution::is_execution_policy<ExPolicy>::value,
        typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
    >::type
    is_sorted_until(ExPolicy && policy, FwdIter first, FwdIter last,
        Pred && pred = Pred())
    {
        static_assert(
            (hpx::traits::is_forward_iterator<FwdIter>::value),
            "Requires at least forward iterator.");

        typedef execution::is_sequential_execution_policy<ExPolicy> is_seq;

        return detail::is_sorted_until<FwdIter>().call(
            std::forward<ExPolicy>(policy), is_seq(), first, last,
            std::forward<Pred>(pred));
    }
}}}

#endif
