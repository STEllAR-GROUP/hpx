//  Copyright (c) 2014-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/uninitialized_move.hpp

#if !defined(HPX_PARALLEL_DETAIL_UNINITIALIZED_MOVE_MAY_30_2017_0843AM)
#define HPX_PARALLEL_DETAIL_UNINITIALIZED_MOVE_MAY_30_2017_0843AM

#include <hpx/config.hpp>
#include <hpx/traits/concepts.hpp>
#include <hpx/traits/is_iterator.hpp>
#include <hpx/util/tagged_pair.hpp>

#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/detail/is_negative.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/tagspec.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/loop.hpp>
#include <hpx/parallel/util/partitioner_with_cleanup.hpp>
#include <hpx/parallel/util/zip_iterator.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace parallel { inline namespace v1
{
    ///////////////////////////////////////////////////////////////////////////
    // uninitialized_move
    namespace detail
    {
        /// \cond NOINTERNAL

        // provide our own implementation of std::uninitialized_move as some
        // versions of MSVC horribly fail at compiling it for some types T
        template <typename InIter, typename FwdIter>
        FwdIter std_uninitialized_move(InIter first, InIter last, FwdIter d_first)
        {
            typedef typename std::iterator_traits<FwdIter>::value_type
                value_type;

            FwdIter current = d_first;
            try {
                for (/* */; first != last; ++first, (void) ++current) {
                    ::new (static_cast<void*>(std::addressof(*current)))
                        value_type(std::move(*first));
                }
                return current;
            }
            catch (...) {
                for (/* */; d_first != current; ++d_first) {
                    (*d_first).~value_type();
                }
                throw;
            }
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename Iter, typename FwdIter>
        FwdIter sequential_uninitialized_move_n(Iter first, std::size_t count,
            FwdIter dest, util::cancellation_token<util::detail::no_data>& tok)
        {
            typedef typename std::iterator_traits<FwdIter>::value_type
                value_type;

            return
                util::loop_with_cleanup_n_with_token(
                    first, count, dest, tok,
                    [](Iter it, FwdIter dest) {
                        ::new (static_cast<void*>(std::addressof(*dest)))
                            value_type(std::move(*it));
                    },
                    [](FwdIter dest) {
                        (*dest).~value_type();
                    });
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename ExPolicy, typename Iter, typename FwdIter>
        typename util::detail::algorithm_result<
            ExPolicy, std::pair<Iter, FwdIter>
        >::type
        parallel_sequential_uninitialized_move_n(ExPolicy && policy,
            Iter first, std::size_t count, FwdIter dest)
        {
            if (count == 0)
            {
                return util::detail::algorithm_result<
                        ExPolicy, std::pair<Iter, FwdIter>
                    >::get(std::make_pair(first, dest));
            }

            typedef hpx::util::zip_iterator<Iter, FwdIter> zip_iterator;
            typedef std::pair<FwdIter, FwdIter> partition_result_type;
            typedef typename std::iterator_traits<FwdIter>::value_type
                value_type;

            util::cancellation_token<util::detail::no_data> tok;
            return util::partitioner_with_cleanup<
                    ExPolicy, std::pair<Iter, FwdIter>, partition_result_type
                >::call(
                    std::forward<ExPolicy>(policy),
                    hpx::util::make_zip_iterator(first, dest), count,
                    [tok](zip_iterator t, std::size_t part_size)
                        mutable -> partition_result_type
                    {
                        using hpx::util::get;
                        auto iters = t.get_iterator_tuple();
                        FwdIter dest = get<1>(iters);
                        return std::make_pair(dest,
                            sequential_uninitialized_move_n(
                                get<0>(iters), part_size,
                                dest, tok));
                    },
                    // finalize, called once if no error occurred
                    [first, dest, count](
                        std::vector<hpx::future<partition_result_type> > &&)
                            mutable -> std::pair<Iter, FwdIter>
                    {
                        std::advance(first, count);
                        std::advance(dest, count);
                        return std::make_pair(first, dest);
                    },
                    // cleanup function, called for each partition which
                    // didn't fail, but only if at least one failed
                    [](partition_result_type && r) -> void
                    {
                        while (r.first != r.second)
                        {
                            (*r.first).~value_type();
                            ++r.first;
                        }
                    });
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename FwdIter>
        struct uninitialized_move
          : public detail::algorithm<uninitialized_move<FwdIter>, FwdIter>
        {
            uninitialized_move()
              : uninitialized_move::algorithm("uninitialized_move")
            {}

            template <typename ExPolicy, typename Iter>
            static FwdIter
            sequential(ExPolicy, Iter first, Iter last, FwdIter dest)
            {
                return std_uninitialized_move(first, last, dest);
            }

            template <typename ExPolicy, typename Iter>
            static typename util::detail::algorithm_result<
                ExPolicy, FwdIter
            >::type
            parallel(ExPolicy && policy, Iter first, Iter last, FwdIter dest)
            {
                return util::detail::convert_to_result(
                    parallel_sequential_uninitialized_move_n(
                        std::forward<ExPolicy>(policy), first,
                        std::distance(first, last), dest
                    ),
                    [](std::pair<Iter, FwdIter> const& p) -> FwdIter
                    {
                        return p.second;
                    });
            }
        };
        /// \endcond
    }

    /// Moves the elements in the range, defined by [first, last), to an
    /// uninitialized memory area beginning at \a dest. If an exception is
    /// thrown during the initialization, some objects in [first, last) are
    /// left in a valid but unspecified state.
    ///
    /// \note   Complexity: Performs exactly \a last - \a first move operations.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam InIter      The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
    /// \tparam FwdIter     The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     forward iterator.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
    ///
    /// The assignments in the parallel \a uninitialized_move algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a uninitialized_move algorithm invoked
    /// with an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an
    /// unordered fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a uninitialized_move algorithm returns a
    ///           \a hpx::future<FwdIter>, if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a FwdIter otherwise.
    ///           The \a uninitialized_move algorithm returns the output
    ///           iterator to the element in the destination range, one past
    ///           the last element moved.
    ///
    template <typename ExPolicy, typename InIter, typename FwdIter,
    HPX_CONCEPT_REQUIRES_(
        execution::is_execution_policy<ExPolicy>::value &&
        hpx::traits::is_iterator<InIter>::value &&
        hpx::traits::is_iterator<FwdIter>::value)>
    typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
    uninitialized_move(ExPolicy && policy, InIter first, InIter last,
        FwdIter dest)
    {
        static_assert(
            (hpx::traits::is_input_iterator<InIter>::value),
            "Required at least input iterator.");
        static_assert(
            (hpx::traits::is_forward_iterator<FwdIter>::value),
            "Requires at least output iterator.");

        typedef std::integral_constant<bool,
                execution::is_sequenced_execution_policy<ExPolicy>::value ||
               !hpx::traits::is_forward_iterator<InIter>::value
            > is_seq;

        return detail::uninitialized_move<FwdIter>().call(
            std::forward<ExPolicy>(policy), is_seq(),
            first, last, dest);
    }

    /////////////////////////////////////////////////////////////////////////////
    // uninitialized_move_n
    namespace detail
    {
        /// \cond NOINTERNAL

        // provide our own implementation of std::uninitialized_move_n as some
        // versions of MSVC horribly fail at compiling it for some types T
        template <typename InIter, typename FwdIter>
        std::pair<InIter, FwdIter> std_uninitialized_move_n(InIter first,
            std::size_t count, FwdIter d_first)
        {
            typedef typename std::iterator_traits<FwdIter>::value_type
                value_type;

            FwdIter current = d_first;
            try {
                for (/* */; count != 0; ++first, (void) ++current, --count) {
                    ::new (static_cast<void*>(std::addressof(*current)))
                        value_type(std::move(*first));
                }
                return std::make_pair(first, current);
            }
            catch (...) {
                for (/* */; d_first != current; ++d_first) {
                    (*d_first).~value_type();
                }
                throw;
            }
        }

        template <typename IterPair>
        struct uninitialized_move_n
          : public detail::algorithm<uninitialized_move_n<IterPair>, IterPair>
        {
            uninitialized_move_n()
              : uninitialized_move_n::algorithm("uninitialized_move_n")
            {}

            template <typename ExPolicy, typename Iter, typename FwdIter>
            static IterPair
            sequential(ExPolicy, Iter first, std::size_t count, FwdIter dest)
            {
                return std_uninitialized_move_n(first, count, dest);
            }

            template <typename ExPolicy, typename Iter, typename FwdIter>
            static typename util::detail::algorithm_result<
                ExPolicy, IterPair
            >::type
            parallel(ExPolicy && policy, Iter first, std::size_t count,
                FwdIter dest)
            {
                return parallel_sequential_uninitialized_move_n(
                    std::forward<ExPolicy>(policy), first, count, dest);
            }
        };
        /// \endcond
    }

    /// Moves the elements in the range [first, first + count), starting from
    /// first and proceeding to first + count - 1., to another range beginning
    /// at dest. If an exception is
    /// thrown during the initialization, some objects in [first, first + count)
    /// are left in a valid but unspecified state.
    ///
    /// \note   Complexity: Performs exactly \a count movements, if
    ///         count > 0, no move operations otherwise.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam InIter      The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
    /// \tparam Size        The type of the argument specifying the number of
    ///                     elements to apply \a f to.
    /// \tparam FwdIter     The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     forward iterator.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param count        Refers to the number of elements starting at
    ///                     \a first the algorithm will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
    ///
    /// The assignments in the parallel \a uninitialized_move_n algorithm
    /// invoked with an execution policy object of type
    /// \a sequenced_policy execute in sequential order in the
    /// calling thread.
    ///
    /// The assignments in the parallel \a uninitialized_move_n algorithm
    /// invoked with an execution policy object of type
    /// \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an
    /// unordered fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a uninitialized_move_n algorithm returns a
    ///           \a hpx::future<std::pair<InIter, FwdIter>> if the execution
    ///           policy is of type \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a std::pair<InIter, FwdIter> otherwise.
    ///           The \a uninitialized_move_n algorithm returns the pair of
    ///           the input iterator to the element past in the source range
    ///           and an output iterator to the element in the destination
    ///           range, one past the last element moved.
    ///
    template <typename ExPolicy, typename InIter, typename Size, typename FwdIter,
    HPX_CONCEPT_REQUIRES_(
        execution::is_execution_policy<ExPolicy>::value &&
        hpx::traits::is_iterator<InIter>::value &&
        hpx::traits::is_iterator<FwdIter>::value)>
    typename util::detail::algorithm_result<
        ExPolicy, hpx::util::tagged_pair<tag::in(InIter), tag::out(FwdIter)>
    >::type
    uninitialized_move_n(ExPolicy && policy, InIter first, Size count,
        FwdIter dest)
    {
        static_assert(
            (hpx::traits::is_input_iterator<InIter>::value),
            "Required at least input iterator.");
        static_assert(
            (hpx::traits::is_forward_iterator<FwdIter>::value),
            "Requires at least forward iterator.");

        // if count is representing a negative value, we do nothing
        if (detail::is_negative(count))
        {
            return hpx::util::make_tagged_pair<tag::in, tag::out>(
                    util::detail::algorithm_result<
                            ExPolicy, std::pair<InIter, FwdIter>
                        >::get(std::make_pair(first, dest)));
        }

        typedef std::integral_constant<bool,
                execution::is_sequenced_execution_policy<ExPolicy>::value ||
               !hpx::traits::is_forward_iterator<InIter>::value
            > is_seq;

        return hpx::util::make_tagged_pair<tag::in, tag::out>(
            detail::uninitialized_move_n<std::pair<InIter, FwdIter> >().call(
                std::forward<ExPolicy>(policy), is_seq(),
                first, std::size_t(count), dest));
    }
}}}

#endif
