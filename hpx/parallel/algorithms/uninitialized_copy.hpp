//  Copyright (c) 2014-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/uninitialized_copy.hpp

#if !defined(HPX_PARALLEL_DETAIL_UNINITIALIZED_COPY_OCT_02_2014_1145AM)
#define HPX_PARALLEL_DETAIL_UNINITIALIZED_COPY_OCT_02_2014_1145AM

#include <hpx/config.hpp>
#include <hpx/traits/is_iterator.hpp>

#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/detail/is_negative.hpp>
#include <hpx/parallel/execution_policy.hpp>
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
    // uninitialized_copy
    namespace detail
    {
        /// \cond NOINTERNAL

        // provide our own implementation of std::uninitialized_copy as some
        // versions of MSVC horribly fail at compiling it for some types T
        template <typename FwdIter1, typename FwdIter2>
        FwdIter2 std_uninitialized_copy(FwdIter1 first, FwdIter1 last, FwdIter2 d_first)
        {
            typedef typename std::iterator_traits<FwdIter2>::value_type
                value_type;

            FwdIter2 current = d_first;
            try {
                for (/* */; first != last; ++first, (void) ++current) {
                    ::new (static_cast<void*>(std::addressof(*current)))
                        value_type(*first);
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
        template <typename Iter, typename FwdIter2>
        FwdIter2
        sequential_uninitialized_copy_n(Iter first, std::size_t count,
            FwdIter2 dest, util::cancellation_token<util::detail::no_data>& tok)
        {
            typedef typename std::iterator_traits<FwdIter2>::value_type
                value_type;

            return
                util::loop_with_cleanup_n_with_token(
                    first, count, dest, tok,
                    [](Iter it, FwdIter2 dest) {
                        ::new (static_cast<void*>(std::addressof(*dest)))
                            value_type(*it);
                    },
                    [](FwdIter2 dest) {
                        (*dest).~value_type();
                    });
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename ExPolicy, typename Iter, typename FwdIter2>
        typename util::detail::algorithm_result<ExPolicy, FwdIter2>::type
        parallel_sequential_uninitialized_copy_n(ExPolicy && policy,
            Iter first, std::size_t count, FwdIter2 dest)
        {
            if (count == 0)
            {
                return util::detail::algorithm_result<ExPolicy, FwdIter2>::get(
                    std::move(dest));
            }

            typedef hpx::util::zip_iterator<Iter, FwdIter2> zip_iterator;
            typedef std::pair<FwdIter2, FwdIter2> partition_result_type;
            typedef typename std::iterator_traits<FwdIter2>::value_type
                value_type;

            util::cancellation_token<util::detail::no_data> tok;
            return util::partitioner_with_cleanup<
                    ExPolicy, FwdIter2, partition_result_type
                >::call(
                    std::forward<ExPolicy>(policy),
                    hpx::util::make_zip_iterator(first, dest), count,
                    [tok](zip_iterator t, std::size_t part_size)
                        mutable -> partition_result_type
                    {
                        using hpx::util::get;
                        auto iters = t.get_iterator_tuple();
                        FwdIter2 dest = get<1>(iters);
                        return std::make_pair(dest,
                            sequential_uninitialized_copy_n(
                                get<0>(iters), part_size,
                                dest, tok));
                    },
                    // finalize, called once if no error occurred
                    [dest, count](
                        std::vector<hpx::future<partition_result_type> > &&)
                            mutable -> FwdIter2
                    {
                        std::advance(dest, count);
                        return dest;
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
        template <typename FwdIter2>
        struct uninitialized_copy
          : public detail::algorithm<uninitialized_copy<FwdIter2>, FwdIter2>
        {
            uninitialized_copy()
              : uninitialized_copy::algorithm("uninitialized_copy")
            {}

            template <typename ExPolicy, typename Iter>
            static FwdIter2
            sequential(ExPolicy, Iter first, Iter last, FwdIter2 dest)
            {
                return std_uninitialized_copy(first, last, dest);
            }

            template <typename ExPolicy, typename Iter>
            static typename util::detail::algorithm_result<
                ExPolicy, FwdIter2
            >::type
            parallel(ExPolicy && policy, Iter first, Iter last,
                FwdIter2 dest)
            {
                return parallel_sequential_uninitialized_copy_n(
                    std::forward<ExPolicy>(policy), first,
                    std::distance(first, last), dest);
            }
        };
        /// \endcond
    }

    /// Copies the elements in the range, defined by [first, last), to an
    /// uninitialized memory area beginning at \a dest. If an exception is
    /// thrown during the copy operation, the function has no effects.
    ///
    /// \note   Complexity: Performs exactly \a last - \a first assignments.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter1    The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam FwdIter2    The type of the iterator representing the
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
    /// The assignments in the parallel \a uninitialized_copy algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a uninitialized_copy algorithm invoked
    /// with an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an
    /// unordered fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a uninitialized_copy algorithm returns a
    ///           \a hpx::future<FwdIter2>, if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a FwdIter2 otherwise.
    ///           The \a uninitialized_copy algorithm returns the output
    ///           iterator to the element in the destination range, one past
    ///           the last element copied.
    ///
    template <typename ExPolicy, typename FwdIter1, typename FwdIter2>
    inline typename std::enable_if<
        execution::is_execution_policy<ExPolicy>::value,
        typename util::detail::algorithm_result<ExPolicy, FwdIter2>::type
    >::type
    uninitialized_copy(ExPolicy && policy, FwdIter1 first, FwdIter1 last,
        FwdIter2 dest)
    {
#if defined(HPX_HAVE_ALGORITHM_INPUT_ITERATOR_SUPPORT)
        static_assert(
            (hpx::traits::is_input_iterator<FwdIter1>::value),
            "Required at least input iterator.");
        static_assert(
            (hpx::traits::is_output_iterator<FwdIter2>::value ||
                hpx::traits::is_forward_iterator<FwdIter2>::value),
            "Requires at least output iterator.");

        typedef std::integral_constant<bool,
                execution::is_sequenced_execution_policy<ExPolicy>::value ||
               !hpx::traits::is_forward_iterator<FwdIter1>::value
               !hpx::traits::is_forward_iterator<FwdIter2>::value
            > is_seq;
#else
        static_assert(
            (hpx::traits::is_forward_iterator<FwdIter1>::value),
            "Required at least forward iterator.");
        static_assert(
            (hpx::traits::is_forward_iterator<FwdIter2>::value),
            "Requires at least forward iterator.");

        typedef execution::is_sequenced_execution_policy<ExPolicy> is_seq;
#endif

        return detail::uninitialized_copy<FwdIter2>().call(
            std::forward<ExPolicy>(policy), is_seq(),
            first, last, dest);
    }

    /////////////////////////////////////////////////////////////////////////////
    // uninitialized_copy_n
    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename FwdIter2>
        struct uninitialized_copy_n
          : public detail::algorithm<uninitialized_copy_n<FwdIter2>, FwdIter2>
        {
            uninitialized_copy_n()
              : uninitialized_copy_n::algorithm("uninitialized_copy_n")
            {}

            template <typename ExPolicy, typename Iter>
            static FwdIter2
            sequential(ExPolicy, Iter first, std::size_t count,
                FwdIter2 dest)
            {
                return std::uninitialized_copy_n(first, count, dest);
            }

            template <typename ExPolicy, typename Iter>
            static typename util::detail::algorithm_result<
                ExPolicy, FwdIter2
            >::type
            parallel(ExPolicy && policy, Iter first, std::size_t count,
                FwdIter2 dest)
            {
                return parallel_sequential_uninitialized_copy_n(
                    std::forward<ExPolicy>(policy), first, count, dest);
            }
        };
        /// \endcond
    }

    /// Copies the elements in the range [first, first + count), starting from
    /// first and proceeding to first + count - 1., to another range beginning
    /// at dest. If an exception is thrown during the copy operation, the
    /// function has no effects.
    ///
    /// \note   Complexity: Performs exactly \a count assignments, if
    ///         count > 0, no assignments otherwise.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter1      The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
    /// \tparam Size        The type of the argument specifying the number of
    ///                     elements to apply \a f to.
    /// \tparam FwdIter2     The type of the iterator representing the
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
    /// The assignments in the parallel \a uninitialized_copy_n algorithm
    /// invoked with an execution policy object of type
    /// \a sequenced_policy execute in sequential order in the
    /// calling thread.
    ///
    /// The assignments in the parallel \a uninitialized_copy_n algorithm
    /// invoked with an execution policy object of type
    /// \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an
    /// unordered fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a uninitialized_copy_n algorithm returns a
    ///           \a hpx::future<FwdIter2> if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a FwdIter2 otherwise.
    ///           The \a uninitialized_copy_n algorithm returns the output
    ///           iterator to the element in the destination range, one past
    ///           the last element copied.
    ///
    template <typename ExPolicy, typename FwdIter1, typename Size, typename FwdIter2>
    inline typename std::enable_if<
        execution::is_execution_policy<ExPolicy>::value,
        typename util::detail::algorithm_result<ExPolicy, FwdIter2>::type
    >::type
    uninitialized_copy_n(ExPolicy && policy, FwdIter1 first, Size count,
        FwdIter2 dest)
    {
#if defined(HPX_HAVE_ALGORITHM_INPUT_ITERATOR_SUPPORT)
        static_assert(
            (hpx::traits::is_input_iterator<FwdIter1>::value),
            "Required at least input iterator.");
        static_assert(
            (hpx::traits::is_output_iterator<FwdIter2>::value ||
                hpx::traits::is_forward_iterator<FwdIter2>::value),
            "Requires at least output iterator.");

        typedef std::integral_constant<bool,
                execution::is_sequenced_execution_policy<ExPolicy>::value ||
               !hpx::traits::is_forward_iterator<FwdIter1>::value
               !hpx::traits::is_forward_iterator<FwdIter2>::value
            > is_seq;
#else
        static_assert(
            (hpx::traits::is_forward_iterator<FwdIter1>::value),
            "Required at least forward iterator.");
        static_assert(
            (hpx::traits::is_forward_iterator<FwdIter2>::value),
            "Requires at least forward iterator.");

        typedef execution::is_sequenced_execution_policy<ExPolicy> is_seq;
#endif

        // if count is representing a negative value, we do nothing
        if (detail::is_negative(count))
        {
            return util::detail::algorithm_result<ExPolicy, FwdIter2>::get(
                std::move(dest));
        }

        return detail::uninitialized_copy_n<FwdIter2>().call(
            std::forward<ExPolicy>(policy), is_seq(),
            first, std::size_t(count), dest);
    }
}}}

#endif
