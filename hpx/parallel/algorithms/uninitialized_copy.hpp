//  Copyright (c) 2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/uninitialized_copy.hpp

#if !defined(HPX_PARALLEL_DETAIL_UNINITIALIZED_COPY_OCT_02_2014_1145AM)
#define HPX_PARALLEL_DETAIL_UNINITIALIZED_COPY_OCT_02_2014_1145AM

#include <hpx/config.hpp>
#include <hpx/util/move.hpp>

#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/detail/is_negative.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/zip_iterator.hpp>
#include <hpx/parallel/util/loop.hpp>
#include <hpx/parallel/util/partitioner_with_cleanup.hpp>

#include <algorithm>
#include <iterator>

#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_base_of.hpp>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
    ///////////////////////////////////////////////////////////////////////////
    // uninitialized_copy
    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename Iter, typename FwdIter>
        FwdIter
        sequential_uninitialized_copy_n(Iter first, std::size_t count,
            FwdIter dest, util::cancellation_token<util::detail::no_data>& tok)
        {
            typedef typename std::iterator_traits<FwdIter>::value_type
                value_type;

            return
                util::loop_with_cleanup_n_with_token(
                    first, count, dest, tok,
                    [](Iter it, FwdIter dest) {
                        ::new (&*dest) value_type(*it);
                    },
                    [](FwdIter dest) {
                        (*dest).~value_type();
                    });
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename ExPolicy, typename Iter, typename FwdIter>
        typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
        parallel_sequential_uninitialized_copy_n(ExPolicy && policy,
            Iter first, std::size_t count, FwdIter dest)
        {
            if (count == 0)
            {
                return util::detail::algorithm_result<ExPolicy, FwdIter>::get(
                    std::move(dest));
            }

            typedef hpx::util::zip_iterator<Iter, FwdIter> zip_iterator;
            typedef std::pair<FwdIter, FwdIter> partition_result_type;
            typedef typename std::iterator_traits<FwdIter>::value_type
                value_type;

            util::cancellation_token<util::detail::no_data> tok;
            return util::partitioner_with_cleanup<
                    ExPolicy, FwdIter, partition_result_type
                >::call(
                    std::forward<ExPolicy>(policy),
                    hpx::util::make_zip_iterator(first, dest), count,
                    [tok](zip_iterator t, std::size_t part_size)
                        mutable -> partition_result_type
                    {
                        using hpx::util::get;
                        FwdIter const& dest = get<1>(t.get_iterator_tuple());
                        return std::make_pair(dest,
                            sequential_uninitialized_copy_n(
                                get<0>(t.get_iterator_tuple()), part_size,
                                dest, tok));
                    },
                    // finalize, called once if no error occurred
                    [dest, count](
                        std::vector<hpx::future<partition_result_type> > &&)
                            mutable -> FwdIter
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
        template <typename FwdIter>
        struct uninitialized_copy
          : public detail::algorithm<uninitialized_copy<FwdIter>, FwdIter>
        {
            uninitialized_copy()
              : uninitialized_copy::algorithm("uninitialized_copy")
            {}

            template <typename ExPolicy, typename Iter>
            static FwdIter
            sequential(ExPolicy, Iter first, Iter last, FwdIter dest)
            {
                return std::uninitialized_copy(first, last, dest);
            }

            template <typename ExPolicy, typename Iter>
            static typename util::detail::algorithm_result<
                ExPolicy, FwdIter
            >::type
            parallel(ExPolicy && policy, Iter first, Iter last,
                FwdIter dest)
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
    /// The assignments in the parallel \a uninitialized_copy algorithm invoked
    /// with an execution policy object of type \a sequential_execution_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a uninitialized_copy algorithm invoked
    /// with an execution policy object of type \a parallel_execution_policy or
    /// \a parallel_task_execution_policy are permitted to execute in an
    /// unordered fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a uninitialized_copy algorithm returns a
    ///           \a hpx::future<FwdIter>, if the execution policy is of type
    ///           \a sequential_task_execution_policy or
    ///           \a parallel_task_execution_policy and
    ///           returns \a FwdIter otherwise.
    ///           The \a uninitialized_copy algorithm returns the output
    ///           iterator to the element in the destination range, one past
    ///           the last element copied.
    ///
    template <typename ExPolicy, typename InIter, typename FwdIter>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
    >::type
    uninitialized_copy(ExPolicy && policy, InIter first, InIter last,
        FwdIter dest)
    {
        typedef typename std::iterator_traits<InIter>::iterator_category
            input_iterator_category;
        typedef typename std::iterator_traits<FwdIter>::iterator_category
            forward_iterator_category;

        static_assert(
            (boost::is_base_of<
                std::input_iterator_tag, input_iterator_category>::value),
            "Requires at least an input iterator.");

        static_assert(
            (boost::is_base_of<
                std::forward_iterator_tag, forward_iterator_category>::value),
            "Requires at least a forward iterator.");

        typedef typename boost::mpl::or_<
            is_sequential_execution_policy<ExPolicy>,
            boost::is_same<std::input_iterator_tag, input_iterator_category>
        >::type is_seq;

        return detail::uninitialized_copy<FwdIter>().call(
            std::forward<ExPolicy>(policy), is_seq(),
            first, last, dest);
    }

    /////////////////////////////////////////////////////////////////////////////
    // uninitialized_copy_n
    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename FwdIter>
        struct uninitialized_copy_n
          : public detail::algorithm<uninitialized_copy_n<FwdIter>, FwdIter>
        {
            uninitialized_copy_n()
              : uninitialized_copy_n::algorithm("uninitialized_copy_n")
            {}

            template <typename ExPolicy, typename Iter>
            static FwdIter
            sequential(ExPolicy, Iter first, std::size_t count,
                FwdIter dest)
            {
                return std::uninitialized_copy_n(first, count, dest);
            }

            template <typename ExPolicy, typename Iter>
            static typename util::detail::algorithm_result<
                ExPolicy, FwdIter
            >::type
            parallel(ExPolicy && policy, Iter first, std::size_t count,
                FwdIter dest)
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
    /// The assignments in the parallel \a uninitialized_copy_n algorithm
    /// invoked with an execution policy object of type
    /// \a sequential_execution_policy execute in sequential order in the
    /// calling thread.
    ///
    /// The assignments in the parallel \a uninitialized_copy_n algorithm
    /// invoked with an execution policy object of type
    /// \a parallel_execution_policy or
    /// \a parallel_task_execution_policy are permitted to execute in an
    /// unordered fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a uninitialized_copy_n algorithm returns a
    ///           \a hpx::future<FwdIter> if the execution policy is of type
    ///           \a sequential_task_execution_policy or
    ///           \a parallel_task_execution_policy and
    ///           returns \a FwdIter otherwise.
    ///           The \a uninitialized_copy_n algorithm returns the output
    ///           iterator to the element in the destination range, one past
    ///           the last element copied.
    ///
    template <typename ExPolicy, typename InIter, typename Size, typename FwdIter>
    typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
    >::type
    uninitialized_copy_n(ExPolicy && policy, InIter first, Size count,
        FwdIter dest)
    {
        typedef typename std::iterator_traits<InIter>::iterator_category
            input_iterator_category;
        typedef typename std::iterator_traits<FwdIter>::iterator_category
            forward_iterator_category;

        static_assert(
            (boost::is_base_of<
                std::input_iterator_tag, input_iterator_category>::value),
            "Requires at least an input iterator.");

        static_assert(
            (boost::is_base_of<
                std::forward_iterator_tag, forward_iterator_category>::value),
            "Requires at least a forward iterator.");

        // if count is representing a negative value, we do nothing
        if (detail::is_negative<Size>::call(count))
        {
            return util::detail::algorithm_result<ExPolicy, FwdIter>::get(
                std::move(dest));
        }

        typedef typename boost::mpl::or_<
            is_sequential_execution_policy<ExPolicy>,
            boost::is_same<std::input_iterator_tag, input_iterator_category>
        >::type is_seq;

        return detail::uninitialized_copy_n<FwdIter>().call(
            std::forward<ExPolicy>(policy), is_seq(),
            first, std::size_t(count), dest);
    }
}}}

#endif
