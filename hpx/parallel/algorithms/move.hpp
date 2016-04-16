//  Copyright (c) 2014 Grant Mercer
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/move.hpp

#if !defined(HPX_PARALLEL_DETAIL_MOVE_JUNE_16_2014_1106AM)
#define HPX_PARALLEL_DETAIL_MOVE_JUNE_16_2014_1106AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/move.hpp>

#include <hpx/traits/segmented_iterator_traits.hpp>
#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/algorithms/for_each.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/zip_iterator.hpp>
#include <hpx/parallel/algorithms/detail/transfer.hpp>

#include <algorithm>
#include <iterator>

#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_base_of.hpp>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
    ///////////////////////////////////////////////////////////////////////////
    // move
    namespace detail
    {
        /// \cond NOINTERNAL

        // sequential move
        template <typename InIter, typename OutIter>
        inline std::pair<InIter, OutIter>
        sequential_move(InIter first, InIter last, OutIter dest)
        {
            while (first != last)
            {
                *dest++ = std::move(*first++);
            }
            return std::make_pair(first, dest);
        }

        template <typename IterPair>
        struct move_pair :
            public detail::algorithm<detail::move_pair<IterPair>, IterPair>
        {
            move_pair()
              : move_pair::algorithm("move")
            {}

            template <typename ExPolicy, typename InIter, typename OutIter>
            static std::pair<InIter, OutIter>
            sequential(ExPolicy, InIter first, InIter last, OutIter dest)
            {
                return sequential_move(first, last, dest);
            }

            template <typename ExPolicy, typename FwdIter, typename OutIter>
            static typename util::detail::algorithm_result<
                ExPolicy, std::pair<FwdIter, OutIter>
            >::type
            parallel(ExPolicy policy, FwdIter first, FwdIter last,
                OutIter dest)
            {
                typedef hpx::util::zip_iterator<FwdIter, OutIter> zip_iterator;
                typedef typename zip_iterator::reference reference;
                typedef typename util::detail::algorithm_result<
                        ExPolicy, std::pair<FwdIter, OutIter>
                    >::type result_type;

                return get_iter_pair(
                    for_each_n<zip_iterator>().call(
                        policy, boost::mpl::false_(),
                        hpx::util::make_zip_iterator(first, dest),
                        std::distance(first, last),
                        [](reference t) {
                            using hpx::util::get;
                            get<1>(t) = std::move(get<0>(t)); //-V573
                        }));
            }
        };

        template<typename InIter, typename OutIter, typename Enable = void>
        struct move;

        template <typename InIter, typename OutIter>
        struct move<
            InIter, OutIter,
            typename boost::enable_if<typename hpx::traits::segmented_iterator_traits<InIter>::is_segmented_iterator>::type
        > : public move_pair<std::pair<
            typename hpx::traits::segmented_iterator_traits<InIter>::local_iterator,
            typename hpx::traits::segmented_iterator_traits<OutIter>::local_iterator
        >>
        {};

        template<typename InIter, typename OutIter>
        struct move<
            InIter, OutIter,
            typename boost::disable_if<typename hpx::traits::segmented_iterator_traits<OutIter>::is_segmented_iterator>::type
        > : public move_pair<std::pair<InIter, OutIter>>
        {};
        /// \endcond
    }

    /// Moves the elements in the range [first, last), to another range
    /// beginning at \a dest. After this operation the elements in the
    /// moved-from range will still contain valid values of the appropriate
    /// type, but not necessarily the same values as before the move.
    ///
    /// \note   Complexity: Performs exactly \a last - \a first move assignments.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the move assignments.
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
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
    ///
    /// The move assignments in the parallel \a move algorithm invoked
    /// with an execution policy object of type
    /// \a sequential_execution_policy execute in sequential order in
    /// the calling thread.
    ///
    /// The move assignments in the parallel \a move algorithm invoked
    /// with an execution policy object of type
    /// \a parallel_execution_policy or \a parallel_task_execution_policy are
    /// permitted to execute in an unordered fashion in unspecified
    /// threads, and indeterminately sequenced within each thread.
    ///
    /// \returns  The \a move algorithm returns a
    ///           \a  hpx::future<tagged_pair<tag::in(InIter), tag::out(OutIter)> >
    ///           if the execution policy is of type
    ///           \a sequential_task_execution_policy or
    ///           \a parallel_task_execution_policy and
    ///           returns \a tagged_pair<tag::in(InIter), tag::out(OutIter)>
    ///           otherwise.
    ///           The \a move algorithm returns the pair of the input iterator
    ///           \a last and the output iterator to the
    ///           element in the destination range, one past the last element
    ///           moved.
    ///
    template <typename ExPolicy, typename InIter, typename OutIter,
    HPX_CONCEPT_REQUIRES_(
        is_execution_policy<ExPolicy>::value &&
        traits::is_iterator<InIter>::value &&
        traits::is_iterator<OutIter>::value)>
    typename util::detail::algorithm_result<
        ExPolicy, hpx::util::tagged_pair<tag::in(InIter), tag::out(OutIter)>
    >::type
    move(ExPolicy && policy, InIter first, InIter last, OutIter dest)
    {
        return detail::transfer<
                detail::move<InIter, OutIter>
            >(std::forward<ExPolicy>(policy), first, last, dest);
    }
}}}

#endif
