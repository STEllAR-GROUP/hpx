//  Copyright (c) 2017 Taeguk Kwon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_ALGORITHM_IS_HEAP_JUN_22_2017_1705PM)
#define HPX_PARALLEL_ALGORITHM_IS_HEAP_JUN_22_2017_1705PM

#include <hpx/config.hpp>
#include <hpx/async.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/traits/concepts.hpp>
#include <hpx/traits/is_callable.hpp>
#include <hpx/traits/is_iterator.hpp>
#include <hpx/util/invoke.hpp>

#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/executors/execution.hpp>
#include <hpx/parallel/traits/projected.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/projection_identity.hpp>
#include <hpx/parallel/util/loop.hpp>
#include <hpx/parallel/util/partitioner.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <list>
#include <vector>
#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { inline namespace v1
{
    ///////////////////////////////////////////////////////////////////////////
    // is_heap
    namespace detail
    {
        /// \cond NOINTERNAL

        // sequential is_heap with projection function
        template <typename RandIter, typename Comp, typename Proj>
        bool
        sequential_is_heap(RandIter first, RandIter last,
            Comp && comp, Proj && proj)
        {
            typedef typename std::iterator_traits<RandIter>::difference_type
                difference_type;

            difference_type count = last - first;

            for (difference_type i = 1; i < count; ++i)
            {
                if (hpx::util::invoke(comp,
                    hpx::util::invoke(proj, *(first + (i - 1) / 2)),
                    hpx::util::invoke(proj, *(first + i))))
                    return false;
            }
            return true;
        }

        struct is_heap_helper
        {
            template <typename ExPolicy, typename RandIter, typename Comp,
                typename Proj>
            typename util::detail::algorithm_result<ExPolicy, bool>::type
            operator()(ExPolicy && policy, RandIter first, RandIter last,
                Comp comp, Proj proj)
            {
                typedef util::detail::algorithm_result<ExPolicy, bool> result;
                typedef typename std::iterator_traits<RandIter>::value_type type;
                typedef typename std::iterator_traits<RandIter>::difference_type
                    difference_type;

                difference_type count = std::distance(first, last);
                if (count <= 1)
                    return result::get(true);

                RandIter second = first + 1;
                --count;
                util::cancellation_token<std::size_t> tok(count);

                return util::partitioner<ExPolicy, bool, void>::
                    call_with_index(
                        std::forward<ExPolicy>(policy), second, count, 1,
                        [tok, first, comp, proj](RandIter it,
                            std::size_t part_size, std::size_t base_idx
                        ) mutable -> void
                        {
                            util::loop_idx_n(
                                base_idx, it, part_size, tok,
                                [&tok, first, &comp, proj](
                                    type const& v, std::size_t i
                                ) -> void
                                {
                                    if (hpx::util::invoke(comp,
                                        hpx::util::invoke(proj, *(first + i / 2)),
                                        hpx::util::invoke(proj, v)))
                                        tok.cancel(0);
                                });
                        },
                        [tok](std::vector<hpx::future<void> > &&) mutable
                            -> bool
                        {
                            difference_type find_res =
                                static_cast<difference_type>(tok.get_data());

                            return find_res != 0;
                        });
            }
        };

        template <typename RandIter>
        struct is_heap
          : public detail::algorithm<is_heap<RandIter>, bool>
        {
            is_heap()
              : is_heap::algorithm("is_heap")
            {}

            template <typename ExPolicy, typename Comp, typename Proj>
            static bool
            sequential(ExPolicy && policy, RandIter first, RandIter last,
                Comp && comp, Proj && proj)
            {
                return sequential_is_heap(first, last,
                    std::forward<Comp>(comp), std::forward<Proj>(proj));
            }

            template <typename ExPolicy, typename Comp, typename Proj>
            static typename util::detail::algorithm_result<
                ExPolicy, bool
            >::type
            parallel(ExPolicy && policy, RandIter first, RandIter last,
                Comp && comp, Proj && proj)
            {
                return is_heap_helper()(
                        std::forward<ExPolicy>(policy), first, last,
                        std::forward<Comp>(comp), std::forward<Proj>(proj));
            }
        };
        /// \endcond
    }

    /// Returns whether the range is max heap. That is, true if the range is
    /// max heap, false otherwise. The function uses the given comparison
    /// function object \a comp (defaults to using operator<()).
    ///
    /// \note   Complexity:
    ///         Performs at most N applications of the comparison \a comp,
    ///         at most 2 * N applications of the projection \a proj,
    ///         where N = last - first.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam RandIter    The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     random access iterator.
    /// \tparam Comp        The type of the function/function object to use
    ///                     (deduced).
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param comp         \a comp is a callable object. The return value of the
    ///                     INVOKE operation applied to an object of type \a Comp,
    ///                     when contextually converted to bool, yields true if
    ///                     the first argument of the call is less than the
    ///                     second, and false otherwise. It is assumed that comp
    ///                     will not apply any non-constant function through the
    ///                     dereferenced iterator.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// \a comp has to induce a strict weak ordering on the values.
    ///
    /// The application of function objects in parallel algorithm
    /// invoked with an execution policy object of type
    /// \a sequenced_policy execute in sequential order in the
    /// calling thread.
    ///
    /// The application of function objects in parallel algorithm
    /// invoked with an execution policy object of type
    /// \a parallel_policy or \a parallel_task_policy are
    /// permitted to execute in an unordered fashion in unspecified
    /// threads, and indeterminately sequenced within each thread.
    ///
    /// \returns  The \a is_heap algorithm returns a \a hpx::future<bool>
    ///           if the execution policy is of type \a sequenced_task_policy or
    ///           \a parallel_task_policy and returns \a bool otherwise.
    ///           The \a is_heap algorithm returns whether the range is max heap.
    ///           That is, true if the range is max heap, false otherwise.
    ///
    template <typename ExPolicy, typename RandIter, typename Comp = detail::less,
        typename Proj = util::projection_identity,
    HPX_CONCEPT_REQUIRES_(
        execution::is_execution_policy<ExPolicy>::value &&
        hpx::traits::is_iterator<RandIter>::value &&
        traits::is_projected<Proj, RandIter>::value &&
        traits::is_indirect_callable<
            ExPolicy, Comp,
            traits::projected<Proj, RandIter>,
            traits::projected<Proj, RandIter>
        >::value)>
    typename util::detail::algorithm_result<ExPolicy, bool>::type
    is_heap(ExPolicy && policy, RandIter first, RandIter last,
        Comp && comp = Comp(), Proj && proj = Proj())
    {
        static_assert(
            (hpx::traits::is_random_access_iterator<RandIter>::value),
            "Requires a random access iterator.");

        typedef execution::is_sequenced_execution_policy<ExPolicy> is_seq;

        return detail::is_heap<RandIter>().call(
                std::forward<ExPolicy>(policy), is_seq(), first, last,
                std::forward<Comp>(comp), std::forward<Proj>(proj));
    }

    ///////////////////////////////////////////////////////////////////////////
    // is_heap_until
    namespace detail
    {
        /// \cond NOINTERNAL

        // sequential is_heap_until with projection function
        template <typename RandIter, typename Comp, typename Proj>
        RandIter
        sequential_is_heap_until(RandIter first, RandIter last,
            Comp && comp, Proj && proj)
        {
            typedef typename std::iterator_traits<RandIter>::difference_type
                difference_type;

            difference_type count = last - first;

            for (difference_type i = 1; i < count; ++i)
            {
                if (hpx::util::invoke(comp,
                    hpx::util::invoke(proj, *(first + (i - 1) / 2)),
                    hpx::util::invoke(proj, *(first + i))))
                    return first + i;
            }
            return last;
        }

        struct is_heap_until_helper
        {
            template <typename ExPolicy, typename RandIter, typename Comp,
                typename Proj>
            typename util::detail::algorithm_result<ExPolicy, RandIter>::type
            operator()(ExPolicy && policy, RandIter first, RandIter last,
                Comp comp, Proj proj)
            {
                typedef util::detail::algorithm_result<ExPolicy, RandIter> result;
                typedef typename std::iterator_traits<RandIter>::value_type type;
                typedef typename std::iterator_traits<RandIter>::difference_type
                    difference_type;

                difference_type count = std::distance(first, last);
                if (count <= 1)
                    return result::get(std::move(last));

                RandIter second = first + 1;
                --count;
                util::cancellation_token<std::size_t> tok(count);

                return util::partitioner<ExPolicy, RandIter, void>::
                    call_with_index(
                        std::forward<ExPolicy>(policy), second, count, 1,
                        [tok, first, comp, proj](RandIter it,
                            std::size_t part_size, std::size_t base_idx) mutable
                        {
                            util::loop_idx_n(
                                base_idx, it, part_size, tok,
                                [&tok, first, &comp, &proj](
                                    type const& v, std::size_t i
                                ) -> void
                                {
                                    if (hpx::util::invoke(comp,
                                        hpx::util::invoke(proj, *(first + i / 2)),
                                        hpx::util::invoke(proj, v)))
                                        tok.cancel(i);
                                });
                        },
                        [tok, second](std::vector<hpx::future<void> > &&) mutable
                            -> RandIter
                        {
                            difference_type find_res =
                                static_cast<difference_type>(tok.get_data());

                            std::advance(second, find_res);

                            return std::move(second);
                        });
            }
        };

        template <typename RandIter>
        struct is_heap_until
          : public detail::algorithm<is_heap_until<RandIter>, RandIter>
        {
            is_heap_until()
              : is_heap_until::algorithm("is_heap_until")
            {}

            template <typename ExPolicy, typename Comp, typename Proj>
            static RandIter
            sequential(ExPolicy && policy, RandIter first, RandIter last,
                Comp && comp, Proj && proj)
            {
                return sequential_is_heap_until(first, last,
                    std::forward<Comp>(comp), std::forward<Proj>(proj));
            }

            template <typename ExPolicy, typename Comp, typename Proj>
            static typename util::detail::algorithm_result<
                ExPolicy, RandIter
            >::type
            parallel(ExPolicy && policy, RandIter first, RandIter last,
                Comp && comp, Proj && proj)
            {
                return is_heap_until_helper()(
                    std::forward<ExPolicy>(policy), first, last,
                    std::forward<Comp>(comp), std::forward<Proj>(proj));
            }
        };
        /// \endcond
    }

    /// Returns the upper bound of the largest range beginning at \a first
    /// which is a max heap. That is, the last iterator \a it for
    /// which range [first, it) is a max heap. The function
    /// uses the given comparison function object \a comp (defaults to using
    /// operator<()).
    ///
    /// \note   Complexity:
    ///         Performs at most N applications of the comparison \a comp,
    ///         at most 2 * N applications of the projection \a proj,
    ///         where N = last - first.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam RandIter    The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     random access iterator.
    /// \tparam Comp        The type of the function/function object to use
    ///                     (deduced).
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param comp         \a comp is a callable object. The return value of the
    ///                     INVOKE operation applied to an object of type \a Comp,
    ///                     when contextually converted to bool, yields true if
    ///                     the first argument of the call is less than the
    ///                     second, and false otherwise. It is assumed that comp
    ///                     will not apply any non-constant function through the
    ///                     dereferenced iterator.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// \a comp has to induce a strict weak ordering on the values.
    ///
    /// The application of function objects in parallel algorithm
    /// invoked with an execution policy object of type
    /// \a sequenced_policy execute in sequential order in the
    /// calling thread.
    ///
    /// The application of function objects in parallel algorithm
    /// invoked with an execution policy object of type
    /// \a parallel_policy or \a parallel_task_policy are
    /// permitted to execute in an unordered fashion in unspecified
    /// threads, and indeterminately sequenced within each thread.
    ///
    /// \returns  The \a is_heap_until algorithm returns a \a hpx::future<RandIter>
    ///           if the execution policy is of type \a sequenced_task_policy or
    ///           \a parallel_task_policy and returns \a RandIter otherwise.
    ///           The \a is_heap_until algorithm returns the upper bound
    ///           of the largest range beginning at first which is a max heap.
    ///           That is, the last iterator \a it for which range [first, it) is a max heap.
    ///
    template <typename ExPolicy, typename RandIter, typename Comp = detail::less,
        typename Proj = util::projection_identity,
    HPX_CONCEPT_REQUIRES_(
        execution::is_execution_policy<ExPolicy>::value &&
        hpx::traits::is_iterator<RandIter>::value &&
        traits::is_projected<Proj, RandIter>::value &&
        traits::is_indirect_callable<
            ExPolicy, Comp,
            traits::projected<Proj, RandIter>,
            traits::projected<Proj, RandIter>
        >::value)>
    typename util::detail::algorithm_result<ExPolicy, RandIter>::type
    is_heap_until(ExPolicy && policy, RandIter first, RandIter last,
        Comp && comp = Comp(), Proj && proj = Proj())
    {
        static_assert(
            (hpx::traits::is_random_access_iterator<RandIter>::value),
            "Requires a random access iterator.");

        typedef execution::is_sequenced_execution_policy<ExPolicy> is_seq;

        return detail::is_heap_until<RandIter>().call(
                std::forward<ExPolicy>(policy), is_seq(), first, last,
                std::forward<Comp>(comp), std::forward<Proj>(proj));
    }
}}}

#endif
