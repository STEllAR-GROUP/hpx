//  Copyright (c) 2017 Taeguk Kwon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_ALGORITHM_MERGE_AUG_08_2017_0819AM)
#define HPX_PARALLEL_ALGORITHM_MERGE_AUG_08_2017_0819AM

#include <hpx/config.hpp>
#include <hpx/traits/concepts.hpp>
#include <hpx/traits/is_iterator.hpp>
#include <hpx/util/invoke.hpp>
#include <hpx/util/tagged_tuple.hpp>

#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/detail/is_negative.hpp>
#include <hpx/parallel/algorithms/detail/predicates.hpp>
#include <hpx/parallel/algorithms/detail/transfer.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/tagspec.hpp>
#include <hpx/parallel/traits/projected.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/detail/handle_local_exceptions.hpp>
#include <hpx/parallel/util/loop.hpp>
#include <hpx/parallel/util/projection_identity.hpp>
#include <hpx/parallel/util/transfer.hpp>

#include <algorithm>
#include <cstddef>
#include <exception>
#include <iterator>
#include <list>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace parallel { inline namespace v1
{
    /////////////////////////////////////////////////////////////////////////////
    // merge
    namespace detail
    {
        /// \cond NOINTERNAL

        // sequential merge with projection function.
        template <typename InIter1, typename InIter2, typename OutIter,
            typename Comp, typename Proj1, typename Proj2>
        hpx::util::tuple<InIter1, InIter2, OutIter>
        sequential_merge(InIter1 first1, InIter1 last1,
            InIter2 first2, InIter2 last2,
            OutIter dest, Comp && comp, Proj1 && proj1, Proj2 && proj2)
        {
            using hpx::util::invoke;

            if (first1 != last1 && first2 != last2)
            {
                while (true)
                {
                    if (invoke(comp,
                        invoke(proj2, *first2),
                        invoke(proj1, *first1)))
                    {
                        *dest++ = *first2++;
                        if (first2 == last2)
                            break;
                    }
                    else
                    {
                        *dest++ = *first1++;
                        if (first1 == last1)
                            break;
                    }
                }
            }
            dest = std::copy(first1, last1, dest);
            dest = std::copy(first2, last2, dest);

            return hpx::util::make_tuple(last1, last2, dest);
        }

        struct upper_bound_helper
        {
            // upper_bound with projection function.
            template<typename FwdIter, typename Type, typename Comp, typename Proj>
            static FwdIter
            call(FwdIter first, FwdIter last, const Type& value,
                Comp comp, Proj proj)
            {
                typedef typename std::iterator_traits<FwdIter>::difference_type
                    difference_type;

                using hpx::util::invoke;

                difference_type count = std::distance(first, last);

                while (count > 0)
                {
                    difference_type step = count / 2;
                    FwdIter mid = std::next(first, step);

                    if (!invoke(comp, value, invoke(proj, *mid)))
                    {
                        first = ++mid;
                        count -= step + 1;
                    }
                    else
                    {
                        count = step;
                    }
                }

                return first;
            }

            typedef struct lower_bound_helper another_type;
        };

        struct lower_bound_helper
        {
            // lower_bound with projection function.
            template<typename FwdIter, typename Type, typename Comp, typename Proj>
            static FwdIter
            call(FwdIter first, FwdIter last, const Type& value,
                Comp comp, Proj proj)
            {
                typedef typename std::iterator_traits<FwdIter>::difference_type
                    difference_type;

                using hpx::util::invoke;

                difference_type count = std::distance(first, last);

                while (count > 0)
                {
                    difference_type step = count / 2;
                    FwdIter mid = std::next(first, step);

                    if (invoke(comp, invoke(proj, *mid), value))
                    {
                        first = ++mid;
                        count -= step + 1;
                    }
                    else
                    {
                        count = step;
                    }
                }

                return first;
            }

            typedef struct upper_bound_helper another_type;
        };

        template <typename ExPolicy,
            typename RandIter1, typename RandIter2, typename RandIter3,
            typename Comp, typename Proj1, typename Proj2, typename BinarySearchHelper,
        HPX_CONCEPT_REQUIRES_(
            hpx::traits::is_random_access_iterator<RandIter1>::value &&
            hpx::traits::is_random_access_iterator<RandIter2>::value &&
            hpx::traits::is_random_access_iterator<RandIter3>::value)>
        void
        parallel_merge_helper(ExPolicy policy,
            RandIter1 first1, RandIter1 last1,
            RandIter2 first2, RandIter2 last2,
            RandIter3 dest, Comp comp,
            Proj1 proj1, Proj2 proj2, BinarySearchHelper)
        {
            using hpx::util::invoke;

            const std::size_t threshold = 65536ul;
            HPX_ASSERT(threshold >= 1ul);

            std::size_t size1 = last1 - first1;
            std::size_t size2 = last2 - first2;

            // Perform sequential merge if data size is smaller than threshold.
            if (size1 + size2 <= threshold)
            {
                sequential_merge(first1, first1 + size1,
                    first2, first2 + size2, dest, comp, proj1, proj2);
                return;
            }

            // Let size1 is bigger than size2 always.
            if (size1 < size2)
            {
                // For stability of algorithm, must switch binary search methods
                //   when swapping size1 and size2.
                parallel_merge_helper(policy,
                    first2, last2, first1, last1, dest, comp, proj2, proj1,
                    typename BinarySearchHelper::another_type());
                return;
            }

            HPX_ASSERT(size1 >= size2);
            HPX_ASSERT(size1 >= 1ul);

            RandIter1 mid1 = first1 + size1 / 2;
            RandIter2 boundary2 = BinarySearchHelper::call(
                first2, last2, invoke(proj1, *mid1), comp, proj2);
            RandIter3 target = dest + (mid1 - first1) + (boundary2 - first2);

            *target = *mid1;

            hpx::future<void> fut = execution::async_execute(policy.executor(),
                [&]() -> void
                {
                    // Process leftside ranges.
                    parallel_merge_helper(policy,
                        first1, mid1, first2, boundary2,
                        dest, comp, proj1, proj2, BinarySearchHelper());
                });

            try {
                // Process rightside ranges.
                parallel_merge_helper(policy,
                    mid1 + 1, last1, boundary2, last2,
                    target + 1, comp, proj1, proj2, BinarySearchHelper());
            }
            catch (...) {
                fut.wait();

                std::vector<hpx::future<void>> futures(2);
                futures[0] = std::move(fut);
                futures[1] = hpx::make_exceptional_future<void>(
                    std::current_exception());

                std::list<std::exception_ptr> errors;
                util::detail::handle_local_exceptions<ExPolicy>::call(
                    futures, errors);

                // Not reachable.
                HPX_ASSERT(false);
            }

            fut.get();
        }

        template <typename ExPolicy,
            typename RandIter1, typename RandIter2, typename RandIter3,
            typename Comp, typename Proj1, typename Proj2,
        HPX_CONCEPT_REQUIRES_(
            hpx::traits::is_random_access_iterator<RandIter1>::value &&
            hpx::traits::is_random_access_iterator<RandIter2>::value &&
            hpx::traits::is_random_access_iterator<RandIter3>::value)>
        hpx::util::tuple<RandIter1, RandIter2, RandIter3>
        parallel_merge(ExPolicy && policy,
            RandIter1 first1, RandIter1 last1,
            RandIter2 first2, RandIter2 last2,
            RandIter3 dest, Comp && comp,
            Proj1 && proj1, Proj2 && proj2)
        {
            parallel_merge_helper(std::forward<ExPolicy>(policy),
                first1, last1, first2, last2, dest,
                std::forward<Comp>(comp),
                std::forward<Proj1>(proj1),
                std::forward<Proj2>(proj2),
                upper_bound_helper());

            return hpx::util::make_tuple(last1, last2,
                dest + (last1 - first1) + (last2 - first2));
        }

        template <typename IterTuple>
        struct merge : public detail::algorithm<merge<IterTuple>, IterTuple>
        {
            merge()
              : merge::algorithm("merge")
            {}

            template <typename ExPolicy,
                typename InIter1, typename InIter2, typename OutIter,
                typename Comp, typename Proj1, typename Proj2>
            static hpx::util::tuple<InIter1, InIter2, OutIter>
            sequential(ExPolicy,
                InIter1 first1, InIter1 last1,
                InIter2 first2, InIter2 last2,
                OutIter dest, Comp && comp,
                Proj1 && proj1, Proj2 && proj2)
            {
                return sequential_merge(
                    first1, last1, first2, last2, dest,
                    std::forward<Comp>(comp),
                    std::forward<Proj1>(proj1),
                    std::forward<Proj2>(proj2));
            }

            template <typename ExPolicy,
                typename FwdIter1, typename FwdIter2, typename FwdIter3,
                typename Comp, typename Proj1, typename Proj2>
            static typename util::detail::algorithm_result<
                ExPolicy, hpx::util::tuple<FwdIter1, FwdIter2, FwdIter3>
            >::type
            parallel(ExPolicy && policy,
                FwdIter1 first1, FwdIter1 last1,
                FwdIter2 first2, FwdIter2 last2,
                FwdIter3 dest, Comp && comp,
                Proj1 && proj1, Proj2 && proj2)
            {
                typedef hpx::util::tuple<FwdIter1, FwdIter2, FwdIter3>
                    result_type;
                typedef util::detail::algorithm_result<
                    ExPolicy, result_type
                > algorithm_result;

                try {
                    return algorithm_result::get(
                        parallel_merge(std::forward<ExPolicy>(policy),
                            first1, last1, first2, last2, dest,
                            std::forward<Comp>(comp),
                            std::forward<Proj1>(proj1),
                            std::forward<Proj2>(proj2)));
                }
                catch (...) {
                    return algorithm_result::get(
                        detail::handle_exception<ExPolicy, result_type>::call(
                            std::current_exception()));
                }
            }
        };
        /// \endcond
    }

    /// Merges two sorted ranges [first1, last1) and [first2, last2)
    /// into one sorted range beginning at \a dest. The order of
    /// equivalent elements in the each of original two ranges is preserved.
    /// For equivalent elements in the original two ranges, the elements from
    /// the first range precede the elements from the second range.
    /// The destination range cannot overlap with either of the input ranges.
    ///
    /// \note   Complexity: Performs
    ///         O(std::distance(first1, last1) + std::distance(first2, last2))
    ///         applications of the comparison \a comp and the each projection.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter1    The type of the source iterators used (deduced)
    ///                     representing the first range.
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam FwdIter2    The type of the source iterators used (deduced)
    ///                     representing the second range.
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam FwdIter3    The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Comp        The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a merge requires \a Comp to meet the
    ///                     requirements of \a CopyConstructible. This defaults
    ///                     to std::less<>
    /// \tparam Proj1       The type of an optional projection function to be
    ///                     used for elements of the first range. This defaults
    ///                     to \a util::projection_identity
    /// \tparam Proj2       The type of an optional projection function to be
    ///                     used for elements of the second range. This defaults
    ///                     to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first1       Refers to the beginning of the first range of elements
    ///                     the algorithm will be applied to.
    /// \param last1        Refers to the end of the first range of elements
    ///                     the algorithm will be applied to.
    /// \param first2       Refers to the beginning of the second range of elements
    ///                     the algorithm will be applied to.
    /// \param last2        Refers to the end of the second range of elements
    ///                     the algorithm will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
    /// \param comp         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last). This is an
    ///                     binary predicate which returns \a true if the first
    ///                     argument is less than the second. The signature of
    ///                     this predicate should be equivalent to:
    ///                     \code
    ///                     bool pred(const Type1 &a, const Type2 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it. The types \a Type1 and \a Type2 must be such that
    ///                     objects of types \a FwdIter1 and \a FwdIter2 can be
    ///                     dereferenced and then implicitly converted to
    ///                     both \a Type1 and \a Type2
    /// \param proj1        Specifies the function (or function object) which
    ///                     will be invoked for each of the elements of the
    ///                     first range as a projection operation before the
    ///                     actual comparison \a comp is invoked.
    /// \param proj2        Specifies the function (or function object) which
    ///                     will be invoked for each of the elements of the
    ///                     second range as a projection operation before the
    ///                     actual comparison \a comp is invoked.
    ///
    /// The assignments in the parallel \a merge algorithm invoked with
    /// an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a merge algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a merge algorithm returns a
    /// \a hpx::future<tagged_tuple<tag::in1(FwdIter1), tag::in2(FwdIter2), tag::out(FwdIter3)> >
    ///           if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and returns
    /// \a tagged_tuple<tag::in1(FwdIter1), tag::in2(FwdIter2), tag::out(FwdIter3)>
    ///           otherwise.
    ///           The \a merge algorithm returns the tuple of
    ///           the source iterator \a last1,
    ///           the source iterator \a last2,
    ///           the destination iterator to the end of the \a dest range.
    ///
    template <typename ExPolicy,
        typename FwdIter1, typename FwdIter2, typename FwdIter3,
        typename Comp = detail::less,
        typename Proj1 = util::projection_identity,
        typename Proj2 = util::projection_identity,
    HPX_CONCEPT_REQUIRES_(
        execution::is_execution_policy<ExPolicy>::value &&
        hpx::traits::is_iterator<FwdIter1>::value &&
        hpx::traits::is_iterator<FwdIter2>::value &&
        hpx::traits::is_iterator<FwdIter3>::value &&
        traits::is_projected<Proj1, FwdIter1>::value &&
        traits::is_projected<Proj2, FwdIter2>::value &&
        traits::is_indirect_callable<
            ExPolicy, Comp,
            traits::projected<Proj1, FwdIter1>,
            traits::projected<Proj2, FwdIter2>
        >::value)>
    typename util::detail::algorithm_result<
        ExPolicy, hpx::util::tagged_tuple<
        tag::in1(FwdIter1), tag::in2(FwdIter2), tag::out(FwdIter3)>
    >::type
    merge(ExPolicy && policy,
        FwdIter1 first1, FwdIter1 last1,
        FwdIter2 first2, FwdIter2 last2,
        FwdIter3 dest, Comp && comp = Comp(),
        Proj1 && proj1 = Proj1(), Proj2 && proj2 = Proj2())
    {
#if defined(HPX_HAVE_ALGORITHM_INPUT_ITERATOR_SUPPORT)
        static_assert(
            (hpx::traits::is_input_iterator<FwdIter1>::value),
            "Required at least input iterator.");
        static_assert(
            (hpx::traits::is_input_iterator<FwdIter2>::value),
            "Required at least input iterator.");
        static_assert(
            (hpx::traits::is_output_iterator<FwdIter3>::value ||
                hpx::traits::is_forward_iterator<FwdIter3>::value),
            "Requires at least output iterator.");

        typedef std::integral_constant<bool,
                execution::is_sequenced_execution_policy<ExPolicy>::value ||
               !hpx::traits::is_forward_iterator<FwdIter1>::value ||
               !hpx::traits::is_forward_iterator<FwdIter2>::value ||
               !hpx::traits::is_forward_iterator<FwdIter3>::value
            > is_seq;
#else
        static_assert(
            (hpx::traits::is_forward_iterator<FwdIter1>::value),
            "Required at least forward iterator.");
        static_assert(
            (hpx::traits::is_forward_iterator<FwdIter2>::value),
            "Requires at least forward iterator.");
        static_assert(
            (hpx::traits::is_forward_iterator<FwdIter3>::value),
            "Requires at least forward iterator.");

        typedef execution::is_sequenced_execution_policy<ExPolicy> is_seq;
#endif

        typedef hpx::util::tuple<FwdIter1, FwdIter2, FwdIter3> result_type;

        return hpx::util::make_tagged_tuple<tag::in1, tag::in2, tag::out>(
            detail::merge<result_type>().call(
                std::forward<ExPolicy>(policy), is_seq(),
                first1, last1, first2, last2, dest,
                std::forward<Comp>(comp),
                std::forward<Proj1>(proj1),
                std::forward<Proj2>(proj2)));
    }
}}}

#endif
