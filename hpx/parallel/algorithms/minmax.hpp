//  Copyright (c) 2014-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// make inspect happy: hpxinspect:nominmax

/// \file parallel/algorithms/minmax.hpp

#if !defined(HPX_PARALLEL_DETAIL_MINMAX_AUG_20_2014_1005AM)
#define HPX_PARALLEL_DETAIL_MINMAX_AUG_20_2014_1005AM

#include <hpx/config.hpp>
#include <hpx/traits/concepts.hpp>
#include <hpx/traits/is_iterator.hpp>
#include <hpx/traits/segmented_iterator_traits.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/invoke.hpp>
#include <hpx/util/tagged_pair.hpp>

#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/tagspec.hpp>
#include <hpx/parallel/traits/projected.hpp>
#include <hpx/parallel/util/compare_projected.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/loop.hpp>
#include <hpx/parallel/util/partitioner.hpp>
#include <hpx/parallel/util/projection_identity.hpp>

#include <algorithm>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
    ///////////////////////////////////////////////////////////////////////////
    // min_element
    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename FwdIter, typename F, typename Proj>
        FwdIter sequential_min_element(FwdIter it, std::size_t count,
            F const& f, Proj const& proj)
        {
            if (count == 0 || count == 1)
                return it;

            FwdIter smallest = it;
            util::loop_n(++it, count-1,
                [&f, &smallest, &proj](FwdIter const& curr)
                {
                    if (hpx::util::invoke(f,
                            hpx::util::invoke(proj, *curr),
                            hpx::util::invoke(proj, *smallest)))
                    {
                        smallest = curr;
                    }
                });
            return smallest;
        }

        template <typename Iter>
        struct min_element
          : public detail::algorithm<min_element<Iter>, Iter>
        {
            min_element()
              : min_element::algorithm("min_element")
            {}

            template <typename FwdIter, typename F, typename Proj>
            static typename std::iterator_traits<FwdIter>::value_type
            sequential_minmax_element_ind(FwdIter it, std::size_t count,
                F const& f, Proj const& proj)
            {
                HPX_ASSERT(count != 0);

                if (count == 1)
                    return *it;

                typename std::iterator_traits<FwdIter>::value_type smallest = *it;
                util::loop_n(++it, count-1,
                    [&f, &smallest, &proj](FwdIter const& curr)
                    {
                        if (hpx::util::invoke(f,
                                hpx::util::invoke(proj, **curr),
                                hpx::util::invoke(proj, *smallest)))
                        {
                            smallest = *curr;
                        }
                    });
                return smallest;
            }

            template <typename ExPolicy, typename FwdIter, typename F, typename Proj>
            static FwdIter
            sequential(ExPolicy, FwdIter first, FwdIter last, F && f, Proj && proj)
            {
                return std::min_element(first, last,
                    util::compare_projected<F, Proj>(
                            std::forward<F>(f), std::forward<Proj>(proj)
                        ));
            }

            template <typename ExPolicy, typename FwdIter, typename F, typename Proj>
            static typename util::detail::algorithm_result<
                ExPolicy, FwdIter
            >::type
            parallel(ExPolicy && policy, FwdIter first, FwdIter last, F && f,
                Proj && proj)
            {
                if (first == last)
                {
                    return util::detail::algorithm_result<ExPolicy, FwdIter>::
                        get(std::move(first));
                }

                return util::partitioner<ExPolicy, FwdIter, FwdIter>::call(
                        std::forward<ExPolicy>(policy),
                        first, std::distance(first, last),
                        [f, proj](FwdIter it, std::size_t part_count)
                        {
                            return sequential_min_element(
                                it, part_count, f, proj);
                        },
                        hpx::util::unwrapped(
                            [f, proj](std::vector<FwdIter> && positions)
                            {
                                return min_element::sequential_minmax_element_ind(
                                    positions.begin(), positions.size(), f, proj);
                            }
                        )
                    );
            }
        };

        ///////////////////////////////////////////////////////////////////////
        // non-segmented implementation
        template <typename ExPolicy, typename FwdIter, typename F, typename Proj>
        inline typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
        min_element_(ExPolicy && policy, FwdIter first, FwdIter last, F && f,
            Proj && proj, std::false_type)
        {
            typedef parallel::is_sequential_execution_policy<ExPolicy> is_seq;

            return detail::min_element<FwdIter>().call(
                std::forward<ExPolicy>(policy), is_seq(),
                first, last, std::forward<F>(f), std::forward<Proj>(proj));
        }

        ///////////////////////////////////////////////////////////////////////
        // segmented implementation
        template <typename ExPolicy, typename FwdIter, typename F, typename Proj>
        inline typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
        min_element_(ExPolicy && policy, FwdIter first, FwdIter last, F && f,
            Proj && proj, std::true_type);

        /// \endcond
    }

    /// Finds the smallest element in the range [first, last) using the given
    /// comparison function \a f.
    ///
    /// \note   Complexity: Exactly \a max(N-1, 0) comparisons, where
    ///                     N = std::distance(first, last).
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
    ///                     overload of \a min_element requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param f            The binary predicate which returns true if the
    ///                     the left argument is less than the right element.
    ///                     The signature
    ///                     of the predicate function should be equivalent to
    ///                     the following:
    ///                     \code
    ///                     bool pred(const Type1 &a, const Type1 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type1 must be such that objects of
    ///                     type \a FwdIter can be dereferenced and then
    ///                     implicitly converted to \a Type1.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The comparisons in the parallel \a min_element algorithm invoked with
    /// an execution policy object of type \a sequential_execution_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The comparisons in the parallel \a min_element algorithm invoked with
    /// an execution policy object of type \a parallel_execution_policy or
    /// \a parallel_task_execution_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a min_element algorithm returns a \a hpx::future<FwdIter>
    ///           if the execution policy is of type
    ///           \a sequential_task_execution_policy or
    ///           \a parallel_task_execution_policy
    ///           and returns \a FwdIter otherwise.
    ///           The \a min_element algorithm returns the iterator to the
    ///           smallest element in the range [first, last). If several
    ///           elements in the range are equivalent to the smallest element,
    ///           returns the iterator to the first such element. Returns last
    ///           if the range is empty.
    ///
    template <typename ExPolicy, typename FwdIter,
        typename Proj = util::projection_identity,
        typename F = std::less<
            typename std::remove_reference<
                typename traits::projected_result_of<Proj, FwdIter>::type
            >::type>,
    HPX_CONCEPT_REQUIRES_(
        is_execution_policy<ExPolicy>::value &&
        hpx::traits::is_iterator<FwdIter>::value &&
        traits::is_projected<Proj, FwdIter>::value &&
        traits::is_indirect_callable<
            F,
                traits::projected<Proj, FwdIter>,
                traits::projected<Proj, FwdIter>
        >::value)>
    typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
    min_element(ExPolicy && policy, FwdIter first, FwdIter last, F && f = F(),
        Proj && proj = Proj())
    {
        static_assert(
            (hpx::traits::is_forward_iterator<FwdIter>::value),
            "Requires at least forward iterator.");

        typedef hpx::traits::is_segmented_iterator<FwdIter> is_segmented;

        return detail::min_element_(
                std::forward<ExPolicy>(policy), first, last,
                std::forward<F>(f), std::forward<Proj>(proj), is_segmented());
    }

    ///////////////////////////////////////////////////////////////////////////
    // max_element
    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename FwdIter, typename F, typename Proj>
        FwdIter sequential_max_element(FwdIter it, std::size_t count,
            F const& f, Proj const& proj)
        {
            if (count == 0 || count == 1)
                return it;

            FwdIter greatest = it;
            util::loop_n(++it, count-1,
                [&f, &greatest, &proj](FwdIter const& curr)
                {
                    if (hpx::util::invoke(f,
                            hpx::util::invoke(proj, *greatest),
                            hpx::util::invoke(proj, *curr)))
                    {
                        greatest = curr;
                    }
                });
            return greatest;
        }

        template <typename Iter>
        struct max_element
          : public detail::algorithm<max_element<Iter>, Iter>
        {
            max_element()
              : max_element::algorithm("max_element")
            {}

            template <typename FwdIter, typename F, typename Proj>
            static typename std::iterator_traits<FwdIter>::value_type
            sequential_minmax_element_ind(FwdIter it, std::size_t count,
                F const& f, Proj const& proj)
            {
                HPX_ASSERT(count != 0);

                if (count == 1)
                    return *it;

                typename std::iterator_traits<FwdIter>::value_type greatest = *it;
                util::loop_n(++it, count-1,
                    [&f, &greatest, &proj](FwdIter const& curr)
                    {
                        if (hpx::util::invoke(f,
                                hpx::util::invoke(proj, *greatest),
                                hpx::util::invoke(proj, **curr)))
                        {
                            greatest = *curr;
                        }
                    });
                return greatest;
            }

            template <typename ExPolicy, typename FwdIter, typename F,
                typename Proj>
            static FwdIter
            sequential(ExPolicy, FwdIter first, FwdIter last, F && f,
                Proj && proj)
            {
                return std::max_element(first, last,
                    util::compare_projected<F, Proj>(
                            std::forward<F>(f), std::forward<Proj>(proj)
                        ));
            }

            template <typename ExPolicy, typename FwdIter, typename F,
                typename Proj>
            static typename util::detail::algorithm_result<
                ExPolicy, FwdIter
            >::type
            parallel(ExPolicy && policy, FwdIter first, FwdIter last, F && f,
                Proj && proj)
            {
                if (first == last)
                {
                    return util::detail::algorithm_result<ExPolicy, FwdIter>::
                        get(std::move(first));
                }

                return util::partitioner<ExPolicy, FwdIter, FwdIter>::call(
                        std::forward<ExPolicy>(policy),
                        first, std::distance(first, last),
                        [f, proj](FwdIter it, std::size_t part_count)
                        {
                            return sequential_max_element(
                                it, part_count, f, proj);
                        },
                        hpx::util::unwrapped(
                            [f, proj](std::vector<FwdIter> && positions)
                            {
                                return max_element::sequential_minmax_element_ind(
                                    positions.begin(), positions.size(), f, proj);
                            }
                        )
                    );
            }
        };

        ///////////////////////////////////////////////////////////////////////
        // non-segmented implementation
        template <typename ExPolicy, typename FwdIter, typename F, typename Proj>
        inline typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
        max_element_(ExPolicy && policy, FwdIter first, FwdIter last, F && f,
            Proj && proj, std::false_type)
        {
            typedef parallel::is_sequential_execution_policy<ExPolicy> is_seq;

            return detail::max_element<FwdIter>().call(
                std::forward<ExPolicy>(policy), is_seq(),
                first, last, std::forward<F>(f), std::forward<Proj>(proj));
        }

        ///////////////////////////////////////////////////////////////////////
        // segmented implementation
        template <typename ExPolicy, typename FwdIter, typename F, typename Proj>
        inline typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
        max_element_(ExPolicy && policy, FwdIter first, FwdIter last, F && f,
            Proj && proj, std::true_type);

        /// \endcond
    }

    /// Finds the greatest element in the range [first, last) using the given
    /// comparison function \a f.
    ///
    /// \note   Complexity: Exactly \a max(N-1, 0) comparisons, where
    ///                     N = std::distance(first, last).
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
    ///                     overload of \a max_element requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param f            The binary predicate which returns true if the
    ///                     This argument is optional and defaults to std::less.
    ///                     the left argument is less than the right element.
    ///                     The signature
    ///                     of the predicate function should be equivalent to
    ///                     the following:
    ///                     \code
    ///                     bool pred(const Type1 &a, const Type1 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type1 must be such that objects of
    ///                     type \a FwdIter can be dereferenced and then
    ///                     implicitly converted to \a Type1.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The comparisons in the parallel \a max_element algorithm invoked with
    /// an execution policy object of type \a sequential_execution_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The comparisons in the parallel \a max_element algorithm invoked with
    /// an execution policy object of type \a parallel_execution_policy or
    /// \a parallel_task_execution_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a max_element algorithm returns a \a hpx::future<FwdIter>
    ///           if the execution policy is of type
    ///           \a sequential_task_execution_policy or
    ///           \a parallel_task_execution_policy
    ///           and returns \a FwdIter otherwise.
    ///           The \a max_element algorithm returns the iterator to the
    ///           smallest element in the range [first, last). If several
    ///           elements in the range are equivalent to the smallest element,
    ///           returns the iterator to the first such element. Returns last
    ///           if the range is empty.
    ///
    template <typename ExPolicy, typename FwdIter,
        typename Proj = util::projection_identity,
        typename F = std::less<
            typename std::remove_reference<
                typename traits::projected_result_of<Proj, FwdIter>::type
            >::type>,
    HPX_CONCEPT_REQUIRES_(
        is_execution_policy<ExPolicy>::value &&
        hpx::traits::is_iterator<FwdIter>::value &&
        traits::is_projected<Proj, FwdIter>::value &&
        traits::is_indirect_callable<
            F,
                traits::projected<Proj, FwdIter>,
                traits::projected<Proj, FwdIter>
        >::value)>
    typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
    max_element(ExPolicy && policy, FwdIter first, FwdIter last, F && f = F(),
        Proj && proj = Proj())
    {
        static_assert(
            (hpx::traits::is_forward_iterator<FwdIter>::value),
            "Requires at least forward iterator.");

        typedef hpx::traits::is_segmented_iterator<FwdIter> is_segmented;

        return detail::max_element_(
                std::forward<ExPolicy>(policy), first, last,
                std::forward<F>(f), std::forward<Proj>(proj), is_segmented());
    }

    ///////////////////////////////////////////////////////////////////////////
    // minmax_element
    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename FwdIter, typename F, typename Proj>
        std::pair<FwdIter, FwdIter>
        sequential_minmax_element(FwdIter it, std::size_t count, F const& f,
            Proj const& proj)
        {
            std::pair<FwdIter, FwdIter> result(it, it);

            if (count == 0 || count == 1)
                return result;

            util::loop_n(++it, count-1,
                [&f, &result, &proj](FwdIter const& curr)
                {
                    if (hpx::util::invoke(f,
                            hpx::util::invoke(proj, *curr),
                            hpx::util::invoke(proj, *result.first)))
                    {
                        result.first = curr;
                    }

                    if (!hpx::util::invoke(f,
                            hpx::util::invoke(proj, *curr),
                            hpx::util::invoke(proj, *result.second)))
                    {
                        result.second = curr;
                    }
                });
            return result;
        }

        template <typename PairIter, typename F, typename Proj>
        typename std::iterator_traits<PairIter>::value_type
        sequential_minmax_element_ind(PairIter it, std::size_t count,
            F const& f, Proj const& proj)
        {
            HPX_ASSERT(count != 0);

            if (count == 1)
                return *it;

            typename std::iterator_traits<PairIter>::value_type result = *it;
            util::loop_n(++it, count-1,
                [&f, &result, &proj](PairIter const& curr)
                {
                    if (hpx::util::invoke(f,
                            hpx::util::invoke(proj, *curr->first),
                            hpx::util::invoke(proj, *result.first)))
                    {
                        result.first = curr->first;
                    }

                    if (!hpx::util::invoke(f,
                            hpx::util::invoke(proj, *curr->second),
                            hpx::util::invoke(proj, *result.second)))
                    {
                        result.second = curr->second;
                    }
                });
            return result;
        }

        template <typename Iter>
        struct minmax_element
          : public detail::algorithm<
                minmax_element<Iter>, std::pair<Iter, Iter>
            >
        {
            minmax_element()
              : minmax_element::algorithm("minmax_element")
            {}

            template <typename ExPolicy, typename FwdIter, typename F,
                typename Proj>
            static std::pair<FwdIter, FwdIter>
            sequential(ExPolicy, FwdIter first, FwdIter last, F && f,
                Proj && proj)
            {
                return std::minmax_element(first, last,
                    util::compare_projected<F, Proj>(
                            std::forward<F>(f), std::forward<Proj>(proj)
                        ));
            }

            template <typename ExPolicy, typename FwdIter, typename F,
                typename Proj>
            static typename util::detail::algorithm_result<
                ExPolicy, std::pair<FwdIter, FwdIter>
            >::type
            parallel(ExPolicy && policy, FwdIter first, FwdIter last,
                F && f, Proj && proj)
            {
                typedef std::pair<FwdIter, FwdIter> result_type;

                result_type result(first, first);
                if (first == last || ++first == last)
                {
                    return util::detail::algorithm_result<
                            ExPolicy, result_type
                        >::get(std::move(result));
                }

                return util::partitioner<ExPolicy, result_type, result_type>::
                    call(
                        std::forward<ExPolicy>(policy),
                        result.first, std::distance(result.first, last),
                        [f, proj](FwdIter it, std::size_t part_count)
                        {
                            return sequential_minmax_element(
                                it, part_count, f, proj);
                        },
                        hpx::util::unwrapped(
                            [f, proj](std::vector<result_type> && positions)
                            {
                                return sequential_minmax_element_ind(
                                    positions.begin(), positions.size(), f,
                                    proj);
                            }
                        )
                    );
            }
        };

        ///////////////////////////////////////////////////////////////////////
        // non-segmented implementation
        template <typename ExPolicy, typename FwdIter, typename F, typename Proj>
        inline typename util::detail::algorithm_result<
            ExPolicy, std::pair<FwdIter, FwdIter>
        >::type
        minmax_element_(ExPolicy && policy, FwdIter first, FwdIter last, F && f,
            Proj && proj, std::false_type)
        {
            typedef parallel::is_sequential_execution_policy<ExPolicy> is_seq;

            return detail::minmax_element<FwdIter>().call(
                std::forward<ExPolicy>(policy), is_seq(),
                first, last, std::forward<F>(f), std::forward<Proj>(proj));
        }

        ///////////////////////////////////////////////////////////////////////
        // segmented implementation
        template <typename ExPolicy, typename FwdIter, typename F, typename Proj>
        inline typename util::detail::algorithm_result<
            ExPolicy, std::pair<FwdIter, FwdIter>
        >::type
        minmax_element_(ExPolicy && policy, FwdIter first, FwdIter last, F && f,
            Proj && proj, std::true_type);

        /// \endcond
    }

    /// Finds the greatest element in the range [first, last) using the given
    /// comparison function \a f.
    ///
    /// \note   Complexity: At most \a max(floor(3/2*(N-1)), 0) applications of
    ///                     the predicate, where N = std::distance(first, last).
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
    ///                     overload of \a minmax_element requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param f            The binary predicate which returns true if the
    ///                     the left argument is less than the right element.
    ///                     This argument is optional and defaults to std::less.
    ///                     The signature
    ///                     of the predicate function should be equivalent to
    ///                     the following:
    ///                     \code
    ///                     bool pred(const Type1 &a, const Type1 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type1 must be such that objects of
    ///                     type \a FwdIter can be dereferenced and then
    ///                     implicitly converted to \a Type1.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The comparisons in the parallel \a minmax_element algorithm invoked with
    /// an execution policy object of type \a sequential_execution_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The comparisons in the parallel \a minmax_element algorithm invoked with
    /// an execution policy object of type \a parallel_execution_policy or
    /// \a parallel_task_execution_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a minmax_element algorithm returns a
    /// \a hpx::future<tagged_pair<tag::min(FwdIter), tag::max(FwdIter)> >
    ///           if the execution policy is of type
    ///           \a sequential_task_execution_policy or
    ///           \a parallel_task_execution_policy
    ///           and returns \a tagged_pair<tag::min(FwdIter), tag::max(FwdIter)>
    ///           otherwise.
    ///           The \a minmax_element algorithm returns a pair consisting of
    ///           an iterator to the smallest element as the first element and
    ///           an iterator to the greatest element as the second. Returns
    ///           std::make_pair(first, first) if the range is empty. If
    ///           several elements are equivalent to the smallest element, the
    ///           iterator to the first such element is returned. If several
    ///           elements are equivalent to the largest element, the iterator
    ///           to the last such element is returned.
    ///
#if defined(HPX_MSVC)
#pragma push_macro("min")
#pragma push_macro("max")
#undef min
#undef max
#endif
    template <typename ExPolicy, typename FwdIter,
        typename Proj = util::projection_identity,
        typename F = std::less<
            typename std::remove_reference<
                typename traits::projected_result_of<Proj, FwdIter>::type
            >::type>,
    HPX_CONCEPT_REQUIRES_(
        is_execution_policy<ExPolicy>::value &&
        hpx::traits::is_iterator<FwdIter>::value &&
        traits::is_projected<Proj, FwdIter>::value &&
        traits::is_indirect_callable<
            F, traits::projected<Proj, FwdIter>, traits::projected<Proj, FwdIter>
        >::value)>
    typename util::detail::algorithm_result<
        ExPolicy, hpx::util::tagged_pair<tag::min(FwdIter), tag::max(FwdIter)>
    >::type
    minmax_element(ExPolicy && policy, FwdIter first, FwdIter last,
        F && f = F(), Proj && proj = Proj())
    {
        static_assert(
            (hpx::traits::is_forward_iterator<FwdIter>::value),
            "Requires at least forward iterator.");

        typedef hpx::traits::is_segmented_iterator<FwdIter> is_segmented;

        return hpx::util::make_tagged_pair<tag::min, tag::max>(
            detail::minmax_element_(
                std::forward<ExPolicy>(policy), first, last,
                std::forward<F>(f), std::forward<Proj>(proj), is_segmented()));
    }
#if defined(HPX_MSVC)
#pragma pop_macro("min")
#pragma pop_macro("max")
#endif
}}}

#endif
