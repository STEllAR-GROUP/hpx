//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/for_each.hpp

#if !defined(HPX_PARALLEL_DETAIL_FOR_EACH_MAY_29_2014_0932PM)
#define HPX_PARALLEL_DETAIL_FOR_EACH_MAY_29_2014_0932PM

#include <hpx/config.hpp>
#include <hpx/util/invoke.hpp>
#include <hpx/traits/segmented_iterator_traits.hpp>
#include <hpx/traits/is_callable.hpp>
#include <hpx/traits/is_iterator.hpp>
#include <hpx/traits/concepts.hpp>

#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/detail/is_negative.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/foreach_partitioner.hpp>
#include <hpx/parallel/util/loop.hpp>
#include <hpx/parallel/util/projection_identity.hpp>
#include <hpx/parallel/traits/projected.hpp>

#include <algorithm>
#include <iterator>
#include <type_traits>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
    ///////////////////////////////////////////////////////////////////////////
    // for_each_n
    namespace detail
    {
        template <typename F, typename Proj>
        struct for_each_iteration
        {
            typename hpx::util::decay<F>::type f_;
            typename hpx::util::decay<Proj>::type proj_;

            template <typename Iter>
            HPX_HOST_DEVICE
            void operator()(std::size_t /*part_index*/,
                Iter part_begin, std::size_t part_size)
            {
                util::loop_n(
                    part_begin, part_size,
                    [this](Iter curr) mutable
                    {
                        hpx::util::invoke(
                            f_, hpx::util::invoke(proj_, *curr));
                    });
            }
        };

        /// \cond NOINTERNAL
        template <typename Iter>
        struct for_each_n : public detail::algorithm<for_each_n<Iter>, Iter>
        {
            for_each_n()
              : for_each_n::algorithm("for_each_n")
            {}

            template <typename ExPolicy, typename InIter, typename F,
                typename Proj = util::projection_identity>
            HPX_HOST_DEVICE
            static Iter
            sequential(ExPolicy, InIter first, std::size_t count, F && f,
                Proj && proj/* = Proj()*/)
            {
                return util::loop_n(first, count,
                    [&f, &proj](InIter curr)
                    {
                        hpx::util::invoke(f, hpx::util::invoke(proj, *curr));
                    });
            }

            template <typename ExPolicy, typename InIter, typename F,
                typename Proj = util::projection_identity>
            static typename util::detail::algorithm_result<ExPolicy, InIter>::type
            parallel(ExPolicy && policy, InIter first, std::size_t count,
                F && f, Proj && proj/* = Proj()*/)
            {
                if (count != 0)
                {
                    // Some compilers complain about loosing const type
                    // modifiers if the lambdas below are not mutable.
                    return util::foreach_partitioner<ExPolicy>::call(
                        std::forward<ExPolicy>(policy), first, count,
                        for_each_iteration<F, Proj>{
                            std::forward<F>(f), std::forward<Proj>(proj)
                        },
                        [](InIter && last) -> InIter
                        {
                            return std::move(last);
                        });
                }

                return util::detail::algorithm_result<ExPolicy, InIter>::get(
                    std::move(first));
            }
        };
        /// \endcond
    }

    /// Applies \a f to the result of dereferencing every iterator in the range
    /// [first, first + count), starting from first and proceeding to
    /// first + count - 1.
    ///
    /// \note   Complexity: Applies \a f exactly \a count times.
    ///
    /// If \a f returns a result, the result is ignored.
    ///
    /// If the type of \a first satisfies the requirements of a mutable
    /// iterator, \a f may apply non-constant functions through the
    /// dereferenced iterator.
    ///
    /// Unlike its sequential form, the parallel overload of
    /// \a for_each does not return a copy of its \a Function parameter,
    /// since parallelization may not permit efficient state
    /// accumulation.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it applies user-provided function objects.
    /// \tparam InIter      The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
    /// \tparam Size        The type of the argument specifying the number of
    ///                     elements to apply \a f to.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a for_each requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param count        Refers to the number of elements starting at
    ///                     \a first the algorithm will be applied to.
    /// \param f            Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last).
    ///                     The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     <ignored> pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&. The
    ///                     type \a Type must be such that an object of
    ///                     type \a InIter can be dereferenced and then
    ///                     implicitly converted to Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a f is invoked.
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
    /// \returns  The \a for_each_n algorithm returns a
    ///           \a hpx::future<InIter> if the execution policy is of
    ///           type
    ///           \a sequential_task_execution_policy or
    ///           \a parallel_task_execution_policy and returns \a InIter
    ///           otherwise.
    ///           It returns \a first + \a count for non-negative values of
    ///           \a count and \a first for negative values.
    ///
    template <typename ExPolicy, typename InIter, typename Size, typename F,
        typename Proj,
    HPX_CONCEPT_REQUIRES_(
        is_execution_policy<ExPolicy>::value &&
        hpx::traits::is_iterator<InIter>::value &&
        parallel::traits::is_projected<Proj, InIter>::value &&
        parallel::traits::is_indirect_callable<
            F, traits::projected<Proj, InIter>
        >::value)>
    typename util::detail::algorithm_result<ExPolicy, InIter>::type
    for_each_n(ExPolicy && policy, InIter first, Size count, F && f,
        Proj && proj)
    {
        static_assert(
            (hpx::traits::is_input_iterator<InIter>::value),
            "Requires at least input iterator.");

        // if count is representing a negative value, we do nothing
        if (detail::is_negative(count))
        {
            return util::detail::algorithm_result<ExPolicy, InIter>::get(
                std::move(first));
        }

        typedef std::integral_constant<bool,
                is_sequential_execution_policy<ExPolicy>::value ||
               !hpx::traits::is_forward_iterator<InIter>::value
            > is_seq;

        return detail::for_each_n<InIter>().call(
            std::forward<ExPolicy>(policy), is_seq(),
            first, std::size_t(count), std::forward<F>(f),
            std::forward<Proj>(proj));
    }

    /// \cond NOINTERNAL
    template <typename ExPolicy, typename InIter, typename Size, typename F,
    HPX_CONCEPT_REQUIRES_(
        is_execution_policy<ExPolicy>::value &&
        hpx::traits::is_iterator<InIter>::value &&
        parallel::traits::is_indirect_callable<
            F, traits::projected<util::projection_identity, InIter>
        >::value)>
    typename util::detail::algorithm_result<ExPolicy, InIter>::type
    for_each_n(ExPolicy && policy, InIter first, Size count, F && f)
    {
        return for_each_n(std::forward<ExPolicy>(policy),
            first, count, std::forward<F>(f), util::projection_identity());
    }
    /// \endcond

    ///////////////////////////////////////////////////////////////////////////
    // for_each
    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename Iter>
        struct for_each : public detail::algorithm<for_each<Iter>, Iter>
        {
            for_each()
              : for_each::algorithm("for_each")
            {}

            template <typename ExPolicy, typename InIter, typename F,
                typename Proj>
            static Iter
            sequential(ExPolicy, InIter first, InIter last, F && f,
                Proj && proj)
            {
                return util::loop(first, last,
                    [&f, &proj](Iter curr)
                    {
                        f(hpx::util::invoke(proj, *curr));
                    });
            }

            template <typename ExPolicy, typename InIter, typename F>
            static InIter
            sequential(ExPolicy, InIter first, InIter last, F && f,
                util::projection_identity)
            {
                std::for_each(first, last, std::forward<F>(f));
                return last;
            }

            template <typename ExPolicy, typename InIter, typename F,
                typename Proj>
            static typename util::detail::algorithm_result<ExPolicy, InIter>::type
            parallel(ExPolicy && policy, InIter first, InIter last, F && f,
                Proj && proj)
            {
                return detail::for_each_n<Iter>().call(
                    std::forward<ExPolicy>(policy), std::false_type(),
                    first, std::distance(first, last), std::forward<F>(f),
                    std::forward<Proj>(proj));
            }
        };

        ///////////////////////////////////////////////////////////////////////
        // non-segmented implementation
        template <typename ExPolicy, typename InIter, typename F, typename Proj>
        inline typename util::detail::algorithm_result<ExPolicy, InIter>::type
        for_each_(ExPolicy && policy, InIter first, InIter last, F && f,
            Proj && proj, std::false_type)
        {
            typedef std::integral_constant<bool,
                    parallel::is_sequential_execution_policy<ExPolicy>::value ||
                   !hpx::traits::is_forward_iterator<InIter>::value
                > is_seq;

            if (first == last)
            {
                typedef util::detail::algorithm_result<ExPolicy, InIter> result;
                return result::get(std::move(last));
            }

            return for_each<InIter>().call(
                std::forward<ExPolicy>(policy), is_seq(),
                first, last, std::forward<F>(f), std::forward<Proj>(proj));
        }

        // forward declare the segmented version of this algorithm
        template <typename ExPolicy, typename SegIter, typename F, typename Proj>
        inline typename util::detail::algorithm_result<ExPolicy, SegIter>::type
        for_each_(ExPolicy && policy, SegIter first, SegIter last, F && f,
            Proj && proj, std::true_type);

        /// \endcond
    }

    /// Applies \a f to the result of dereferencing every iterator in the
    /// range [first, last).
    ///
    /// \note   Complexity: Applies \a f exactly \a last - \a first times.
    ///
    /// If \a f returns a result, the result is ignored.
    ///
    /// If the type of \a first satisfies the requirements of a mutable
    /// iterator, \a f may apply non-constant functions through the
    /// dereferenced iterator.
    ///
    /// Unlike its sequential form, the parallel overload of
    /// \a for_each does not return a copy of its \a Function parameter,
    /// since parallelization may not permit efficient state
    /// accumulation.
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
    ///                     overload of \a for_each requires \a F to meet the
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
    /// \param f            Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last).
    ///                     The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     <ignored> pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&. The
    ///                     type \a Type must be such that an object of
    ///                     type \a InIter can be dereferenced and then
    ///                     implicitly converted to Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a f is invoked.
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
    /// \returns  The \a for_each algorithm returns a
    ///           \a hpx::future<InIter> if the execution policy is of
    ///           type
    ///           \a sequential_task_execution_policy or
    ///           \a parallel_task_execution_policy and returns \a InIter
    ///           otherwise.
    ///           It returns \a last.
    ///
    template <typename ExPolicy, typename InIter, typename F, typename Proj,
    HPX_CONCEPT_REQUIRES_(
        is_execution_policy<ExPolicy>::value &&
        hpx::traits::is_iterator<InIter>::value &&
        parallel::traits::is_projected<Proj, InIter>::value &&
        parallel::traits::is_indirect_callable<
            F, traits::projected<Proj, InIter>
        >::value)>
    typename util::detail::algorithm_result<ExPolicy, InIter>::type
    for_each(ExPolicy && policy, InIter first, InIter last, F && f,
        Proj && proj)
    {
        static_assert(
            (hpx::traits::is_input_iterator<InIter>::value),
            "Requires at least input iterator.");

        typedef hpx::traits::is_segmented_iterator<InIter> is_segmented;

        return detail::for_each_(
            std::forward<ExPolicy>(policy), first, last,
            std::forward<F>(f), std::forward<Proj>(proj), is_segmented());
    }

    /// \cond NOINTERNAL
    template <typename ExPolicy, typename InIter, typename F,
    HPX_CONCEPT_REQUIRES_(
        is_execution_policy<ExPolicy>::value &&
        hpx::traits::is_iterator<InIter>::value &&
        parallel::traits::is_indirect_callable<
            F, traits::projected<util::projection_identity, InIter>
        >::value)>
    typename util::detail::algorithm_result<ExPolicy, InIter>::type
    for_each(ExPolicy && policy, InIter first, InIter last, F && f)
    {
        return for_each(std::forward<ExPolicy>(policy), first, last,
            std::forward<F>(f), util::projection_identity());
    }
    /// \endcond
}}}

#endif
