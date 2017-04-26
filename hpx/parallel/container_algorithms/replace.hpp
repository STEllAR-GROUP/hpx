//  Copyright (c) 2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/container_algorithms/replace.hpp

#if !defined(HPX_PARALLEL_CONTAINER_ALGORITHM_REPLACE_DEC_18_2015_0103PM)
#define HPX_PARALLEL_CONTAINER_ALGORITHM_REPLACE_DEC_18_2015_0103PM

#include <hpx/config.hpp>
#include <hpx/traits/concepts.hpp>
#include <hpx/util/tagged_pair.hpp>

#include <hpx/parallel/algorithms/replace.hpp>
#include <hpx/parallel/tagspec.hpp>
#include <hpx/parallel/traits/is_range.hpp>
#include <hpx/parallel/traits/projected_range.hpp>
#include <hpx/parallel/traits/range_traits.hpp>
#include <hpx/parallel/util/projection_identity.hpp>

#include <boost/range/functions.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
    /// Replaces all elements satisfying specific criteria with \a new_value
    /// in the range [first, last).
    ///
    /// \note   Complexity: Performs exactly \a last - \a first assignments.
    ///
    /// Effects: Substitutes elements referred by the iterator it in the range
    ///          [first,last) with new_value, when the following corresponding
    ///          conditions hold: INVOKE(proj, *i) == old_value
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of a forward iterator.
    /// \tparam T1          The type of the old value to replace (deduced).
    /// \tparam T2          The type of the new values to replace (deduced).
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param old_value    Refers to the old value of the elements to replace.
    /// \param new_value    Refers to the new value to use as the replacement.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a replace algorithm invoked with an
    /// execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a replace algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a replace algorithm returns a \a hpx::future<void> if
    ///           the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a void otherwise.
    ///
    template <typename ExPolicy, typename Rng, typename T1, typename T2,
        typename Proj = util::projection_identity,
    HPX_CONCEPT_REQUIRES_(
        execution::is_execution_policy<ExPolicy>::value &&
        traits::is_range<Rng>::value &&
        traits::is_projected_range<Proj, Rng>::value &&
        traits::is_indirect_callable<
            ExPolicy, std::equal_to<T1>,
                traits::projected_range<Proj, Rng>,
                traits::projected<Proj, T1 const*>
        >::value)>
    typename util::detail::algorithm_result<
        ExPolicy, typename traits::range_traits<Rng>::iterator_type
    >::type
    replace(ExPolicy && policy, Rng && rng, T1 const& old_value,
        T2 const& new_value, Proj && proj = Proj())
    {
        return replace(std::forward<ExPolicy>(policy),
            boost::begin(rng), boost::end(rng), old_value, new_value,
            std::forward<Proj>(proj));
    }

    /// Replaces all elements satisfying specific criteria (for which predicate
    /// \a f returns true) with \a new_value in the range [first, last).
    ///
    /// \note   Complexity: Performs exactly \a last - \a first applications of
    ///         the predicate.
    ///
    /// Effects: Substitutes elements referred by the iterator it in the range
    ///          [first, last) with new_value, when the following corresponding
    ///          conditions hold: INVOKE(f, INVOKE(proj, *it)) != false
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of a forward iterator.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a equal requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    ///                     (deduced).
    /// \tparam T           The type of the new values to replace (deduced).
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param f            Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last).This is an
    ///                     unary predicate which returns \a true for the
    ///                     elements which need to replaced. The
    ///                     signature of this predicate should be equivalent
    ///                     to:
    ///                     \code
    ///                     bool pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that an object of
    ///                     type \a FwdIter can be dereferenced and then
    ///                     implicitly converted to \a Type.
    /// \param new_value    Refers to the new value to use as the replacement.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a replace_if algorithm invoked with an
    /// execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a replace_if algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a replace_if algorithm returns a \a hpx::future<FwdIter>
    ///           if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy
    ///           and returns \a void otherwise.
    ///           It returns \a last.
    ///
    template <typename ExPolicy, typename Rng, typename F, typename T,
        typename Proj = util::projection_identity,
    HPX_CONCEPT_REQUIRES_(
        execution::is_execution_policy<ExPolicy>::value &&
        traits::is_range<Rng>::value &&
        traits::is_projected_range<Proj, Rng>::value &&
        traits::is_indirect_callable<
            ExPolicy, F, traits::projected_range<Proj, Rng>
        >::value)>
    typename util::detail::algorithm_result<
        ExPolicy, typename traits::range_traits<Rng>::iterator_type
    >::type
    replace_if(ExPolicy && policy, Rng && rng, F && f, T const& new_value,
        Proj && proj = Proj())
    {
        return replace_if(std::forward<ExPolicy>(policy),
            boost::begin(rng), boost::end(rng),
            std::forward<F>(f), new_value, std::forward<Proj>(proj));
    }

    /// Copies the all elements from the range [first, last) to another range
    /// beginning at \a dest replacing all elements satisfying a specific
    /// criteria with \a new_value.
    ///
    /// Effects: Assigns to every iterator it in the range
    ///          [result, result + (last - first)) either new_value or
    ///          *(first + (it - result)) depending on whether the following
    ///          corresponding condition holds:
    ///          INVOKE(proj, *(first + (i - result))) == old_value
    ///
    /// \note   Complexity: Performs exactly \a last - \a first applications of
    ///         the predicate.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an input iterator.
    /// \tparam OutIter     The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     output iterator.
    /// \tparam T1          The type of the old value to replace (deduced).
    /// \tparam T2          The type of the new values to replace (deduced).
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
    /// \param old_value    Refers to the old value of the elements to replace.
    /// \param new_value    Refers to the new value to use as the replacement.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a replace_copy algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a replace_copy algorithm invoked
    /// with an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a replace_copy algorithm returns a
    ///           \a hpx::future<tagged_pair<tag::in(InIter), tag::out(OutIter)> >
    ///           if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a tagged_pair<tag::in(InIter), tag::out(OutIter)>
    ///           otherwise.
    ///           The \a copy algorithm returns the pair of the input iterator
    ///           \a last and the output iterator to the
    ///           element in the destination range, one past the last element
    ///           copied.
    ///
    template <typename ExPolicy, typename Rng, typename OutIter, typename T1,
        typename T2, typename Proj = util::projection_identity,
    HPX_CONCEPT_REQUIRES_(
        execution::is_execution_policy<ExPolicy>::value &&
        traits::is_range<Rng>::value &&
        traits::is_projected_range<Proj, Rng>::value &&
        traits::is_indirect_callable<
            ExPolicy, std::equal_to<T1>,
                traits::projected_range<Proj, Rng>,
                traits::projected<Proj, T1 const*>
        >::value)>
    typename util::detail::algorithm_result<
        ExPolicy,
        hpx::util::tagged_pair<
            tag::in(typename traits::range_traits<Rng>::iterator_type),
            tag::out(OutIter)
        >
    >::type
    replace_copy(ExPolicy && policy, Rng && rng, OutIter dest,
        T1 const& old_value, T2 const& new_value, Proj && proj = Proj())
    {
        return replace_copy(std::forward<ExPolicy>(policy),
            boost::begin(rng), boost::end(rng), dest, old_value, new_value,
            std::forward<Proj>(proj));
    }

    /// Copies the all elements from the range [first, last) to another range
    /// beginning at \a dest replacing all elements satisfying a specific
    /// criteria with \a new_value.
    ///
    /// Effects: Assigns to every iterator it in the range
    ///          [result, result + (last - first)) either new_value or
    ///          *(first + (it - result)) depending on whether the following
    ///          corresponding condition holds:
    ///          INVOKE(f, INVOKE(proj, *(first + (i - result)))) != false
    ///
    /// \note   Complexity: Performs exactly \a last - \a first applications of
    ///         the predicate.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an input iterator.
    /// \tparam OutIter     The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     output iterator.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a equal requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    ///                     (deduced).
    /// \tparam T           The type of the new values to replace (deduced).
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
    /// \param f            Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last).This is an
    ///                     unary predicate which returns \a true for the
    ///                     elements which need to replaced. The
    ///                     signature of this predicate should be equivalent
    ///                     to:
    ///                     \code
    ///                     bool pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that an object of
    ///                     type \a FwdIter can be dereferenced and then
    ///                     implicitly converted to \a Type.
    /// \param new_value    Refers to the new value to use as the replacement.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a replace_copy_if algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a replace_copy_if algorithm invoked
    /// with an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a replace_copy_if algorithm returns a
    ///           \a hpx::future<tagged_pair<tag::in(InIter), tag::out(OutIter)> >
    ///           if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy
    ///           and returns \a tagged_pair<tag::in(InIter), tag::out(OutIter)>
    ///           otherwise.
    ///           The \a replace_copy_if algorithm returns the input iterator
    ///           \a last and the output iterator to the
    ///           element in the destination range, one past the last element
    ///           copied.
    ///
    template <typename ExPolicy, typename Rng, typename OutIter, typename F,
        typename T, typename Proj = util::projection_identity,
    HPX_CONCEPT_REQUIRES_(
        execution::is_execution_policy<ExPolicy>::value &&
        traits::is_range<Rng>::value &&
        traits::is_projected_range<Proj, Rng>::value &&
        traits::is_indirect_callable<
            ExPolicy, F, traits::projected_range<Proj, Rng>
        >::value)>
    typename util::detail::algorithm_result<
        ExPolicy,
        hpx::util::tagged_pair<
            tag::in(typename traits::range_traits<Rng>::iterator_type),
            tag::out(OutIter)
        >
    >::type
    replace_copy_if(ExPolicy && policy, Rng && rng, OutIter dest,
        F && f, T const& new_value, Proj && proj = Proj())
    {
        return replace_copy_if(std::forward<ExPolicy>(policy),
            boost::begin(rng), boost::end(rng), dest, std::forward<F>(f),
            new_value, std::forward<Proj>(proj));
    }
}}}

#endif
