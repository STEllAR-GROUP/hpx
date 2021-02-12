//  Copyright (c) 2015 Hartmut Kaiser
//  Copyright (c) 2021 Giannis Gonidelis
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/container_algorithms/remove_copy.hpp

#pragma once

#if defined(DOXYGEN)
namespace hpx {

    /// Copies the elements in the range, defined by [first, last), to another
    /// range beginning at \a dest. Copies only the elements for which the
    /// comparison operator returns false when compare to val.
    /// The order of the elements that are not removed is preserved.
    ///
    /// Effects: Copies all the elements referred to by the iterator it in the
    ///          range [first,last) for which the following corresponding
    ///          conditions do not hold: INVOKE(proj, *it) == value
    ///
    /// \note   Complexity: Performs not more than \a last - \a first
    ///         assignments, exactly \a last - \a first applications of the
    ///         predicate \a f.
    ///
    /// \tparam I           The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
    /// \tparam S           The type of the end iterators used (deduced). This
    ///                     sentinel type must be a sentinel for I.
    /// \tparam O           The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     output iterator.
    /// \tparam T           The type that the result of dereferencing InIter is
    ///                     compared to.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param result       Refers to the beginning of the destination range.
    /// \param val          Value to be removed.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a remove_copy algorithm
    /// execute in sequential order in the calling thread.
    ///
    /// \returns  The \a ranges::remove_copy algorithm returns a
    ///           \a ranges::remove_copy_result<I, O>
    ///           The \a ranges::remove_copy algorithm returns an object
    ///           {last, result + N}, where N is the number of
    ///           elements copied.
    ///
    template <typename I, typename S, typename O, typename T,
        typename Proj = hpx::parallel::util::projection_identity>
    ranges::remove_copy_result<I, O> ranges::remove_copy(
        I first, S last, O result, const T& val, Proj proj = {});

    /// Copies the elements in the range, defined by [first, last), to another
    /// range beginning at \a dest. Copies only the elements for which the
    /// comparison operator returns false when compare to val.
    /// The order of the elements that are not removed is preserved.
    ///
    /// Effects: Copies all the elements referred to by the iterator it in the
    ///          range [first,last) for which the following corresponding
    ///          conditions do not hold: INVOKE(proj, *it) == value
    ///
    /// \note   Complexity: Performs not more than \a last - \a first
    ///         assignments, exactly \a last - \a first applications of the
    ///         predicate \a f.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam I           The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     Forward iterator.
    /// \tparam S           The type of the end iterators used (deduced). This
    ///                     sentinel type must be a sentinel for I.
    /// \tparam O           The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     output iterator.
    /// \tparam T           The type that the result of dereferencing InIter is
    ///                     compared to.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param result       Refers to the beginning of the destination range.
    /// \param val          Value to be removed.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a remove_copy algorithm invoked with
    /// an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a remove_copy algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a ranges::remove_copy algorithm returns a
    ///           \a hpx::future<ranges::remove_copy_result<I, O>>
    ///           if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a ranges::remove_copy_result<I, O>
    ///           otherwise.
    ///           The \a ranges::remove_copy algorithm returns an object
    ///           {last, result + N}, where N is the number of
    ///           elements copied.
    ///
    template <typename ExPolicy, typename I, typename S, typename O, typename T,
        typename Proj = hpx::parallel::util::projection_identity>
    ranges::remove_copy_result<I, O> ranges::remove_copy(ExPolicy&& policy,
        I first, S last, O result, const T& val, Proj proj = {});

    /// Copies the elements in the range, defined by rng, to another
    /// range beginning at \a dest. Copies only the elements for which the
    /// comparison operator returns false when compare to val.
    /// The order of the elements that are not removed is preserved.
    ///
    /// Effects: Copies all the elements referred to by the iterator it in the
    ///          range [first,last) for which the following corresponding
    ///          conditions do not hold: INVOKE(proj, *it) == value
    ///
    /// \note   Complexity: Performs not more than \a last - \a first
    ///         assignments, exactly \a last - \a first applications of the
    ///         predicate \a f.
    ///
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an input iterator.
    /// \tparam O           The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     output iterator.
    /// \tparam T           The type that the result of dereferencing InIter is
    ///                     compared to.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
    /// \param val          Value to be removed.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a remove_copy algorithm
    /// execute in sequential order in the calling thread.
    ///
    /// \returns  The \a ranges::remove_copy algorithm returns a
    ///           \a remove_copy_result<
    ///            typename hpx::traits::range_iterator<Rng>::type, O>.
    ///           The \a ranges::remove_copy algorithm returns an object
    ///           {last, result + N}, where N is the number of
    ///           elements copied.
    ///
    ///
    template <typename Rng, typename O, typename T,
        typename Proj = hpx::parallel::util::projection_identity>
    remove_copy_result<typename hpx::traits::range_iterator<Rng>::type, O>
    ranges::remove_copy(Rng&& rng, O dest, T const& val, Proj&& proj = Proj());

    /// Copies the elements in the range, defined by rng, to another
    /// range beginning at \a dest. Copies only the elements for which the
    /// comparison operator returns false when compare to val.
    /// The order of the elements that are not removed is preserved.
    ///
    /// Effects: Copies all the elements referred to by the iterator it in the
    ///          range [first,last) for which the following corresponding
    ///          conditions do not hold: INVOKE(proj, *it) == value
    ///
    /// \note   Complexity: Performs not more than \a last - \a first
    ///         assignments, exactly \a last - \a first applications of the
    ///         predicate \a f.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an input iterator.
    /// \tparam O           The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     output iterator.
    /// \tparam T           The type that the result of dereferencing InIter is
    ///                     compared to.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
    /// \param val          Value to be removed.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a remove_copy algorithm invoked with
    /// an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a remove_copy algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a ranges::remove_copy algorithm returns a
    ///           \a hpx::future<remove_copy_result<
    ///            typename hpx::traits::range_iterator<Rng>::type, O>>
    ///           if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a remove_copy_result<
    ///            typename hpx::traits::range_iterator<Rng>::type, O>
    ///           otherwise.
    ///           The \a ranges::remove_copy algorithm returns an object
    ///           {last, result + N}, where N is the number of
    ///           elements copied.
    ///
    template <typename ExPolicy, typename Rng, typename O, typename T,
        typename Proj = hpx::parallel::util::projection_identity>
    typename parallel::util::detail::algorithm_result<ExPolicy,
        remove_copy_result<typename hpx::traits::range_iterator<Rng>::type,
            O>>::type
    ranges::remove_copy(ExPolicy&& policy, Rng&& rng, O dest, T const& val,
        Proj&& proj = Proj());

    /////////////////////////////////////////////////////////////////////////////
    /// Copies the elements in the range, defined by [first, last), to another
    /// range beginning at \a dest. Copies only the elements for which the
    /// predicate \a pred returns false. The order of the elements that are not
    /// removed is preserved.
    ///
    /// Effects: Copies all the elements referred to by the iterator it in the
    ///          range [first,last) for which the following corresponding
    ///          conditions do not hold:
    ///          INVOKE(pred, INVOKE(proj, *it)) != false.
    ///
    /// \note   Complexity: Performs not more than \a last - \a first
    ///         assignments, exactly \a last - \a first applications of the
    ///         predicate \a f.
    ///
    /// \tparam I           The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     Input iterator.
    /// \tparam S           The type of the end iterators used (deduced). This
    ///                     sentinel type must be a sentinel for I.
    /// \tparam O           The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     output iterator.
    /// \tparam Pred        The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a remove_copy_if requires \a Pred to meet the
    ///                     requirements of \a CopyConstructible.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param result       Refers to the beginning of the destination range.
    /// \param pred         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last).This is an
    ///                     unary predicate which returns \a true for the
    ///                     elements to be removed. The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     bool pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that an object of
    ///                     type \a InIter can be dereferenced and then
    ///                     implicitly converted to Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a remove_copy_if algorithm
    /// execute in sequential order in the calling thread.
    ///
    /// \returns  The \a ranges::remove_copy_if algorithm
    ///           returns \a ranges::remove_copy_if_result<I, O>.
    ///           The \a ranges::remove_copy algorithm returns an object
    ///           {last, result + N}, where N is the number of
    ///           elements copied.
    ///
    template <typename I, typename Sent, typename O, typename Pred,
        typename Proj = hpx::parallel::util::projection_identity>
    ranges::remove_copy_if_result<I, O> ranges::remove_copy_if(
        I first, Sent last, O dest, Pred&& pred, Proj&& proj = Proj());

    /////////////////////////////////////////////////////////////////////////////
    /// Copies the elements in the range, defined by [first, last), to another
    /// range beginning at \a dest. Copies only the elements for which the
    /// predicate \a pred returns false. The order of the elements that are not
    /// removed is preserved.
    ///
    /// Effects: Copies all the elements referred to by the iterator it in the
    ///          range [first,last) for which the following corresponding
    ///          conditions do not hold:
    ///          INVOKE(pred, INVOKE(proj, *it)) != false.
    ///
    /// \note   Complexity: Performs not more than \a last - \a first
    ///         assignments, exactly \a last - \a first applications of the
    ///         predicate \a f.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam I           The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     Forward iterator.
    /// \tparam S           The type of the end iterators used (deduced). This
    ///                     sentinel type must be a sentinel for I.
    /// \tparam O           The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     output iterator.
    /// \tparam Pred        The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a remove_copy_if requires \a Pred to meet the
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
    /// \param result       Refers to the beginning of the destination range.
    /// \param pred         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last).This is an
    ///                     unary predicate which returns \a true for the
    ///                     elements to be removed. The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     bool pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that an object of
    ///                     type \a InIter can be dereferenced and then
    ///                     implicitly converted to Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a remove_copy_if algorithm invoked with
    /// an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a remove_copy_if algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a ranges::remove_copy_if algorithm returns a
    ///           \a hpx::future<ranges::remove_copy_if_result<I, O>>
    ///           if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a ranges::remove_copy_if_result<I, O>
    ///           otherwise.
    ///           The \a ranges::remove_copy algorithm returns an object
    ///           {last, result + N}, where N is the number of
    ///           elements copied.
    ///
    template <typename ExPolicy, typename I, typename Sent, typename O,
        typename Pred, typename Proj = hpx::parallel::util::projection_identity>
    typename parallel::util::detail::algorithm_result<ExPolicy,
        remove_copy_if_result<I, O>>::type
    ranges::remove_copy_if(ExPolicy&& policy, I first, Sent last, O dest,
        Pred&& pred, Proj&& proj = Proj());

    /////////////////////////////////////////////////////////////////////////////
    /// Copies the elements in the range, defined by rng, to another
    /// range beginning at \a dest. Copies only the elements for which the
    /// predicate \a pred returns false. The order of the elements that are not
    /// removed is preserved.
    ///
    /// Effects: Copies all the elements referred to by the iterator it in the
    ///          range rng for which the following corresponding
    ///          conditions do not hold:
    ///          INVOKE(pred, INVOKE(proj, *it)) != false.
    ///
    /// \note   Complexity: Performs not more than \a last - \a first
    ///         assignments, exactly \a last - \a first applications of the
    ///         predicate \a pred.
    ///
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an input iterator.
    /// \tparam O           The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     output iterator.
    /// \tparam Pred        The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a copy_if requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
    /// \param pred         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last).This is an
    ///                     unary predicate which returns \a true for the
    ///                     elements to be removed. The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     bool pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that an object of
    ///                     type \a InIter can be dereferenced and then
    ///                     implicitly converted to Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a remove_copy_if algorithm
    /// execute in sequential order in the calling thread.
    ///
    /// \returns  The \a ranges::remove_copy_if algorithm returns a
    ///           \a remove_copy_if_result<
    ///             typename hpx::traits::range_iterator<Rng>::type, O>>.
    ///           The \a ranges::remove_copy algorithm returns an object
    ///           {last, result + N}, where N is the number of
    ///           elements copied.
    ///
    template <typename Rng, typename O, typename Pred,
        typename Proj = hpx::parallel::util::projection_identity>
    remove_copy_if_result<typename hpx::traits::range_iterator<Rng>::type, O>
    ranges::remove_copy_if(
        Rng&& rng, O dest, Pred&& pred, Proj&& proj = Proj());

    /////////////////////////////////////////////////////////////////////////////
    /// Copies the elements in the range, defined by rng, to another
    /// range beginning at \a dest. Copies only the elements for which the
    /// predicate \a pred returns false. The order of the elements that are not
    /// removed is preserved.
    ///
    /// Effects: Copies all the elements referred to by the iterator it in the
    ///          range rng for which the following corresponding
    ///          conditions do not hold:
    ///          INVOKE(pred, INVOKE(proj, *it)) != false.
    ///
    /// \note   Complexity: Performs not more than \a last - \a first
    ///         assignments, exactly \a last - \a first applications of the
    ///         predicate \a pred.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an input iterator.
    /// \tparam O           The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     output iterator.
    /// \tparam Pred        The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a copy_if requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
    /// \param pred         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last).This is an
    ///                     unary predicate which returns \a true for the
    ///                     elements to be removed. The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     bool pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that an object of
    ///                     type \a InIter can be dereferenced and then
    ///                     implicitly converted to Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a remove_copy_if algorithm invoked with
    /// an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a remove_copy_if algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a ranges::remove_copy_if algorithm returns a
    ///           \a hpx::future<remove_copy_if_result<
    ///             typename hpx::traits::range_iterator<Rng>::type, O>>
    ///           if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a remove_copy_if_result<
    ///             typename hpx::traits::range_iterator<Rng>::type, O>>
    ///           otherwise.
    ///           The \a ranges::remove_copy algorithm returns an object
    ///           {last, result + N}, where N is the number of
    ///           elements copied.
    ///
    template <typename ExPolicy, typename Rng, typename O, typename Pred,
        typename Proj = hpx::parallel::util::projection_identity>
    typename parallel::util::detail::algorithm_result<ExPolicy,
        remove_copy_if_result<typename hpx::traits::range_iterator<Rng>::type,
            O>>::type
    ranges::remove_copy_if(ExPolicy&& policy, Rng&& rng, O dest, Pred&& pred,
        Proj&& proj = Proj());

}    // namespace hpx

#else    // DOXYGEN

#include <hpx/config.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/iterator_support/range.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/iterator_support/traits/is_range.hpp>
#include <hpx/parallel/util/tagged_pair.hpp>

#include <hpx/algorithms/traits/projected.hpp>
#include <hpx/algorithms/traits/projected_range.hpp>
#include <hpx/parallel/algorithms/remove_copy.hpp>
#include <hpx/parallel/tagspec.hpp>
#include <hpx/parallel/util/projection_identity.hpp>
#include <hpx/parallel/util/result_types.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { inline namespace v1 {

    template <typename ExPolicy, typename Rng, typename OutIter, typename T,
        typename Proj = util::projection_identity,
        HPX_CONCEPT_REQUIRES_(hpx::is_execution_policy<ExPolicy>::value&&
                hpx::traits::is_range<Rng>::value&& hpx::traits::is_iterator<
                    OutIter>::value&& traits::is_projected_range<Proj,
                    Rng>::value&& traits::is_indirect_callable<ExPolicy,
                    std::equal_to<T>, traits::projected_range<Proj, Rng>,
                    traits::projected<Proj, T const*>>::value)>
    typename util::detail::algorithm_result<ExPolicy,
        util::in_out_result<
            typename hpx::traits::range_traits<Rng>::iterator_type,
            OutIter>>::type
    remove_copy(ExPolicy&& policy, Rng&& rng, OutIter dest, T const& val,
        Proj&& proj = Proj())
    {
        return remove_copy(std::forward<ExPolicy>(policy),
            hpx::util::begin(rng), hpx::util::end(rng), dest, val,
            std::forward<Proj>(proj));
    }

    template <typename ExPolicy, typename Rng, typename OutIter, typename F,
        typename Proj = util::projection_identity,
        HPX_CONCEPT_REQUIRES_(hpx::is_execution_policy<ExPolicy>::value&&
                hpx::traits::is_range<Rng>::value&& hpx::traits::is_iterator<
                    OutIter>::value&& traits::is_projected_range<Proj,
                    Rng>::value&& traits::is_indirect_callable<ExPolicy, F,
                    traits::projected_range<Proj, Rng>>::value)>
    typename util::detail::algorithm_result<ExPolicy,
        util::in_out_result<
            typename hpx::traits::range_traits<Rng>::iterator_type,
            OutIter>>::type
    remove_copy_if(
        ExPolicy&& policy, Rng&& rng, OutIter dest, F&& f, Proj&& proj = Proj())
    {
        return remove_copy_if(std::forward<ExPolicy>(policy),
            hpx::util::begin(rng), hpx::util::end(rng), dest,
            std::forward<F>(f), std::forward<Proj>(proj));
    }
}}}    // namespace hpx::parallel::v1

namespace hpx { namespace ranges {
    template <typename I, typename O>
    using remove_copy_result = hpx::parallel::util::in_out_result<I, O>;

    template <typename I, typename O>
    using remove_copy_if_result = hpx::parallel::util::in_out_result<I, O>;

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::ranges::remove_copy_if
    HPX_INLINE_CONSTEXPR_VARIABLE struct remove_copy_if_t final
      : hpx::functional::tag<remove_copy_if_t>
    {
        // clang-format off
        template <typename I, typename Sent, typename O, typename Pred,
            typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator<I>::value &&
                hpx::parallel::traits::is_projected<Proj, I>::value &&
                hpx::traits::is_sentinel_for<Sent, I>::value &&
                hpx::traits::is_iterator<O>::value &&
                hpx::is_invocable_v<Pred,
                    typename std::iterator_traits<I>::value_type
                >
            )>
        // clang-format on
        friend remove_copy_if_result<I, O> tag_invoke(
            hpx::ranges::remove_copy_if_t, I first, Sent last, O dest,
            Pred&& pred, Proj&& proj = Proj())
        {
            static_assert((hpx::traits::is_input_iterator<I>::value),
                "Required input iterator.");

            static_assert((hpx::traits::is_output_iterator<O>::value),
                "Required output iterator.");

            return hpx::parallel::v1::detail::remove_copy_if<
                hpx::parallel::util::in_out_result<I, O>>()
                .call(hpx::execution::seq, std::true_type{}, first, last, dest,
                    std::forward<Pred>(pred), std::forward<Proj>(proj));
        }

        // clang-format off
        template <typename Rng, typename O, typename Pred,
            typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_range<Rng>::value&&
                hpx::parallel::traits::is_projected_range<Proj,Rng>::value &&
                hpx::is_invocable_v<Pred,
                    typename std::iterator_traits<
                        typename hpx::traits::range_iterator<Rng>::type
                    >::value_type
                >
            )>
        // clang-format on
        friend remove_copy_if_result<
            typename hpx::traits::range_iterator<Rng>::type, O>
        tag_invoke(hpx::ranges::remove_copy_if_t, Rng&& rng, O dest,
            Pred&& pred, Proj&& proj = Proj())
        {
            static_assert(
                (hpx::traits::is_input_iterator<
                    typename hpx::traits::range_iterator<Rng>::type>::value),
                "Required at least input iterator.");

            return hpx::parallel::v1::detail::remove_copy_if<
                hpx::parallel::util::in_out_result<
                    typename hpx::traits::range_iterator<Rng>::type, O>>()
                .call(hpx::execution::seq, std::true_type{},
                    hpx::util::begin(rng), hpx::util::end(rng), dest,
                    std::forward<Pred>(pred), std::forward<Proj>(proj));
        }

        // clang-format off
        template <typename ExPolicy, typename I, typename Sent, typename O,
         typename Pred, typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value&&
                hpx::traits::is_iterator<I>::value &&
                hpx::traits::is_sentinel_for<Sent, I>::value &&
                hpx::traits::is_iterator<O>::value &&
                hpx::parallel::traits::is_projected<Proj, I>::value &&
                hpx::is_invocable_v<Pred,
                    typename std::iterator_traits<I>::value_type
                >
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            remove_copy_if_result<I, O>>::type
        tag_invoke(hpx::ranges::remove_copy_if_t, ExPolicy&& policy, I first,
            Sent last, O dest, Pred&& pred, Proj&& proj = Proj())
        {
            static_assert((hpx::traits::is_forward_iterator<I>::value),
                "Required at least forward iterator.");

            static_assert((hpx::traits::is_forward_iterator<O>::value),
                "Required at least forward iterator.");

            typedef hpx::is_sequenced_execution_policy<ExPolicy> is_seq;

            return hpx::parallel::v1::detail::remove_copy_if<
                hpx::parallel::util::in_out_result<I, O>>()
                .call(std::forward<ExPolicy>(policy), is_seq(), first, last,
                    dest, std::forward<Pred>(pred), std::forward<Proj>(proj));
        }

        // clang-format off
        template <typename ExPolicy, typename Rng, typename O, typename Pred,
            typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                parallel::execution::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_range<Rng>::value &&
                hpx::parallel::traits::is_projected_range<Proj, Rng>::value &&
                hpx::is_invocable_v<Pred,
                    typename std::iterator_traits<
                        typename hpx::traits::range_iterator<Rng>::type
                    >::value_type
                >
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            remove_copy_if_result<
                typename hpx::traits::range_iterator<Rng>::type, O>>::type
        tag_invoke(hpx::ranges::remove_copy_if_t, ExPolicy&& policy, Rng&& rng,
            O dest, Pred&& pred, Proj&& proj = Proj())
        {
            static_assert(
                (hpx::traits::is_forward_iterator<
                    typename hpx::traits::range_iterator<Rng>::type>::value),
                "Required at least forward iterator.");

            typedef hpx::is_sequenced_execution_policy<ExPolicy> is_seq;

            return hpx::parallel::v1::detail::remove_copy_if<
                hpx::parallel::util::in_out_result<
                    typename hpx::traits::range_iterator<Rng>::type, O>>()
                .call(std::forward<ExPolicy>(policy), is_seq(),
                    hpx::util::begin(rng), hpx::util::end(rng), dest,
                    std::forward<Pred>(pred), std::forward<Proj>(proj));
        }
    } remove_copy_if{};

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::ranges::remove_copy
    HPX_INLINE_CONSTEXPR_VARIABLE struct remove_copy_t final
      : hpx::functional::tag<remove_copy_t>
    {
    private:
        // clang-format off
        template <typename I, typename Sent, typename O,
        typename T, typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator<I>::value &&
                hpx::traits::is_sentinel_for<Sent, I>::value &&
                hpx::traits::is_iterator<O>::value &&
                hpx::parallel::traits::is_projected<Proj, I>::value
            )>
        // clang-format on
        friend remove_copy_result<I, O> tag_invoke(hpx::ranges::remove_copy_t,
            I first, Sent last, O dest, T const& value, Proj&& proj = Proj())
        {
            typedef typename std::iterator_traits<I>::value_type Type;

            return hpx::ranges::remove_copy_if(
                first, last, dest,
                [value](Type const& a) -> bool { return value == a; },
                std::forward<Proj>(proj));
        }

        // clang-format off
        template <typename Rng, typename O, typename T,
            typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_range<Rng>::value &&
                hpx::parallel::traits::is_projected_range<Proj, Rng>::value
            )>
        // clang-format on
        friend remove_copy_result<
            typename hpx::traits::range_iterator<Rng>::type, O>
        tag_invoke(hpx::ranges::remove_copy_t, Rng&& rng, O dest,
            T const& value, Proj&& proj = Proj())
        {
            typedef typename std::iterator_traits<
                typename hpx::traits::range_iterator<Rng>::type>::value_type
                Type;

            return hpx::ranges::remove_copy_if(
                std::forward<Rng>(rng), dest,
                [value](Type const& a) -> bool { return value == a; },
                std::forward<Proj>(proj));
        }

        // clang-format off
        template <typename ExPolicy, typename I, typename Sent, typename O,
         typename T, typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value&&
                hpx::traits::is_iterator<I>::value &&
                hpx::traits::is_sentinel_for<Sent, I>::value &&
                hpx::traits::is_iterator<O>::value &&
                hpx::parallel::traits::is_projected<Proj, I>::value
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            remove_copy_result<I, O>>::type
        tag_invoke(hpx::ranges::remove_copy_t, ExPolicy&& policy, I first,
            Sent last, O dest, T const& value, Proj&& proj = Proj())
        {
            typedef typename std::iterator_traits<I>::value_type Type;

            return hpx::ranges::remove_copy_if(
                std::forward<ExPolicy>(policy), first, last, dest,
                [value](Type const& a) -> bool { return value == a; },
                std::forward<Proj>(proj));
        }

        // clang-format off
        template <typename ExPolicy, typename Rng, typename O, typename T,
            typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                parallel::execution::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_range<Rng>::value &&
                hpx::parallel::traits::is_projected_range<Proj, Rng>::value
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            remove_copy_result<typename hpx::traits::range_iterator<Rng>::type,
                O>>::type
        tag_invoke(hpx::ranges::remove_copy_t, ExPolicy&& policy, Rng&& rng,
            O dest, T const& value, Proj&& proj = Proj())
        {
            typedef typename std::iterator_traits<
                typename hpx::traits::range_iterator<Rng>::type>::value_type
                Type;

            return hpx::ranges::remove_copy_if(
                std::forward<ExPolicy>(policy), std::forward<Rng>(rng), dest,
                [value](Type const& a) -> bool { return value == a; },
                std::forward<Proj>(proj));
        }

    } remove_copy{};
}}    // namespace hpx::ranges

#endif    // DOXYGEN
