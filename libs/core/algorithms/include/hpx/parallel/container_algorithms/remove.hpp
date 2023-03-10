//  Copyright (c) 2017 Taeguk Kwon
//  Copyright (c) 2021 Giannis Gonidelis
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/container_algorithms/remove.hpp

#pragma once

#if defined(DOXYGEN)
namespace hpx { namespace ranges {
    /// Removes all elements for which predicate \a pred returns true
    /// from the range [first, last) and returns a subrange [ret, last),
    /// where ret is a past-the-end iterator for the new end of the range.
    ///
    /// \note   Complexity: Performs not more than \a last - \a first
    ///         assignments, exactly \a last - \a first applications of
    ///         the predicate \a pred and the projection \a proj.
    ///
    /// \tparam Iter        The type of the source iterators used for the
    ///                     This iterator type must meet the requirements of a
    ///                     forward iterator.
    /// \tparam Sent        The type of the end iterators used (deduced). This
    ///                     sentinel type must be a sentinel for FwdIter.
    /// \tparam Pred        The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a remove_if requires \a Pred to meet the
    ///                     requirements of \a CopyConstructible..
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a hpx::identity
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param sent         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param pred         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last).This is an
    ///                     unary predicate which returns \a true for the
    ///                     required elements. The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     bool pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that an object of
    ///                     type \a FwdIter can be dereferenced and then
    ///                     implicitly converted to Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a remove_if algorithm
    /// execute in sequential order in the calling thread.
    ///
    /// \returns  The \a remove_if algorithm returns a \a
    ///           subrange_t<FwdIter, Sent>.
    ///           The \a remove_if algorithm returns an object {ret, last},
    ///           where ret is a past-the-end iterator for a new
    ///           subrange of the values all in valid but unspecified state.
    ///
    template <typename Iter, typename Sent, typename Pred,
        typename Proj = hpx::identity>
    subrange_t<Iter, Sent> remove_if(
        Iter first, Sent sent, Pred&& pred, Proj&& proj = Proj());

    /// Removes all elements that are equal to \a value from the range
    /// \a rng and and returns a subrange [ret, util::end(rng)), where ret
    /// is a past-the-end iterator for the new end of the range.
    ///
    /// \note   Complexity: Performs not more than \a util::end(rng)
    ///         - \a util::begin(rng) assignments, exactly
    ///         \a util::end(rng) - \a util::begin(rng) applications of
    ///         the operator==() and the projection \a proj.
    ///
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an forward iterator.
    /// \tparam Pred        The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a remove_if requires \a Pred to meet the
    ///                     requirements of \a CopyConstructible.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a hpx::identity
    ///
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param pred         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last).This is an
    ///                     unary predicate which returns \a true for the
    ///                     required elements. The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     bool pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that an object of
    ///                     type \a FwdIter can be dereferenced and then
    ///                     implicitly converted to Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a remove_if algorithm
    /// execute in sequential order in the calling thread.
    ///
    /// \returns  The \a remove_if algorithm returns a \a
    ///           subrange_t<hpx::traits::range_iterator_t<Rng>>.
    ///           The \a remove_if algorithm returns an object {ret, last},
    ///           where ret is a past-the-end iterator for a new
    ///           subrange of the values all in valid but unspecified state.
    ///
    template <typename Rng, typename Pred, typename Proj = hpx::identity>
    subrange_t<hpx::traits::range_iterator_t<Rng>> remove_if(
        Rng&& rng, Pred&& pred, Proj&& proj = Proj());

    /// Removes all elements for which predicate \a pred returns true
    /// from the range [first, last) and returns a subrange [ret, last),
    /// where ret is a past-the-end iterator for the new end of the range.
    ///
    /// \note   Complexity: Performs not more than \a last - \a first
    ///         assignments, exactly \a last - \a first applications of
    ///         the predicate \a pred and the projection \a proj.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter     The type of the source iterators used for the
    ///                     This iterator type must meet the requirements of a
    ///                     forward iterator.
    /// \tparam Sent        The type of the end iterators used (deduced). This
    ///                     sentinel type must be a sentinel for FwdIter.
    /// \tparam Pred        The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a remove_if requires \a Pred to meet the
    ///                     requirements of \a CopyConstructible.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a hpx::identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param sent         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param pred         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last).This is an
    ///                     unary predicate which returns \a true for the
    ///                     required elements. The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     bool pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that an object of
    ///                     type \a FwdIter can be dereferenced and then
    ///                     implicitly converted to Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a remove_if algorithm invoked with
    /// an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a remove_if algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a remove_if algorithm returns a \a
    ///           hpx::future<subrange_t<FwdIter, Sent>>.
    ///           The \a remove_if algorithm returns an object {ret, last},
    ///           where ret is a past-the-end iterator for a new
    ///           subrange of the values all in valid but unspecified state.
    ///
    template <typename ExPolicy, typename FwdIter, typename Sent, typename Pred,
        typename Proj = hpx::identity>
    typename parallel::util::detail::algorithm_result<ExPolicy,
        subrange_t<FwdIter, Sent>>::type
    remove_if(ExPolicy&& policy, FwdIter first, Sent sent, Pred&& pred,
        Proj&& proj = Proj());

    /// Removes all elements that are equal to \a value from the range
    /// \a rng and and returns a subrange [ret, util::end(rng)), where ret
    /// is a past-the-end iterator for the new end of the range.
    ///
    /// \note   Complexity: Performs not more than \a util::end(rng)
    ///         - \a util::begin(rng) assignments, exactly
    ///         \a util::end(rng) - \a util::begin(rng) applications of
    ///         the operator==() and the projection \a proj.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an forward iterator.
    /// \tparam Pred        The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a remove_if requires \a Pred to meet the
    ///                     requirements of \a CopyConstructible.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a hpx::identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param pred         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last).This is an
    ///                     unary predicate which returns \a true for the
    ///                     required elements. The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     bool pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that an object of
    ///                     type \a FwdIter can be dereferenced and then
    ///                     implicitly converted to Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a remove_if algorithm invoked with
    /// an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a remove_if algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a remove_if algorithm returns a \a
    ///           hpx::future<subrange_t<
    ///           hpx::traits::range_iterator_t<Rng>>>
    ///           if the execution policy is of type
    ///           \a sequenced_task_policy or \a parallel_task_policy and
    ///           returns \a FwdIter otherwise.
    ///           The \a remove_if algorithm returns an object {ret, last},
    ///           where ret is a past-the-end iterator for a new
    ///           subrange of the values all in valid but unspecified state.
    ///
    template <typename ExPolicy, typename Rng, typename Pred,
        typename Proj = hpx::identity>
    typename parallel::util::detail::algorithm_result<ExPolicy,
        subrange_t<hpx::traits::range_iterator_t<Rng>>>
    remove_if(ExPolicy&& policy, Rng&& rng, Pred&& pred, Proj&& proj = Proj());

    /// Removes all elements that are equal to \a value from the range
    /// [first, last) and and returns a subrange [ret, last), where ret
    /// is a past-the-end iterator for the new end of the range.
    ///
    /// \note   Complexity: Performs not more than \a last - \a first
    ///         assignments, exactly \a last - \a first applications of
    ///         the operator==() and the projection \a proj.
    ///
    /// \tparam Iter        The type of the source iterators used for the
    ///                     This iterator type must meet the requirements of a
    ///                     forward iterator.
    /// \tparam Sent        The type of the end iterators used (deduced). This
    ///                     sentinel type must be a sentinel for FwdIter.
    /// \tparam T           The type of the value to remove (deduced).
    ///                     This value type must meet the requirements of
    ///                     \a CopyConstructible.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a hpx::identity
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param value        Specifies the value of elements to remove.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a remove algorithm
    /// execute in sequential order in the calling thread.
    ///
    /// \returns  The \a remove algorithm returns a \a
    ///           subrange_t<FwdIter, Sent>.
    ///           The \a remove algorithm returns an object {ret, last},
    ///           where ret is a past-the-end iterator for a new
    ///           subrange of the values all in valid but unspecified state.
    ///
    template <typename Iter, typename Sent, typename Proj = hpx::identity,
        typename T =
            typename hpx::parallel::traits::projected<Iter, Proj>::value_type>
    subrange_t<Iter, Sent> remove(
        Iter first, Sent last, T const& value, Proj&& proj = Proj());

    /// Removes all elements that are equal to \a value from the range
    /// \a rng and and returns a subrange [ret, util::end(rng)), where ret
    /// is a past-the-end iterator for the new end of the range.
    ///
    /// \note   Complexity: Performs not more than \a util::end(rng)
    ///         - \a util::begin(rng) assignments, exactly
    ///         \a util::end(rng) - \a util::begin(rng) applications of
    ///         the operator==() and the projection \a proj.
    ///
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an forward iterator.
    /// \tparam T           The type of the value to remove (deduced).
    ///                     This value type must meet the requirements of
    ///                     \a CopyConstructible.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a hpx::identity
    ///
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param value        Specifies the value of elements to remove.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a remove algorithm
    /// execute in sequential order in the calling thread.
    ///
    /// \returns  The \a remove algorithm returns a \a
    ///           subrange_t<hpx::traits::range_iterator_t<Rng>>.
    ///           The \a remove algorithm returns an object {ret, last},
    ///           where ret is a past-the-end iterator for a new
    ///           subrange of the values all in valid but unspecified state.
    ///
    template <typename Rng, typename Proj = hpx::identity,
        typename T = typename hpx::parallel::traits::projected<
            hpx::traits::range_iterator_t<Rng>, Proj>::value_type>
    subrange_t<hpx::traits::range_iterator_t<Rng>> remove(
        Rng&& rng, T const& value, Proj&& proj = Proj());

    /// Removes all elements that are equal to \a value from the range
    /// [first, last) and and returns a subrange [ret, last), where ret
    /// is a past-the-end iterator for the new end of the range.
    ///
    /// \note   Complexity: Performs not more than \a last - \a first
    ///         assignments, exactly \a last - \a first applications of
    ///         the operator==() and the projection \a proj.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter     The type of the source iterators used for the
    ///                     This iterator type must meet the requirements of a
    ///                     forward iterator.
    /// \tparam Sent        The type of the end iterators used (deduced). This
    ///                     sentinel type must be a sentinel for FwdIter.
    /// \tparam T           The type of the value to remove (deduced).
    ///                     This value type must meet the requirements of
    ///                     \a CopyConstructible.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a hpx::identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param value        Specifies the value of elements to remove.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a remove algorithm invoked with
    /// an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a remove algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a remove algorithm returns a \a
    ///           hpx::future<subrange_t<FwdIter, Sent>>.
    ///           The \a remove algorithm returns an object {ret, last},
    ///           where ret is a past-the-end iterator for a new
    ///           subrange of the values all in valid but unspecified state.
    ///
    template <typename ExPolicy, typename FwdIter, typename Sent,
        typename Proj = hpx::identity,
        typename T = typename hpx::parallel::traits::projected<FwdIter,
            Proj>::value_type>
    typename parallel::util::detail::algorithm_result<ExPolicy,
        subrange_t<FwdIter, Sent>>::type
    remove(ExPolicy&& policy, FwdIter first, Sent last, T const& value,
        Proj&& proj = Proj());

    /// Removes all elements that are equal to \a value from the range
    /// \a rng and and returns a subrange [ret, util::end(rng)), where ret
    /// is a past-the-end iterator for the new end of the range.
    ///
    /// \note   Complexity: Performs not more than \a util::end(rng)
    ///         - \a util::begin(rng) assignments, exactly
    ///         \a util::end(rng) - \a util::begin(rng) applications of
    ///         the operator==() and the projection \a proj.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an forward iterator.
    /// \tparam T           The type of the value to remove (deduced).
    ///                     This value type must meet the requirements of
    ///                     \a CopyConstructible.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a hpx::identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param value        Specifies the value of elements to remove.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a remove algorithm invoked with
    /// an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a remove algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a remove algorithm returns a \a hpx::future<
    ///           subrange_t<hpx::traits::range_iterator_t<Rng>>>
    ///           if the execution policy is of type
    ///           \a sequenced_task_policy or \a parallel_task_policy and
    ///           returns \a FwdIter otherwise.
    ///           The \a remove algorithm returns the iterator to the new end
    ///           of the range.
    ///
    template <typename ExPolicy, typename Rng, typename Proj = hpx::identity,
        typename T = typename hpx::parallel::traits::projected<
            hpx::traits::range_iterator_t<Rng>, Proj>::value_type>
    typename parallel::util::detail::algorithm_result<ExPolicy,
        subrange_t<hpx::traits::range_iterator_t<Rng>>>
    remove(ExPolicy&& policy, Rng&& rng, T const& value, Proj&& proj = Proj());

}}    // namespace hpx::ranges

#else    // DOXYGEN

#include <hpx/config.hpp>
#include <hpx/algorithms/traits/projected.hpp>
#include <hpx/algorithms/traits/projected_range.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/iterator_support/iterator_range.hpp>
#include <hpx/iterator_support/range.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/iterator_support/traits/is_range.hpp>
#include <hpx/parallel/algorithms/remove.hpp>
#include <hpx/parallel/util/detail/sender_util.hpp>
#include <hpx/parallel/util/result_types.hpp>

#include <type_traits>
#include <utility>

namespace hpx::ranges {

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::ranges::remove_if
    inline constexpr struct remove_if_t final
      : hpx::detail::tag_parallel_algorithm<remove_if_t>
    {
    private:
        // clang-format off
        template <typename Iter, typename Sent, typename Pred,
            typename Proj = hpx::identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator_v<Iter> &&
                hpx::parallel::traits::is_projected_v<Proj, Iter> &&
                hpx::traits::is_sentinel_for_v<Sent, Iter> &&
                hpx::is_invocable_v<Pred,
                    typename std::iterator_traits<Iter>::value_type
                >
            )>
        // clang-format on
        friend subrange_t<Iter, Sent> tag_fallback_invoke(
            hpx::ranges::remove_if_t, Iter first, Sent sent, Pred pred,
            Proj proj = Proj())
        {
            static_assert(hpx::traits::is_input_iterator_v<Iter>,
                "Required at least input iterator.");

            return hpx::parallel::util::make_subrange<Iter, Sent>(
                hpx::parallel::detail::remove_if<Iter>().call(
                    hpx::execution::seq, first, sent, HPX_MOVE(pred),
                    HPX_MOVE(proj)),
                sent);
        }

        // clang-format off
        template <typename Rng, typename Pred,
            typename Proj = hpx::identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_range_v<Rng> &&
                hpx::parallel::traits::is_projected_range_v<Proj, Rng> &&
                hpx::is_invocable_v<Pred,
                    typename std::iterator_traits<
                        hpx::traits::range_iterator_t<Rng>
                    >::value_type
                >
            )>
        // clang-format on
        friend subrange_t<hpx::traits::range_iterator_t<Rng>>
        tag_fallback_invoke(
            hpx::ranges::remove_if_t, Rng&& rng, Pred pred, Proj proj = Proj())
        {
            static_assert(hpx::traits::is_input_iterator<
                              hpx::traits::range_iterator_t<Rng>>::value,
                "Required at least input iterator.");

            return hpx::parallel::util::make_subrange<
                hpx::traits::range_iterator_t<Rng>,
                typename hpx::traits::range_sentinel<Rng>::type>(
                hpx::parallel::detail::remove_if<
                    hpx::traits::range_iterator_t<Rng>>()
                    .call(hpx::execution::seq, hpx::util::begin(rng),
                        hpx::util::end(rng), HPX_MOVE(pred), HPX_MOVE(proj)),
                hpx::util::end(rng));
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter, typename Sent, typename Pred,
        typename Proj = hpx::identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy_v<ExPolicy> &&
                hpx::traits::is_iterator_v<FwdIter> &&
                hpx::traits::is_sentinel_for_v<Sent, FwdIter> &&
                hpx::parallel::traits::is_projected_v<Proj, FwdIter> &&
                hpx::parallel::traits::is_indirect_callable_v<ExPolicy,
                    Pred, hpx::parallel::traits::projected<Proj, FwdIter>
                >
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            subrange_t<FwdIter, Sent>>::type
        tag_fallback_invoke(hpx::ranges::remove_if_t, ExPolicy&& policy,
            FwdIter first, Sent sent, Pred pred, Proj proj = Proj())
        {
            static_assert(hpx::traits::is_forward_iterator_v<FwdIter>,
                "Required at least forward iterator.");

            return hpx::parallel::util::make_subrange<FwdIter, Sent>(
                hpx::parallel::detail::remove_if<FwdIter>().call(
                    HPX_FORWARD(ExPolicy, policy), first, sent, HPX_MOVE(pred),
                    HPX_MOVE(proj)),
                sent);
        }

        // clang-format off
        template <typename ExPolicy, typename Rng, typename Pred,
            typename Proj = hpx::identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy_v<ExPolicy> &&
                hpx::traits::is_range_v<Rng> &&
                hpx::parallel::traits::is_projected_range_v<Proj, Rng> &&
                hpx::parallel::traits::is_indirect_callable_v<ExPolicy,
                    Pred, hpx::parallel::traits::projected_range<Proj, Rng>
                >
            )>
        // clang-format on
        friend parallel::util::detail::algorithm_result_t<ExPolicy,
            subrange_t<hpx::traits::range_iterator_t<Rng>>>
        tag_fallback_invoke(hpx::ranges::remove_if_t, ExPolicy&& policy,
            Rng&& rng, Pred pred, Proj proj = Proj())
        {
            static_assert(hpx::traits::is_forward_iterator<
                              hpx::traits::range_iterator_t<Rng>>::value,
                "Required at least forward iterator.");

            return hpx::parallel::util::make_subrange<
                hpx::traits::range_iterator_t<Rng>,
                typename hpx::traits::range_sentinel<Rng>::type>(
                hpx::parallel::detail::remove_if<
                    hpx::traits::range_iterator_t<Rng>>()
                    .call(HPX_FORWARD(ExPolicy, policy), hpx::util::begin(rng),
                        hpx::util::end(rng), HPX_MOVE(pred), HPX_MOVE(proj)),
                hpx::util::end(rng));
        }
    } remove_if{};

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::ranges::remove
    inline constexpr struct remove_t final
      : hpx::detail::tag_parallel_algorithm<remove_t>
    {
    private:
        // clang-format off
        template <typename Iter, typename Sent,
            typename Proj = hpx::identity,
            typename T = typename hpx::parallel::traits::projected<Iter,
                Proj>::value_type,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator_v<Iter> &&
                hpx::parallel::traits::is_projected_v<Proj, Iter> &&
                hpx::traits::is_sentinel_for_v<Sent, Iter>
            )>
        // clang-format on
        friend subrange_t<Iter, Sent> tag_fallback_invoke(hpx::ranges::remove_t,
            Iter first, Sent last, T const& value, Proj proj = Proj())
        {
            static_assert(hpx::traits::is_input_iterator_v<Iter>,
                "Required at least input iterator.");

            using type = typename std::iterator_traits<Iter>::value_type;

            return hpx::ranges::remove_if(
                first, last,
                [value](type const& a) -> bool { return value == a; },
                HPX_MOVE(proj));
        }

        // clang-format off
        template <typename Rng,
            typename Proj = hpx::identity,
            typename T = typename hpx::parallel::traits::projected<
                hpx::traits::range_iterator_t<Rng>, Proj>::value_type,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_range_v<Rng> &&
                hpx::parallel::traits::is_projected_range_v<Proj, Rng>
            )>
        // clang-format on
        friend subrange_t<hpx::traits::range_iterator_t<Rng>>
        tag_fallback_invoke(hpx::ranges::remove_t, Rng&& rng, T const& value,
            Proj proj = Proj())
        {
            static_assert(hpx::traits::is_input_iterator<
                              hpx::traits::range_iterator_t<Rng>>::value,
                "Required at least input iterator.");

            using type = typename std::iterator_traits<
                hpx::traits::range_iterator_t<Rng>>::value_type;

            return hpx::ranges::remove_if(
                HPX_FORWARD(Rng, rng),
                [value](type const& a) -> bool { return value == a; },
                HPX_MOVE(proj));
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter, typename Sent,
            typename Proj = hpx::identity,
            typename T = typename hpx::parallel::traits::projected<FwdIter,
                Proj>::value_type,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy_v<ExPolicy> &&
                hpx::traits::is_iterator_v<FwdIter> &&
                hpx::traits::is_sentinel_for_v<Sent, FwdIter> &&
                hpx::parallel::traits::is_projected_v<Proj, FwdIter>
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            subrange_t<FwdIter, Sent>>::type
        tag_fallback_invoke(hpx::ranges::remove_t, ExPolicy&& policy,
            FwdIter first, Sent last, T const& value, Proj proj = Proj())
        {
            static_assert(hpx::traits::is_forward_iterator_v<FwdIter>,
                "Required at least forward iterator.");

            using type = typename std::iterator_traits<FwdIter>::value_type;

            return hpx::ranges::remove_if(
                HPX_FORWARD(ExPolicy, policy), first, last,
                [value](type const& a) -> bool { return value == a; },
                HPX_MOVE(proj));
        }

        // clang-format off
        template <typename ExPolicy, typename Rng,
            typename Proj = hpx::identity,
            typename T = typename hpx::parallel::traits::projected<
                hpx::traits::range_iterator_t<Rng>, Proj>::value_type,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy_v<ExPolicy> &&
                hpx::traits::is_range_v<Rng> &&
                hpx::parallel::traits::is_projected_range_v<Proj, Rng>
            )>
        // clang-format on
        friend parallel::util::detail::algorithm_result_t<ExPolicy,
            subrange_t<hpx::traits::range_iterator_t<Rng>>>
        tag_fallback_invoke(hpx::ranges::remove_t, ExPolicy&& policy, Rng&& rng,
            T const& value, Proj proj = Proj())
        {
            static_assert(hpx::traits::is_forward_iterator<
                              hpx::traits::range_iterator_t<Rng>>::value,
                "Required at least forward iterator.");

            using type = typename std::iterator_traits<
                hpx::traits::range_iterator_t<Rng>>::value_type;

            return hpx::ranges::remove_if(
                HPX_FORWARD(ExPolicy, policy), HPX_FORWARD(Rng, rng),
                [value](type const& a) -> bool { return value == a; },
                HPX_MOVE(proj));
        }
    } remove{};
}    // namespace hpx::ranges

#endif    // DOXYGEN
