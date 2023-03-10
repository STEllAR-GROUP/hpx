//  Copyright (c) 2015 Daniel Bourgeois
//  Copyright (c) 2017-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/is_sorted.hpp

#pragma once

#if defined(DOXYGEN)

namespace hpx {
    /// Determines if the range [first, last) is sorted. Uses pred to
    /// compare elements.
    ///
    /// \note   Complexity: at most (N+S-1) comparisons where
    ///         \a N = distance(first, last).
    ///         \a S = number of partitions
    ///
    /// \tparam FwdIter     The type of the source iterators used for the
    ///                     This iterator type must meet the requirements of a
    ///                     forward iterator.
    /// \tparam Pred        The type of an optional function/function object to use.
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     of that the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements of
    ///                     that the algorithm will be applied to.
    /// \param pred         Refers to the binary predicate which returns true
    ///                     if the first argument should be treated as less than
    ///                     the second argument. The signature of the function
    ///                     should be equivalent to
    ///                     \code
    ///                     bool pred(const Type &a, const Type &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that objects of
    ///                     types \a FwdIter can be dereferenced and then
    ///                     implicitly converted to Type.
    ///
    /// The comparison operations in the parallel \a is_sorted algorithm
    /// executes in sequential order in the calling thread.
    ///
    /// \returns  The \a is_sorted algorithm returns a \a bool.
    ///           The \a is_sorted algorithm returns true if each element in
    ///           the sequence [first, last) satisfies the predicate passed.
    ///           If the range [first, last) contains less than two elements,
    ///           the function always returns true.
    ///
    template <typename FwdIter, typename Pred = hpx::parallel::detail::less>
    bool is_sorted(FwdIter first, FwdIter last, Pred&& pred = Pred());

    /// Determines if the range [first, last) is sorted. Uses pred to
    /// compare elements. Executed according to the policy.
    ///
    /// \note   Complexity: at most (N+S-1) comparisons where
    ///         \a N = distance(first, last).
    ///         \a S = number of partitions
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter     The type of the source iterators used for the
    ///                     This iterator type must meet the requirements of a
    ///                     forward iterator.
    /// \tparam Pred        The type of an optional function/function object to use.
    ///                     Unlike its sequential form, the parallel
    ///                     overload of \a is_sorted requires \a Pred to meet the
    ///                     requirements of \a CopyConstructible. This defaults
    ///                     to std::less<>
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     of that the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements of
    ///                     that the algorithm will be applied to.
    /// \param pred         Refers to the binary predicate which returns true
    ///                     if the first argument should be treated as less than
    ///                     the second argument. The signature of the function
    ///                     should be equivalent to
    ///                     \code
    ///                     bool pred(const Type &a, const Type &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that objects of
    ///                     types \a FwdIter can be dereferenced and then
    ///                     implicitly converted to Type.
    ///
    /// The comparison operations in the parallel \a is_sorted algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// executes in sequential order in the calling thread.
    ///
    /// The comparison operations in the parallel \a is_sorted algorithm invoked
    /// with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a is_sorted algorithm returns a \a hpx::future<bool>
    ///           if the execution policy is of type \a task_execution_policy
    ///           and returns \a bool otherwise.
    ///           The \a is_sorted algorithm returns a bool if each element in
    ///           the sequence [first, last) satisfies the predicate passed.
    ///           If the range [first, last) contains less than two elements,
    ///           the function always returns true.
    ///
    template <typename ExPolicy, typename FwdIter,
        typename Pred = hpx::parallel::detail::less>
    hpx::parallel::util::detail::algorithm_result_t<ExPolicy, bool> is_sorted(
        ExPolicy&& policy, FwdIter first, FwdIter last, Pred&& pred = Pred());

    /// Returns the first element in the range [first, last) that is not sorted.
    /// Uses a predicate to compare elements or the less than operator.
    ///
    /// \note   Complexity: at most (N+S-1) comparisons where
    ///         \a N = distance(first, last).
    ///         \a S = number of partitions
    ///
    /// \tparam FwdIter     The type of the source iterators used for the
    ///                     This iterator type must meet the requirements of a
    ///                     forward iterator.
    /// \tparam Pred        The type of an optional function/function object to use.
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     of that the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements of
    ///                     that the algorithm will be applied to.
    /// \param pred         Refers to the binary predicate which returns true
    ///                     if the first argument should be treated as less than
    ///                     the second argument. The signature of the function
    ///                     should be equivalent to
    ///                     \code
    ///                     bool pred(const Type &a, const Type &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that objects of
    ///                     types \a FwdIter can be dereferenced and then
    ///                     implicitly converted to Type.
    ///
    /// The comparison operations in the parallel \a is_sorted_until algorithm
    /// execute in sequential order in the calling thread.
    ///
    /// \returns  The \a is_sorted_until algorithm returns a \a FwdIter.
    ///           The \a is_sorted_until algorithm returns the first unsorted
    ///           element. If the sequence has less than two elements or the
    ///           sequence is sorted, last is returned.
    ///
    template <typename FwdIter, typename Pred = hpx::parallel::detail::less>
    FwdIter is_sorted_until(FwdIter first, FwdIter last, Pred&& pred = Pred());

    /// Returns the first element in the range [first, last) that is not sorted.
    /// Uses a predicate to compare elements or the less than operator.
    /// Executed according to the policy.
    ///
    /// \note   Complexity: at most (N+S-1) comparisons where
    ///         \a N = distance(first, last).
    ///         \a S = number of partitions
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter     The type of the source iterators used for the
    ///                     This iterator type must meet the requirements of a
    ///                     forward iterator.
    /// \tparam Pred        The type of an optional function/function object to use.
    ///                     Unlike its sequential form, the parallel
    ///                     overload of \a is_sorted_until requires \a Pred to meet
    ///                     the requirements of \a CopyConstructible. This defaults
    ///                     to std::less<>
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     of that the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements of
    ///                     that the algorithm will be applied to.
    /// \param pred         Refers to the binary predicate which returns true
    ///                     if the first argument should be treated as less than
    ///                     the second argument. The signature of the function
    ///                     should be equivalent to
    ///                     \code
    ///                     bool pred(const Type &a, const Type &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that objects of
    ///                     types \a FwdIter can be dereferenced and then
    ///                     implicitly converted to Type.
    ///
    /// The comparison operations in the parallel \a is_sorted_until algorithm
    /// invoked with an execution policy object of type
    /// \a sequenced_policy executes in sequential order in the
    /// calling thread.
    ///
    /// The comparison operations in the parallel \a is_sorted_until algorithm
    /// invoked with an execution policy object of type
    /// \a parallel_policy or \a parallel_task_policy are
    /// permitted to execute in an unordered fashion in unspecified threads,
    /// and indeterminately sequenced within each thread.
    ///
    /// \returns  The \a is_sorted_until algorithm returns a \a hpx::future<FwdIter>
    ///           if the execution policy is of type \a task_execution_policy
    ///           and returns \a FwdIter otherwise.
    ///           The \a is_sorted_until algorithm returns the first unsorted
    ///           element. If the sequence has less than two elements or the
    ///           sequence is sorted, last is returned.
    ///
    template <typename ExPolicy, typename FwdIter,
        typename Pred = hpx::parallel::detail::less>
    typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
        FwdIter>::type
    is_sorted_until(
        ExPolicy&& policy, FwdIter first, FwdIter last, Pred&& pred = Pred());
}    // namespace hpx

#else

#include <hpx/config.hpp>
#include <hpx/coroutines/thread_enums.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/functional/invoke.hpp>
#include <hpx/iterator_support/range.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/detail/is_sorted.hpp>
#include <hpx/parallel/util/adapt_placement_mode.hpp>
#include <hpx/parallel/util/cancellation_token.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/detail/clear_container.hpp>
#include <hpx/parallel/util/detail/sender_util.hpp>
#include <hpx/parallel/util/invoke_projected.hpp>
#include <hpx/parallel/util/loop.hpp>
#include <hpx/parallel/util/partitioner.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>

namespace hpx::parallel {

    ////////////////////////////////////////////////////////////////////////////
    // is_sorted
    namespace detail {

        /// \cond NOINTERNAL
        template <typename FwdIter, typename Sent>
        struct is_sorted : public algorithm<is_sorted<FwdIter, Sent>, bool>
        {
            constexpr is_sorted() noexcept
              : algorithm<is_sorted, bool>("is_sorted")
            {
            }

            template <typename ExPolicy, typename Pred, typename Proj>
            static constexpr bool sequential(
                ExPolicy, FwdIter first, Sent last, Pred&& pred, Proj&& proj)
            {
                return is_sorted_sequential(first, last,
                    HPX_FORWARD(Pred, pred), HPX_FORWARD(Proj, proj));
            }

            template <typename ExPolicy, typename Pred, typename Proj>
            static util::detail::algorithm_result_t<ExPolicy, bool> parallel(
                ExPolicy&& policy, FwdIter first, Sent last, Pred&& pred,
                Proj&& proj)
            {
                using difference_type =
                    typename std::iterator_traits<FwdIter>::difference_type;
                using result =
                    typename util::detail::algorithm_result<ExPolicy, bool>;

                difference_type count = std::distance(first, last);
                if (count <= 1)
                    return result::get(true);

                util::invoke_projected<Pred, Proj> pred_projected{
                    HPX_FORWARD(Pred, pred), HPX_FORWARD(Proj, proj)};

                hpx::parallel::util::cancellation_token<> tok;

                // Note: replacing the invoke() with HPX_INVOKE()
                // below makes gcc generate errors
                auto f1 = [tok, last,
                              pred_projected = HPX_MOVE(pred_projected)](
                              FwdIter part_begin,
                              std::size_t part_size) mutable -> bool {
                    FwdIter trail = part_begin++;
                    util::loop_n<std::decay_t<ExPolicy>>(part_begin,
                        part_size - 1,
                        [&trail, &tok, &pred_projected](
                            FwdIter it) mutable -> void {
                            if (hpx::invoke(pred_projected, *it, *trail++))
                            {
                                tok.cancel();
                            }
                        });

                    FwdIter i = trail++;

                    // trail now points one past the current grouping unless
                    // canceled

                    if (!tok.was_cancelled() && trail != last)
                    {
                        return !hpx::invoke(pred_projected, *trail, *i);
                    }

                    return !tok.was_cancelled();
                };

                auto f2 = [](auto&& results) {
                    return std::all_of(hpx::util::begin(results),
                        hpx::util::end(results),
                        [](hpx::future<bool>& val) -> bool {
                            return val.get();
                        });
                };

                return util::partitioner<ExPolicy, bool>::call(
                    HPX_FORWARD(ExPolicy, policy), first, count, HPX_MOVE(f1),
                    HPX_MOVE(f2));
            }
        };
        /// \endcond
    }    // namespace detail

    ////////////////////////////////////////////////////////////////////////////
    // is_sorted_until
    namespace detail {
        /// \cond NOINTERNAL
        template <typename FwdIter, typename Sent>
        struct is_sorted_until
          : public algorithm<is_sorted_until<FwdIter, Sent>, FwdIter>
        {
            constexpr is_sorted_until() noexcept
              : algorithm<is_sorted_until, FwdIter>("is_sorted_until")
            {
            }

            template <typename ExPolicy, typename Pred, typename Proj>
            static constexpr FwdIter sequential(
                ExPolicy, FwdIter first, Sent last, Pred&& pred, Proj&& proj)
            {
                return is_sorted_until_sequential(first, last,
                    HPX_FORWARD(Pred, pred), HPX_FORWARD(Proj, proj));
            }

            template <typename ExPolicy, typename Pred, typename Proj>
            static util::detail::algorithm_result_t<ExPolicy, FwdIter> parallel(
                ExPolicy&& orgpolicy, FwdIter first, Sent last, Pred&& pred,
                Proj&& proj)
            {
                using reference =
                    typename std::iterator_traits<FwdIter>::reference;
                using difference_type =
                    typename std::iterator_traits<FwdIter>::difference_type;
                using result =
                    typename util::detail::algorithm_result<ExPolicy, FwdIter>;

                difference_type count = std::distance(first, last);
                if (count <= 1)
                    return result::get(HPX_MOVE(last));

                util::invoke_projected<Pred, Proj> pred_projected{
                    HPX_FORWARD(Pred, pred), HPX_FORWARD(Proj, proj)};

                decltype(auto) policy = parallel::util::adapt_placement_mode(
                    HPX_FORWARD(ExPolicy, orgpolicy),
                    hpx::threads::thread_placement_hint::breadth_first);

                using policy_type = std::decay_t<decltype(policy)>;

                hpx::parallel::util::cancellation_token<difference_type> tok(
                    count);

                // Note: replacing the invoke() with HPX_INVOKE() below makes
                // gcc generate errors
                auto f1 = [tok, last,
                              pred_projected = HPX_MOVE(pred_projected)](
                              FwdIter part_begin, std::size_t part_size,
                              std::size_t base_idx) mutable -> void {
                    FwdIter trail = part_begin++;
                    util::loop_idx_n<policy_type>(++base_idx, part_begin,
                        part_size - 1, tok,
                        [&trail, &tok, &pred_projected](
                            reference& v, std::size_t ind) -> void {
                            if (hpx::invoke(pred_projected, v, *trail++))
                            {
                                tok.cancel(ind);
                            }
                        });

                    FwdIter i = trail++;

                    // trail now points one past the current grouping unless
                    // canceled

                    if (!tok.was_cancelled(base_idx + part_size) &&
                        trail != last)
                    {
                        if (HPX_INVOKE(pred_projected, *trail, *i))
                        {
                            tok.cancel(base_idx + part_size);
                        }
                    }
                };

                auto f2 = [first, tok](auto&& data) mutable -> FwdIter {
                    // make sure iterators embedded in function object that is
                    // attached to futures are invalidated
                    util::detail::clear_container(data);

                    difference_type loc = tok.get_data();
                    std::advance(first, loc);
                    return HPX_MOVE(first);
                };

                using partitioner_type =
                    util::partitioner<policy_type, FwdIter, void>;
                return partitioner_type::call_with_index(
                    HPX_FORWARD(decltype(policy), policy), first, count, 1,
                    HPX_MOVE(f1), HPX_MOVE(f2));
            }
        };
        /// \endcond
    }    // namespace detail
}    // namespace hpx::parallel

namespace hpx {

    inline constexpr struct is_sorted_t final
      : hpx::detail::tag_parallel_algorithm<is_sorted_t>
    {
    private:
        template <typename FwdIter, typename Pred = hpx::parallel::detail::less,
            // clang-format off
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_forward_iterator_v<FwdIter> &&
                hpx::is_invocable_v<Pred,
                    typename std::iterator_traits<FwdIter>::value_type,
                    typename std::iterator_traits<FwdIter>::value_type
                >
            )>
        // clang-format on
        friend bool tag_fallback_invoke(
            hpx::is_sorted_t, FwdIter first, FwdIter last, Pred pred = Pred())
        {
            return hpx::parallel::detail::is_sorted<FwdIter, FwdIter>().call(
                hpx::execution::seq, first, last, HPX_MOVE(pred),
                hpx::identity_v);
        }

        template <typename ExPolicy, typename FwdIter,
            typename Pred = hpx::parallel::detail::less,
            // clang-format off
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy_v<ExPolicy> &&
                hpx::traits::is_forward_iterator_v<FwdIter> &&
                hpx::is_invocable_v<Pred,
                    typename std::iterator_traits<FwdIter>::value_type,
                    typename std::iterator_traits<FwdIter>::value_type
                >
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            bool>::type
        tag_fallback_invoke(hpx::is_sorted_t, ExPolicy&& policy, FwdIter first,
            FwdIter last, Pred pred = Pred())
        {
            return hpx::parallel::detail::is_sorted<FwdIter, FwdIter>().call(
                HPX_FORWARD(ExPolicy, policy), first, last, HPX_MOVE(pred),
                hpx::identity_v);
        }
    } is_sorted{};

    inline constexpr struct is_sorted_until_t final
      : hpx::detail::tag_parallel_algorithm<is_sorted_until_t>
    {
    private:
        template <typename FwdIter, typename Pred = hpx::parallel::detail::less,
            // clang-format off
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_forward_iterator_v<FwdIter> &&
                hpx::is_invocable_v<Pred,
                    typename std::iterator_traits<FwdIter>::value_type,
                    typename std::iterator_traits<FwdIter>::value_type
                >
            )>
        // clang-format on
        friend FwdIter tag_fallback_invoke(hpx::is_sorted_until_t,
            FwdIter first, FwdIter last, Pred pred = Pred())
        {
            return hpx::parallel::detail::is_sorted_until<FwdIter, FwdIter>()
                .call(hpx::execution::seq, first, last, HPX_MOVE(pred),
                    hpx::identity_v);
        }

        template <typename ExPolicy, typename FwdIter,
            typename Pred = hpx::parallel::detail::less,
            // clang-format off
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy_v<ExPolicy> &&
                hpx::traits::is_forward_iterator_v<FwdIter> &&
                hpx::is_invocable_v<Pred,
                    typename std::iterator_traits<FwdIter>::value_type,
                    typename std::iterator_traits<FwdIter>::value_type
                >
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            FwdIter>::type
        tag_fallback_invoke(hpx::is_sorted_until_t, ExPolicy&& policy,
            FwdIter first, FwdIter last, Pred pred = Pred())
        {
            return hpx::parallel::detail::is_sorted_until<FwdIter, FwdIter>()
                .call(HPX_FORWARD(ExPolicy, policy), first, last,
                    HPX_MOVE(pred), hpx::identity_v);
        }
    } is_sorted_until{};
}    // namespace hpx

#endif
