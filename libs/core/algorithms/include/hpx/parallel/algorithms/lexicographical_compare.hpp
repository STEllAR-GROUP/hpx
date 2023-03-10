//  Copyright (c) 2014 Grant Mercer
//  Copyright (c) 2017-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/lexicographical_compare.hpp

#pragma once

#if defined(DOXYGEN)
namespace hpx {

    /// Checks if the first range [first1, last1) is lexicographically less than
    /// the second range [first2, last2). uses a provided predicate to compare
    /// elements.
    ///
    /// \note   Complexity: At most 2 * min(N1, N2) applications of the comparison
    ///         operation, where N1 = std::distance(first1, last)
    ///         and N2 = std::distance(first2, last2).
    ///
    /// \tparam InIter1     The type of the source iterators used for the
    ///                     first range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
    /// \tparam InIter2     The type of the source iterators used for the
    ///                     second range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
    /// \tparam Pred        The type of an optional function/function object to use.
    ///                     Unlike its sequential form, the parallel
    ///                     overload of \a lexicographical_compare requires \a Pred to
    ///                     meet the requirements of \a CopyConstructible. This defaults
    ///                     to std::less<>
    ///
    /// \param first1       Refers to the beginning of the sequence of elements
    ///                     of the first range the algorithm will be applied to.
    /// \param last1        Refers to the end of the sequence of elements of
    ///                     the first range the algorithm will be applied to.
    /// \param first2       Refers to the beginning of the sequence of elements
    ///                     of the second range the algorithm will be applied to.
    /// \param last2        Refers to the end of the sequence of elements of
    ///                     the second range the algorithm will be applied to.
    /// \param pred         Refers to the comparison function that the first
    ///                     and second ranges will be applied to
    ///
    /// The comparison operations in the parallel \a lexicographical_compare
    /// algorithm invoked without an execution policy object execute in sequential
    /// order in the calling thread.
    ///
    /// \note     Lexicographical comparison is an operation with the
    ///           following properties
    ///             - Two ranges are compared element by element
    ///             - The first mismatching element defines which range
    ///               is lexicographically
    ///               \a less or \a greater than the other
    ///             - If one range is a prefix of another, the shorter range is
    ///               lexicographically \a less than the other
    ///             - If two ranges have equivalent elements and are of the same length,
    ///               then the ranges are lexicographically \a equal
    ///             - An empty range is lexicographically \a less than any non-empty
    ///               range
    ///             - Two empty ranges are lexicographically \a equal
    ///
    /// \returns  The \a lexicographically_compare algorithm returns a
    ///           returns \a bool if the execution policy object is not passed in.
    ///           The \a lexicographically_compare algorithm returns true
    ///           if the first range is lexicographically less, otherwise
    ///           it returns false.
    ///           range [first2, last2), it returns false.
    template <typename InIter1, typename InIter2,
        typename Pred = hpx::parallel::detail::less>
    bool lexicographical_compare(InIter1 first1, InIter1 last1, InIter2 first2,
        InIter2 last2, Pred&& pred);

    /// Checks if the first range [first1, last1) is lexicographically less than
    /// the second range [first2, last2). uses a provided predicate to compare
    /// elements.
    ///
    /// \note   Complexity: At most 2 * min(N1, N2) applications of the comparison
    ///         operation, where N1 = std::distance(first1, last)
    ///         and N2 = std::distance(first2, last2).
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter1    The type of the source iterators used for the
    ///                     first range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam FwdIter2    The type of the source iterators used for the
    ///                     second range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Pred        The type of an optional function/function object to use.
    ///                     Unlike its sequential form, the parallel
    ///                     overload of \a lexicographical_compare requires \a Pred to
    ///                     meet the requirements of \a CopyConstructible. This defaults
    ///                     to std::less<>
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first1       Refers to the beginning of the sequence of elements
    ///                     of the first range the algorithm will be applied to.
    /// \param last1        Refers to the end of the sequence of elements of
    ///                     the first range the algorithm will be applied to.
    /// \param first2       Refers to the beginning of the sequence of elements
    ///                     of the second range the algorithm will be applied to.
    /// \param last2        Refers to the end of the sequence of elements of
    ///                     the second range the algorithm will be applied to.
    /// \param pred         Refers to the comparison function that the first
    ///                     and second ranges will be applied to
    ///
    /// The comparison operations in the parallel \a lexicographical_compare
    /// algorithm invoked with an execution policy object of type
    /// \a sequenced_policy execute in sequential order in the
    /// calling thread.
    ///
    /// The comparison operations in the parallel \a lexicographical_compare
    /// algorithm invoked with an execution policy object of type
    /// \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \note     Lexicographical comparison is an operation with the
    ///           following properties
    ///             - Two ranges are compared element by element
    ///             - The first mismatching element defines which range
    ///               is lexicographically
    ///               \a less or \a greater than the other
    ///             - If one range is a prefix of another, the shorter range is
    ///               lexicographically \a less than the other
    ///             - If two ranges have equivalent elements and are of the same length,
    ///               then the ranges are lexicographically \a equal
    ///             - An empty range is lexicographically \a less than any non-empty
    ///               range
    ///             - Two empty ranges are lexicographically \a equal
    ///
    /// \returns  The \a lexicographically_compare algorithm returns a
    ///           \a hpx::future<bool> if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a bool otherwise.
    ///           The \a lexicographically_compare algorithm returns true
    ///           if the first range is lexicographically less, otherwise
    ///           it returns false.
    ///           range [first2, last2), it returns false.
    template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
        typename Pred = hpx::parallel::detail::less>
    hpx::parallel::util::detail::algorithm_result_t<ExPolicy, bool>
    lexicographical_compare(ExPolicy&& policy, FwdIter1 first1, FwdIter1 last1,
        FwdIter2 first2, FwdIter2 last2, Pred&& pred);

}    // namespace hpx

#else    // DOXYGEN

#include <hpx/config.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/execution/algorithms/detail/predicates.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/functional/invoke.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/mismatch.hpp>
#include <hpx/parallel/util/cancellation_token.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/detail/clear_container.hpp>
#include <hpx/parallel/util/detail/sender_util.hpp>
#include <hpx/parallel/util/loop.hpp>
#include <hpx/parallel/util/partitioner.hpp>
#include <hpx/parallel/util/zip_iterator.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>

namespace hpx::parallel {

    ///////////////////////////////////////////////////////////////////////////
    // lexicographical_compare
    namespace detail {

        /// \cond NOINTERNAL
        struct lexicographical_compare
          : public algorithm<lexicographical_compare, bool>
        {
            constexpr lexicographical_compare() noexcept
              : algorithm("lexicographical_compare")
            {
            }

            template <typename ExPolicy, typename InIter1, typename Sent1,
                typename InIter2, typename Sent2, typename Pred, typename Proj1,
                typename Proj2>
            static constexpr bool sequential(ExPolicy, InIter1 first1,
                Sent1 last1, InIter2 first2, Sent2 last2, Pred&& pred,
                Proj1&& proj1, Proj2&& proj2)
            {
                for (; first1 != last1 && first2 != last2;
                     (void) ++first1, ++first2)
                {
                    if (HPX_INVOKE(pred, HPX_INVOKE(proj1, *first1),
                            HPX_INVOKE(proj2, *first2)))
                    {
                        return true;
                    }
                    if (HPX_INVOKE(pred, HPX_INVOKE(proj2, *first2),
                            HPX_INVOKE(proj1, *first1)))
                    {
                        return false;
                    }
                }
                return first1 == last1 && first2 != last2;
            }

            template <typename ExPolicy, typename FwdIter1, typename Sent1,
                typename FwdIter2, typename Sent2, typename Pred,
                typename Proj1, typename Proj2>
            static util::detail::algorithm_result_t<ExPolicy, bool> parallel(
                ExPolicy&& orgpolicy, FwdIter1 first1, Sent1 last1,
                FwdIter2 first2, Sent2 last2, Pred&& pred, Proj1&& proj1,
                Proj2&& proj2)
            {
                using zip_iterator =
                    hpx::util::zip_iterator<FwdIter1, FwdIter2>;
                using reference = typename zip_iterator::reference;

                std::size_t const count1 = detail::distance(first1, last1);
                std::size_t const count2 = detail::distance(first2, last2);

                // An empty range is lexicographically less than any non-empty
                // range
                if (count1 == 0 && count2 != 0)
                {
                    return util::detail::algorithm_result<ExPolicy, bool>::get(
                        true);
                }

                if (count2 == 0 && count1 != 0)
                {
                    return util::detail::algorithm_result<ExPolicy, bool>::get(
                        false);
                }

                decltype(auto) policy = parallel::util::adapt_placement_mode(
                    HPX_FORWARD(ExPolicy, orgpolicy),
                    hpx::threads::thread_placement_hint::breadth_first);

                using policy_type = std::decay_t<decltype(policy)>;

                std::size_t count = (std::min)(count1, count2);
                hpx::parallel::util::cancellation_token<std::size_t> tok(count);

                auto f1 = [tok, pred, proj1, proj2](zip_iterator it,
                              std::size_t part_count,
                              std::size_t base_idx) mutable -> void {
                    util::loop_idx_n<policy_type>(base_idx, it, part_count, tok,
                        [&pred, &tok, &proj1, &proj2](
                            reference t, std::size_t i) mutable -> void {
                            using hpx::get;
                            using hpx::invoke;
                            // gcc10/cuda11 complains about using HPX_INVOKE here
                            if (invoke(pred, invoke(proj1, get<0>(t)),
                                    invoke(proj2, get<1>(t))) ||
                                invoke(pred, invoke(proj2, get<1>(t)),
                                    invoke(proj1, get<0>(t))))
                            {
                                tok.cancel(i);
                            }
                        });
                };

                auto f2 = [tok, first1, first2, last1, last2, pred, proj1,
                              proj2](auto&& data) mutable -> bool {
                    // make sure iterators embedded in function object that is
                    // attached to futures are invalidated
                    util::detail::clear_container(data);

                    std::size_t mismatched = tok.get_data();

                    std::advance(first1, mismatched);
                    std::advance(first2, mismatched);

                    if (first1 != last1 && first2 != last2)
                    {
                        using hpx::invoke;
                        return invoke(pred, invoke(proj1, *first1),
                            invoke(proj2, *first2));
                    }

                    return first2 != last2;
                };

                using partitioner_type =
                    util::partitioner<policy_type, bool, void>;
                return partitioner_type::call_with_index(
                    HPX_FORWARD(decltype(policy), policy),
                    zip_iterator(first1, first2), count, 1, HPX_MOVE(f1),
                    HPX_MOVE(f2));
            }
        };
        /// \endcond
    }    // namespace detail
}    // namespace hpx::parallel

namespace hpx {

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::lexicographical_compare
    inline constexpr struct lexicographical_compare_t final
      : hpx::detail::tag_parallel_algorithm<lexicographical_compare_t>
    {
        // clang-format off
        template <typename InIter1, typename InIter2,
            typename Pred = hpx::parallel::detail::less,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator_v<InIter1> &&
                hpx::traits::is_iterator_v<InIter2> &&
                hpx::is_invocable_v<Pred,
                    typename std::iterator_traits<InIter1>::value_type,
                    typename std::iterator_traits<InIter2>::value_type
                >
            )>
        // clang-format on
        friend bool tag_fallback_invoke(hpx::lexicographical_compare_t,
            InIter1 first1, InIter1 last1, InIter2 first2, InIter2 last2,
            Pred pred = Pred())
        {
            static_assert(hpx::traits::is_input_iterator_v<InIter1>,
                "Requires at least input iterator.");
            static_assert(hpx::traits::is_input_iterator_v<InIter2>,
                "Requires at least input iterator.");

            return hpx::parallel::detail::lexicographical_compare().call(
                hpx::execution::seq, first1, last1, first2, last2,
                HPX_MOVE(pred), hpx::identity_v, hpx::identity_v);
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
            typename Pred = hpx::parallel::detail::less,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy_v<ExPolicy> &&
                hpx::traits::is_iterator_v<FwdIter1> &&
                hpx::traits::is_iterator_v<FwdIter2> &&
                hpx::is_invocable_v<Pred,
                    typename std::iterator_traits<FwdIter1>::value_type,
                    typename std::iterator_traits<FwdIter2>::value_type
                >
            )>
        // clang-format on
        friend parallel::util::detail::algorithm_result_t<ExPolicy, bool>
        tag_fallback_invoke(hpx::lexicographical_compare_t, ExPolicy&& policy,
            FwdIter1 first1, FwdIter1 last1, FwdIter2 first2, FwdIter2 last2,
            Pred pred = Pred())
        {
            static_assert(hpx::traits::is_forward_iterator_v<FwdIter1>,
                "Requires at least forward iterator.");
            static_assert(hpx::traits::is_forward_iterator_v<FwdIter2>,
                "Requires at least forward iterator.");

            return hpx::parallel::detail::lexicographical_compare().call(
                HPX_FORWARD(ExPolicy, policy), first1, last1, first2, last2,
                HPX_MOVE(pred), hpx::identity_v, hpx::identity_v);
        }
    } lexicographical_compare{};
}    // namespace hpx

#endif    // DOXYGEN
