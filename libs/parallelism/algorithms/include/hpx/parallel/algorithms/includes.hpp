//  Copyright (c) 2007-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/includes.hpp

#pragma once

#if defined(DOXYGEN)
namespace hpx {
    // clang-format off

    /// Returns true if every element from the sorted range [first2, last2) is
    /// found within the sorted range [first1, last1). Also returns true if
    /// [first2, last2) is empty. The version expects both ranges to be sorted
    /// with the user supplied binary predicate \a f.
    ///
    /// \note   At most 2*(N1+N2-1) comparisons, where
    ///         N1 = std::distance(first1, last1) and
    ///         N2 = std::distance(first2, last2).
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
    ///                     overload of \a includes requires \a Pred to meet the
    ///                     requirements of \a CopyConstructible. This defaults
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
    /// \param op           The binary predicate which returns true if the
    ///                     elements should be treated as includes. The signature
    ///                     of the predicate function should be equivalent to
    ///                     the following:
    ///                     \code
    ///                     bool pred(const Type1 &a, const Type2 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The types \a Type1 and \a Type2 must be such
    ///                     that objects of types \a FwdIter1 and \a FwdIter2 can
    ///                     be dereferenced and then implicitly converted to
    ///                     \a Type1 and \a Type2 respectively
    ///
    /// The comparison operations in the parallel \a includes algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The comparison operations in the parallel \a includes algorithm invoked
    /// with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a includes algorithm returns a \a hpx::future<bool> if the
    ///           execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a bool otherwise.
    ///           The \a includes algorithm returns true every element from the
    ///           sorted range [first2, last2) is found within the sorted range
    ///           [first1, last1). Also returns true if [first2, last2) is empty.
    ///
    template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
        typename Pred = detail::less>
    typename util::detail::algorithm_result<ExPolicy, bool>::type>::type
    includes(ExPolicy&& policy, FwdIter1 first1, FwdIter1 last1,
            FwdIter2 first2, FwdIter2 last2, Pred&& op = Pred());

    // clang-format on
}    // namespace hpx

#else    // DOXYGEN

#include <hpx/config.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/functional/invoke.hpp>
#include <hpx/functional/tag_invoke.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>

#include <hpx/execution/algorithms/detail/predicates.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/detail/upper_lower_bound.hpp>
#include <hpx/parallel/util/cancellation_token.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/projection_identity.hpp>
#include <hpx/parallel/util/result_types.hpp>
#include <hpx/parallel/util/partitioner.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace parallel { inline namespace v1 {
    ///////////////////////////////////////////////////////////////////////////
    // includes
    namespace detail {

        template <typename Iter1, typename Sent1, typename Iter2,
            typename Sent2, typename F, typename Proj1, typename Proj2,
            typename CancelToken>
        bool sequential_includes(Iter1 first1, Sent1 last1, Iter2 first2,
            Sent2 last2, F&& f, Proj1&& proj1, Proj2&& proj2, CancelToken& tok)
        {
            while (first2 != last2)
            {
                if (tok.was_cancelled())
                {
                    return false;
                }

                if (first1 == last1)
                {
                    return false;
                }

                auto&& value1 = hpx::util::invoke(proj1, *first1);
                auto&& value2 = hpx::util::invoke(proj2, *first2);

                if (hpx::util::invoke(f, value2, value1))
                {
                    return false;
                }

                if (!hpx::util::invoke(f, value1, value2))
                {
                    ++first2;
                }

                ++first1;
            }
            return true;
        }

        template <typename Iter1, typename Sent1, typename Iter2,
            typename Sent2, typename F, typename Proj1, typename Proj2>
        constexpr bool sequential_includes(Iter1 first1, Sent1 last1,
            Iter2 first2, Sent2 last2, F&& f, Proj1&& proj1, Proj2&& proj2)
        {
            while (first2 != last2)
            {
                if (first1 == last1)
                {
                    return false;
                }

                auto&& value1 = hpx::util::invoke(proj1, *first1);
                auto&& value2 = hpx::util::invoke(proj2, *first2);

                if (hpx::util::invoke(f, value2, value1))
                {
                    return false;
                }

                if (!hpx::util::invoke(f, value1, value2))
                {
                    ++first2;
                }

                ++first1;
            }
            return true;
        }

        ///////////////////////////////////////////////////////////////////////
        struct includes : public detail::algorithm<includes, bool>
        {
            includes()
              : includes::algorithm("includes")
            {
            }

            template <typename ExPolicy, typename Iter1, typename Sent1,
                typename Iter2, typename Sent2, typename F, typename Proj1,
                typename Proj2>
            static bool sequential(ExPolicy, Iter1 first1, Sent1 last1,
                Iter2 first2, Sent2 last2, F&& f, Proj1&& proj1, Proj2&& proj2)
            {
                return sequential_includes(first1, last1, first2, last2,
                    std::forward<F>(f), std::forward<Proj1>(proj1),
                    std::forward<Proj2>(proj2));
            }

            template <typename ExPolicy, typename Iter1, typename Sent1,
                typename Iter2, typename Sent2, typename F, typename Proj1,
                typename Proj2>
            static typename util::detail::algorithm_result<ExPolicy, bool>::type
            parallel(ExPolicy&& policy, Iter1 first1, Sent1 last1, Iter2 first2,
                Sent2 last2, F&& f, Proj1&& proj1, Proj2&& proj2)
            {
                if (first1 == last1)
                {
                    return util::detail::algorithm_result<ExPolicy, bool>::get(
                        false);
                }
                if (first2 == last2)
                {
                    return util::detail::algorithm_result<ExPolicy, bool>::get(
                        true);
                }

                util::cancellation_token<> tok;

                auto f1 =
                    [first1, last1, first2, last2, tok, f = std::forward<F>(f),
                        proj1 = std::forward<Proj1>(proj1),
                        proj2 = std::forward<Proj2>(proj2)](Iter2 part_begin,
                        std::size_t part_count) mutable -> bool {
                    Iter2 part_end = detail::next(part_begin, part_count);

                    auto value = hpx::util::invoke(proj2, *part_begin);
                    if (first2 != part_begin && part_count > 1)
                    {
                        part_begin = detail::upper_bound(
                            part_begin, part_end, value, f, proj2, tok);
                        if (tok.was_cancelled())
                        {
                            return false;
                        }
                        if (part_begin == part_end)
                        {
                            return true;
                        }
                        value = hpx::util::invoke(proj2, *part_begin);
                    }

                    Iter1 low = detail::lower_bound(
                        first1, last1, value, f, proj1, tok);
                    if (tok.was_cancelled())
                    {
                        return false;
                    }

                    if (low == last1 ||
                        hpx::util::invoke(
                            f, value, hpx::util::invoke(proj1, *low)))
                    {
                        tok.cancel();
                        return false;
                    }

                    Iter1 high = last1;
                    if (part_end != last2)
                    {
                        auto&& value1 = hpx::util::invoke(proj2, *part_end);

                        high = detail::upper_bound(
                            low, last1, value1, f, proj1, tok);
                        part_end = detail::upper_bound(
                            part_end, last2, value1, f, proj2, tok);

                        if (tok.was_cancelled())
                        {
                            return false;
                        }
                    }

                    if (!sequential_includes(low, high, part_begin, part_end, f,
                            proj1, proj2, tok))
                    {
                        tok.cancel();
                    }
                    return !tok.was_cancelled();
                };

                auto f2 = [](std::vector<hpx::future<bool>>&& results) {
                    return std::all_of(hpx::util::begin(results),
                        hpx::util::end(results),
                        [](hpx::future<bool>& val) { return val.get(); });
                };

                return util::partitioner<ExPolicy, bool>::call(
                    std::forward<ExPolicy>(policy), first2,
                    detail::distance(first2, last2), std::move(f1),
                    std::move(f2));
            }
        };
    }    // namespace detail

    // clang-format off
    template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
        typename Pred = detail::less,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy<ExPolicy>::value &&
            hpx::traits::is_iterator<FwdIter1>::value &&
            hpx::traits::is_iterator<FwdIter2>::value &&
            hpx::traits::is_invocable<Pred,
                typename std::iterator_traits<FwdIter1>::value_type,
                typename std::iterator_traits<FwdIter2>::value_type
            >::value
        )>
    // clang-format on
    HPX_DEPRECATED_V(1, 6,
        "hpx::parallel::includes is deprecated, use hpx::includes instead")
        typename util::detail::algorithm_result<ExPolicy, bool>::type
        includes(ExPolicy&& policy, FwdIter1 first1, FwdIter1 last1,
            FwdIter2 first2, FwdIter2 last2, Pred&& op = Pred())
    {
        static_assert((hpx::traits::is_forward_iterator<FwdIter1>::value),
            "Requires at least forward iterator.");
        static_assert((hpx::traits::is_forward_iterator<FwdIter2>::value),
            "Requires at least forward iterator.");

        using is_seq = hpx::is_sequenced_execution_policy<ExPolicy>;

        return detail::includes().call(std::forward<ExPolicy>(policy), is_seq(),
            first1, last1, first2, last2, std::forward<Pred>(op),
            util::projection_identity(), util::projection_identity());
    }
}}}    // namespace hpx::parallel::v1

namespace hpx {

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::includes
    HPX_INLINE_CONSTEXPR_VARIABLE struct includes_t final
      : hpx::functional::tag<includes_t>
    {
    private:
        // clang-format off
        template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
            typename Pred = hpx::parallel::v1::detail::less,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_iterator<FwdIter1>::value &&
                hpx::traits::is_iterator<FwdIter2>::value &&
                hpx::traits::is_invocable<Pred,
                    typename std::iterator_traits<FwdIter1>::value_type,
                    typename std::iterator_traits<FwdIter2>::value_type
                >::value
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            bool>::type
        tag_invoke(includes_t, ExPolicy&& policy, FwdIter1 first1,
            FwdIter1 last1, FwdIter2 first2, FwdIter2 last2, Pred&& op = Pred())
        {
            static_assert((hpx::traits::is_forward_iterator<FwdIter1>::value),
                "Requires at least forward iterator.");
            static_assert((hpx::traits::is_forward_iterator<FwdIter2>::value),
                "Requires at least forward iterator.");

            using is_seq =
                hpx::parallel::execution::is_sequenced_execution_policy<
                    ExPolicy>;

            return hpx::parallel::v1::detail::includes().call(
                std::forward<ExPolicy>(policy), is_seq(), first1, last1, first2,
                last2, std::forward<Pred>(op),
                hpx::parallel::util::projection_identity(),
                hpx::parallel::util::projection_identity());
        }

        // clang-format off
        template <typename FwdIter1, typename FwdIter2,
            typename Pred = hpx::parallel::v1::detail::less,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator<FwdIter1>::value &&
                hpx::traits::is_iterator<FwdIter2>::value &&
                hpx::traits::is_invocable<Pred,
                    typename std::iterator_traits<FwdIter1>::value_type,
                    typename std::iterator_traits<FwdIter2>::value_type
                >::value
            )>
        // clang-format on
        friend bool tag_invoke(includes_t, FwdIter1 first1, FwdIter1 last1,
            FwdIter2 first2, FwdIter2 last2, Pred&& op = Pred())
        {
            static_assert((hpx::traits::is_forward_iterator<FwdIter1>::value),
                "Requires at least forward iterator.");
            static_assert((hpx::traits::is_forward_iterator<FwdIter2>::value),
                "Requires at least forward iterator.");

            return hpx::parallel::v1::detail::includes().call(
                hpx::execution::seq, std::true_type(), first1, last1, first2,
                last2, std::forward<Pred>(op),
                hpx::parallel::util::projection_identity(),
                hpx::parallel::util::projection_identity());
        }
    } includes{};
}    // namespace hpx

#endif    // DOXYGEN
