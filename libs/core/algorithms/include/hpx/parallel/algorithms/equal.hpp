//  Copyright (c) 2007-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/equal.hpp

#pragma once

#if defined(DOXYGEN)
namespace hpx {
    // clang-format off

    /// Returns true if the range [first1, last1) is equal to the range
    /// [first2, last2), and false otherwise.
    ///
    /// \note   Complexity: At most min(last1 - first1, last2 - first2)
    ///         applications of the predicate \a f.
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
    ///                     overload of \a equal requires \a Pred to meet the
    ///                     requirements of \a CopyConstructible. This defaults
    ///                     to std::equal_to<>
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
    ///                     elements should be treated as equal. The signature
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
    /// The comparison operations in the parallel \a equal algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The comparison operations in the parallel \a equal algorithm invoked
    /// with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \note     The two ranges are considered equal if, for every iterator
    ///           i in the range [first1,last1), *i equals *(first2 + (i - first1)).
    ///           This overload of equal uses operator== to determine if two
    ///           elements are equal.
    ///
    /// \returns  The \a equal algorithm returns a \a hpx::future<bool> if the
    ///           execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a bool otherwise.
    ///           The \a equal algorithm returns true if the elements in the
    ///           two ranges are equal, otherwise it returns false.
    ///           If the length of the range [first1, last1) does not equal
    ///           the length of the range [first2, last2), it returns false.
    ///
    template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
        typename Pred = detail::equal_to>
    typename util::detail::algorithm_result<ExPolicy, bool>::type
    equal(ExPolicy&& policy, FwdIter1 first1, FwdIter1 last1,
            FwdIter2 first2, FwdIter2 last2, Pred&& op = Pred());

    /// Returns true if the range [first1, last1) is equal to the range
    /// starting at first2, and false otherwise.
    ///
    /// \note   Complexity: At most \a last1 - \a first1 applications of the
    ///         predicate \a f.
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
    ///                     overload of \a equal requires \a Pred to meet the
    ///                     requirements of \a CopyConstructible. This defaults
    ///                     to std::equal_to<>
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first1       Refers to the beginning of the sequence of elements
    ///                     of the first range the algorithm will be applied to.
    /// \param last1        Refers to the end of the sequence of elements of
    ///                     the first range the algorithm will be applied to.
    /// \param first2       Refers to the beginning of the sequence of elements
    ///                     of the second range the algorithm will be applied to.
    /// \param op           The binary predicate which returns true if the
    ///                     elements should be treated as equal. The signature
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
    /// The comparison operations in the parallel \a equal algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The comparison operations in the parallel \a equal algorithm invoked
    /// with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \note     The two ranges are considered equal if, for every iterator
    ///           i in the range [first1,last1), *i equals *(first2 + (i - first1)).
    ///           This overload of equal uses operator== to determine if two
    ///           elements are equal.
    ///
    /// \returns  The \a equal algorithm returns a \a hpx::future<bool> if the
    ///           execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a bool otherwise.
    ///           The \a equal algorithm returns true if the elements in the
    ///           two ranges are equal, otherwise it returns false.
    ///
    template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
        typename Pred = detail::equal_to>
    typename util::detail::algorithm_result<ExPolicy, bool>::type
    equal(ExPolicy&& policy, FwdIter1 first1, FwdIter1 last1,
            FwdIter2 first2, Pred&& op = Pred());

    // clang-format on
}    // namespace hpx

#else    // DOXYGEN

#include <hpx/config.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/functional/invoke.hpp>
#include <hpx/iterator_support/range.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>

#include <hpx/execution/algorithms/detail/predicates.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/detail/distance.hpp>
#include <hpx/parallel/algorithms/detail/equal.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/detail/sender_util.hpp>
#include <hpx/parallel/util/loop.hpp>
#include <hpx/parallel/util/partitioner.hpp>
#include <hpx/parallel/util/projection_identity.hpp>
#include <hpx/parallel/util/zip_iterator.hpp>
#include <hpx/type_support/unused.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace parallel { inline namespace v1 {
    ///////////////////////////////////////////////////////////////////////////
    // equal (binary)
    namespace detail {
        /// \cond NOINTERNAL

        ///////////////////////////////////////////////////////////////////////
        struct equal_binary : public detail::algorithm<equal_binary, bool>
        {
            equal_binary()
              : equal_binary::algorithm("equal_binary")
            {
            }

            template <typename ExPolicy, typename Iter1, typename Sent1,
                typename Iter2, typename Sent2, typename F, typename Proj1,
                typename Proj2>
            static bool sequential(ExPolicy, Iter1 first1, Sent1 last1,
                Iter2 first2, Sent2 last2, F&& f, Proj1&& proj1, Proj2&& proj2)
            {
                return sequential_equal_binary<std::decay_t<ExPolicy>>(first1,
                    last1, first2, last2, HPX_FORWARD(F, f),
                    HPX_FORWARD(Proj1, proj1), HPX_FORWARD(Proj2, proj2));
            }

            template <typename ExPolicy, typename Iter1, typename Sent1,
                typename Iter2, typename Sent2, typename F, typename Proj1,
                typename Proj2>
            static typename util::detail::algorithm_result<ExPolicy, bool>::type
            parallel(ExPolicy&& policy, Iter1 first1, Sent1 last1, Iter2 first2,
                Sent2 last2, F&& f, Proj1&& proj1, Proj2&& proj2)
            {
                typedef typename std::iterator_traits<Iter1>::difference_type
                    difference_type1;
                typedef typename std::iterator_traits<Iter2>::difference_type
                    difference_type2;

                if (first1 == last1)
                {
                    return util::detail::algorithm_result<ExPolicy, bool>::get(
                        first2 == last2);
                }

                if (first2 == last2)
                {
                    return util::detail::algorithm_result<ExPolicy, bool>::get(
                        false);
                }

                difference_type1 count1 = detail::distance(first1, last1);

                // The specification of std::equal(_binary) states that if FwdIter1
                // and FwdIter2 meet the requirements of RandomAccessIterator and
                // last1 - first1 != last2 - first2 then no applications of the
                // predicate p are made.
                //
                // We perform this check for any iterator type better than input
                // iterators. This could turn into a QoI issue.
                difference_type2 count2 = detail::distance(first2, last2);
                if (count1 != count2)
                {
                    return util::detail::algorithm_result<ExPolicy, bool>::get(
                        false);
                }

                typedef hpx::util::zip_iterator<Iter1, Iter2> zip_iterator;

                util::cancellation_token<> tok;

                // Note: replacing the invoke() with HPX_INVOKE()
                // below makes gcc generate errors
                auto f1 = [tok, f = HPX_FORWARD(F, f),
                              proj1 = HPX_FORWARD(Proj1, proj1),
                              proj2 = HPX_FORWARD(Proj2, proj2)](
                              zip_iterator it,
                              std::size_t part_count) mutable -> bool {
                    sequential_equal_binary<std::decay_t<ExPolicy>>(it,
                        part_count, tok, HPX_FORWARD(F, f),
                        HPX_FORWARD(Proj1, proj1), HPX_FORWARD(Proj2, proj2));
                    return !tok.was_cancelled();
                };

                return util::partitioner<ExPolicy, bool>::call(
                    HPX_FORWARD(ExPolicy, policy),
                    hpx::util::make_zip_iterator(first1, first2), count1,
                    HPX_MOVE(f1), [](auto&& results) -> bool {
                        return std::all_of(hpx::util::begin(results),
                            hpx::util::end(results),
                            [](hpx::future<bool>& val) { return val.get(); });
                    });
            }
        };
        /// \endcond
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    // equal
    namespace detail {
        /// \cond NOINTERNAL
        struct equal : public detail::algorithm<equal, bool>
        {
            equal()
              : equal::algorithm("equal")
            {
            }

            template <typename ExPolicy, typename InIter1, typename InIter2,
                typename F>
            static bool sequential(
                ExPolicy, InIter1 first1, InIter1 last1, InIter2 first2, F&& f)
            {
                return sequential_equal<std::decay_t<ExPolicy>>(
                    first1, last1, first2, HPX_FORWARD(F, f));
            }

            template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
                typename F>
            static typename util::detail::algorithm_result<ExPolicy, bool>::type
            parallel(ExPolicy&& policy, FwdIter1 first1, FwdIter1 last1,
                FwdIter2 first2, F&& f)
            {
                if (first1 == last1)
                {
                    return util::detail::algorithm_result<ExPolicy, bool>::get(
                        true);
                }

                typedef typename std::iterator_traits<FwdIter1>::difference_type
                    difference_type;
                difference_type count = std::distance(first1, last1);

                typedef hpx::util::zip_iterator<FwdIter1, FwdIter2>
                    zip_iterator;

                util::cancellation_token<> tok;
                auto f1 = [f, tok](zip_iterator it,
                              std::size_t part_count) mutable -> bool {
                    sequential_equal<std::decay_t<ExPolicy>>(
                        it, part_count, tok, HPX_FORWARD(F, f));
                    return !tok.was_cancelled();
                };

                return util::partitioner<ExPolicy, bool>::call(
                    HPX_FORWARD(ExPolicy, policy),
                    hpx::util::make_zip_iterator(first1, first2), count,
                    HPX_MOVE(f1), [](auto&& results) {
                        return std::all_of(hpx::util::begin(results),
                            hpx::util::end(results),
                            [](hpx::future<bool>& val) { return val.get(); });
                    });
            }
        };
        /// \endcond
    }    // namespace detail
}}}      // namespace hpx::parallel::v1

namespace hpx {

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::equal
    inline constexpr struct equal_t final
      : hpx::detail::tag_parallel_algorithm<equal_t>
    {
    private:
        // clang-format off
        template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
            typename Pred,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_iterator<FwdIter1>::value &&
                hpx::traits::is_iterator<FwdIter2>::value &&
                hpx::is_invocable_v<Pred,
                    typename std::iterator_traits<FwdIter1>::value_type,
                    typename std::iterator_traits<FwdIter2>::value_type
                >
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            bool>::type
        tag_fallback_invoke(equal_t, ExPolicy&& policy, FwdIter1 first1,
            FwdIter1 last1, FwdIter2 first2, FwdIter2 last2, Pred&& op)
        {
            static_assert((hpx::traits::is_forward_iterator<FwdIter1>::value),
                "Requires at least forward iterator.");
            static_assert((hpx::traits::is_forward_iterator<FwdIter2>::value),
                "Requires at least forward iterator.");

            return hpx::parallel::v1::detail::equal_binary().call(
                HPX_FORWARD(ExPolicy, policy), first1, last1, first2, last2,
                HPX_FORWARD(Pred, op),
                hpx::parallel::util::projection_identity{},
                hpx::parallel::util::projection_identity{});
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_iterator<FwdIter1>::value &&
                hpx::traits::is_iterator<FwdIter2>::value
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            bool>::type
        tag_fallback_invoke(equal_t, ExPolicy&& policy, FwdIter1 first1,
            FwdIter1 last1, FwdIter2 first2, FwdIter2 last2)
        {
            static_assert((hpx::traits::is_forward_iterator<FwdIter1>::value),
                "Requires at least forward iterator.");
            static_assert((hpx::traits::is_forward_iterator<FwdIter2>::value),
                "Requires at least forward iterator.");

            return hpx::parallel::v1::detail::equal_binary().call(
                HPX_FORWARD(ExPolicy, policy), first1, last1, first2, last2,
                hpx::parallel::v1::detail::equal_to{},
                hpx::parallel::util::projection_identity{},
                hpx::parallel::util::projection_identity{});
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
            typename Pred,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_iterator<FwdIter1>::value &&
                hpx::traits::is_iterator<FwdIter2>::value &&
                hpx::is_invocable_v<Pred,
                    typename std::iterator_traits<FwdIter1>::value_type,
                    typename std::iterator_traits<FwdIter2>::value_type
                >
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            bool>::type
        tag_fallback_invoke(equal_t, ExPolicy&& policy, FwdIter1 first1,
            FwdIter1 last1, FwdIter2 first2, Pred&& op)
        {
            static_assert((hpx::traits::is_forward_iterator<FwdIter1>::value),
                "Requires at least forward iterator.");
            static_assert((hpx::traits::is_forward_iterator<FwdIter2>::value),
                "Requires at least forward iterator.");

            return hpx::parallel::v1::detail::equal().call(
                HPX_FORWARD(ExPolicy, policy), first1, last1, first2,
                HPX_FORWARD(Pred, op));
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_iterator<FwdIter1>::value &&
                hpx::traits::is_iterator<FwdIter2>::value
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            bool>::type
        tag_fallback_invoke(equal_t, ExPolicy&& policy, FwdIter1 first1,
            FwdIter1 last1, FwdIter2 first2)
        {
            static_assert((hpx::traits::is_forward_iterator<FwdIter1>::value),
                "Requires at least forward iterator.");
            static_assert((hpx::traits::is_forward_iterator<FwdIter2>::value),
                "Requires at least forward iterator.");

            return hpx::parallel::v1::detail::equal().call(
                HPX_FORWARD(ExPolicy, policy), first1, last1, first2,
                hpx::parallel::v1::detail::equal_to{});
        }

        // clang-format off
        template <typename FwdIter1, typename FwdIter2, typename Pred,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator<FwdIter1>::value &&
                hpx::traits::is_iterator<FwdIter2>::value &&
                hpx::is_invocable_v<Pred,
                    typename std::iterator_traits<FwdIter1>::value_type,
                    typename std::iterator_traits<FwdIter2>::value_type
                >
            )>
        // clang-format on
        friend bool tag_fallback_invoke(equal_t, FwdIter1 first1,
            FwdIter1 last1, FwdIter2 first2, FwdIter2 last2, Pred&& op)
        {
            static_assert((hpx::traits::is_forward_iterator<FwdIter1>::value),
                "Requires at least forward iterator.");
            static_assert((hpx::traits::is_forward_iterator<FwdIter2>::value),
                "Requires at least forward iterator.");

            return hpx::parallel::v1::detail::equal_binary().call(
                hpx::execution::seq, first1, last1, first2, last2,
                HPX_FORWARD(Pred, op),
                hpx::parallel::util::projection_identity{},
                hpx::parallel::util::projection_identity{});
        }

        // clang-format off
        template <typename FwdIter1, typename FwdIter2,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator<FwdIter1>::value &&
                hpx::traits::is_iterator<FwdIter2>::value
            )>
        // clang-format on
        friend bool tag_fallback_invoke(equal_t, FwdIter1 first1,
            FwdIter1 last1, FwdIter2 first2, FwdIter2 last2)
        {
            static_assert((hpx::traits::is_forward_iterator<FwdIter1>::value),
                "Requires at least forward iterator.");
            static_assert((hpx::traits::is_forward_iterator<FwdIter2>::value),
                "Requires at least forward iterator.");

            return hpx::parallel::v1::detail::equal_binary().call(
                hpx::execution::seq, first1, last1, first2, last2,
                hpx::parallel::v1::detail::equal_to{},
                hpx::parallel::util::projection_identity{},
                hpx::parallel::util::projection_identity{});
        }

        // clang-format off
        template <typename FwdIter1, typename FwdIter2, typename Pred,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator<FwdIter1>::value &&
                hpx::traits::is_iterator<FwdIter2>::value &&
                hpx::is_invocable_v<Pred,
                    typename std::iterator_traits<FwdIter1>::value_type,
                    typename std::iterator_traits<FwdIter2>::value_type
                >
            )>
        // clang-format on
        friend bool tag_fallback_invoke(equal_t, FwdIter1 first1,
            FwdIter1 last1, FwdIter2 first2, Pred&& op)
        {
            static_assert((hpx::traits::is_forward_iterator<FwdIter1>::value),
                "Requires at least forward iterator.");
            static_assert((hpx::traits::is_forward_iterator<FwdIter2>::value),
                "Requires at least forward iterator.");

            return hpx::parallel::v1::detail::equal().call(hpx::execution::seq,
                first1, last1, first2, HPX_FORWARD(Pred, op));
        }

        // clang-format off
        template <typename FwdIter1, typename FwdIter2,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator<FwdIter1>::value &&
                hpx::traits::is_iterator<FwdIter2>::value
            )>
        // clang-format on
        friend bool tag_fallback_invoke(
            equal_t, FwdIter1 first1, FwdIter1 last1, FwdIter2 first2)
        {
            static_assert((hpx::traits::is_forward_iterator<FwdIter1>::value),
                "Requires at least forward iterator.");
            static_assert((hpx::traits::is_forward_iterator<FwdIter2>::value),
                "Requires at least forward iterator.");

            return hpx::parallel::v1::detail::equal().call(hpx::execution::seq,
                first1, last1, first2, hpx::parallel::v1::detail::equal_to{});
        }
    } equal{};
}    // namespace hpx

#endif    // DOXYGEN
