//  Copyright (c) 2007-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/mismatch.hpp

#pragma once

#if defined(DOXYGEN)
namespace hpx {
    // clang-format off

    /// Returns true if the range [first1, last1) is mismatch to the range
    /// [first2, last2), and false otherwise.
    ///
    /// \note   Complexity: At most min(last1 - first1, last2 - first2)
    ///         applications of the predicate \a f. If \a FwdIter1
    ///         and \a FwdIter2 meet the requirements of \a RandomAccessIterator
    ///         and (last1 - first1) != (last2 - first2) then no applications
    ///         of the predicate \a f are made.
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
    ///                     overload of \a mismatch requires \a Pred to meet the
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
    ///                     elements should be treated as mismatch. The signature
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
    /// The comparison operations in the parallel \a mismatch algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The comparison operations in the parallel \a mismatch algorithm invoked
    /// with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \note     The two ranges are considered mismatch if, for every iterator
    ///           i in the range [first1,last1), *i mismatchs *(first2 + (i - first1)).
    ///           This overload of mismatch uses operator== to determine if two
    ///           elements are mismatch.
    ///
    /// \returns  The \a mismatch algorithm returns a \a hpx::future<bool> if the
    ///           execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a bool otherwise.
    ///           The \a mismatch algorithm returns true if the elements in the
    ///           two ranges are mismatch, otherwise it returns false.
    ///           If the length of the range [first1, last1) does not mismatch
    ///           the length of the range [first2, last2), it returns false.
    ///
    template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
        typename Pred = detail::equal_to>
    typename util::detail::algorithm_result<
        ExPolicy, std::pair<FwdIter1, FwdIter2>>::type
    mismatch(ExPolicy&& policy, FwdIter1 first1, FwdIter1 last1,
            FwdIter2 first2, FwdIter2 last2, Pred&& op = Pred());

    /// Returns std::pair with iterators to the first two non-equivalent
    /// elements.
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
    ///                     overload of \a mismatch requires \a Pred to meet the
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
    ///                     elements should be treated as mismatch. The signature
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
    /// The comparison operations in the parallel \a mismatch algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The comparison operations in the parallel \a mismatch algorithm invoked
    /// with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a mismatch algorithm returns a
    ///           \a hpx::future<std::pair<FwdIter1, FwdIter2> > if the
    ///           execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a std::pair<FwdIter1, FwdIter2> otherwise.
    ///           The \a mismatch algorithm returns the first mismatching pair
    ///           of elements from two ranges: one defined by [first1, last1)
    ///           and another defined by [first2, last2).
    ///
    template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
        typename Pred = detail::equal_to>
    typename util::detail::algorithm_result<
        ExPolicy, std::pair<FwdIter1, FwdIter2>>::type
    mismatch(ExPolicy&& policy, FwdIter1 first1, FwdIter1 last1, FwdIter2 first2,
        Pred&& op = Pred());

    // clang-format on
}    // namespace hpx

#else    // DOXYGEN

#include <hpx/config.hpp>
#include <hpx/functional/invoke.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>

#include <hpx/execution/algorithms/detail/predicates.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/detail/distance.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/loop.hpp>
#include <hpx/parallel/util/partitioner.hpp>
#include <hpx/parallel/util/result_types.hpp>
#include <hpx/parallel/util/zip_iterator.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace parallel { inline namespace v1 {
    ///////////////////////////////////////////////////////////////////////////
    // mismatch (binary)
    namespace detail {

        ///////////////////////////////////////////////////////////////////////
        template <typename Iter1, typename Sent1, typename Iter2,
            typename Sent2, typename F, typename Proj1, typename Proj2>
        util::in_in_result<Iter1, Iter2> sequential_mismatch_binary(
            Iter1 first1, Sent1 last1, Iter2 first2, Sent2 last2, F&& f,
            Proj1&& proj1, Proj2&& proj2)
        {
            while (first1 != last1 && first2 != last2 &&
                hpx::util::invoke(f, hpx::util::invoke(proj1, *first1),
                    hpx::util::invoke(proj2, *first2)))
            {
                ++first1, ++first2;
            }
            return {first1, first2};
        }

        template <typename IterPair>
        struct mismatch_binary
          : public detail::algorithm<mismatch_binary<IterPair>, IterPair>
        {
            mismatch_binary()
              : mismatch_binary::algorithm("mismatch_binary")
            {
            }

            template <typename ExPolicy, typename Iter1, typename Sent1,
                typename Iter2, typename Sent2, typename F, typename Proj1,
                typename Proj2>
            static util::in_in_result<Iter1, Iter2> sequential(ExPolicy,
                Iter1 first1, Sent1 last1, Iter2 first2, Sent2 last2, F&& f,
                Proj1&& proj1, Proj2&& proj2)
            {
                return sequential_mismatch_binary(first1, last1, first2, last2,
                    std::forward<F>(f), std::forward<Proj1>(proj1),
                    std::forward<Proj2>(proj2));
            }

            template <typename ExPolicy, typename Iter1, typename Sent1,
                typename Iter2, typename Sent2, typename F, typename Proj1,
                typename Proj2>
            static typename util::detail::algorithm_result<ExPolicy,
                util::in_in_result<Iter1, Iter2>>::type
            parallel(ExPolicy&& policy, Iter1 first1, Sent1 last1, Iter2 first2,
                Sent2 last2, F&& f, Proj1&& proj1, Proj2&& proj2)
            {
                if (first1 == last1 || first2 == last2)
                {
                    return util::detail::algorithm_result<ExPolicy,
                        util::in_in_result<Iter1, Iter2>>::
                        get(util::in_in_result<Iter1, Iter2>{first1, first2});
                }

                typedef typename std::iterator_traits<Iter1>::difference_type
                    difference_type1;
                difference_type1 count1 = detail::distance(first1, last1);

                // The specification of std::mismatch(_binary) states that if FwdIter1
                // and FwdIter2 meet the requirements of RandomAccessIterator and
                // last1 - first1 != last2 - first2 then no applications of the
                // predicate p are made.
                //
                // We perform this check for any iterator type better than input
                // iterators. This could turn into a QoI issue.
                typedef typename std::iterator_traits<Iter2>::difference_type
                    difference_type2;
                difference_type2 count2 = detail::distance(first2, last2);
                if (count1 != count2)
                {
                    return util::detail::algorithm_result<ExPolicy,
                        util::in_in_result<Iter1, Iter2>>::
                        get(util::in_in_result<Iter1, Iter2>{first1, first2});
                }

                typedef hpx::util::zip_iterator<Iter1, Iter2> zip_iterator;
                typedef typename zip_iterator::reference reference;

                util::cancellation_token<std::size_t> tok(count1);

                auto f1 = [tok, f = std::forward<F>(f),
                              proj1 = std::forward<Proj1>(proj1),
                              proj2 = std::forward<Proj2>(proj2)](
                              zip_iterator it, std::size_t part_count,
                              std::size_t base_idx) mutable -> void {
                    util::loop_idx_n(base_idx, it, part_count, tok,
                        [&f, &proj1, &proj2, &tok](reference t, std::size_t i) {
                            if (!hpx::util::invoke(f,
                                    hpx::util::invoke(
                                        proj1, hpx::util::get<0>(t)),
                                    hpx::util::invoke(
                                        proj2, hpx::util::get<1>(t))))
                            {
                                tok.cancel(i);
                            }
                        });
                };

                auto f2 = [=](std::vector<hpx::future<void>>&&) mutable
                    -> util::in_in_result<Iter1, Iter2> {
                    difference_type1 mismatched =
                        static_cast<difference_type1>(tok.get_data());
                    if (mismatched != count1)
                    {
                        std::advance(first1, mismatched);
                        std::advance(first2, mismatched);
                    }
                    else
                    {
                        first1 = last1;
                        first2 = last2;
                    }
                    return {first1, first2};
                };

                return util::partitioner<ExPolicy,
                    util::in_in_result<Iter1, Iter2>,
                    void>::call_with_index(std::forward<ExPolicy>(policy),
                    hpx::util::make_zip_iterator(first1, first2), count1, 1,
                    std::move(f1), std::move(f2));
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename I1, typename I2>
        std::pair<I1, I2> get_pair(util::in_in_result<I1, I2>&& p)
        {
            return {p.in1, p.in2};
        }

        template <typename I1, typename I2>
        hpx::future<std::pair<I1, I2>> get_pair(
            hpx::future<util::in_in_result<I1, I2>>&& f)
        {
            return lcos::make_future<std::pair<I1, I2>>(std::move(f),
                [](util::in_in_result<I1, I2>&& p) -> std::pair<I1, I2> {
                    return {std::move(p.in1), std::move(p.in2)};
                });
        }
    }    // namespace detail

    // clang-format off
    template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
        typename Pred = detail::equal_to,
        HPX_CONCEPT_REQUIRES_(
            execution::is_execution_policy<ExPolicy>::value &&
            hpx::traits::is_iterator<FwdIter1>::value &&
            hpx::traits::is_iterator<FwdIter2>::value &&
            hpx::traits::is_invocable<Pred,
                typename std::iterator_traits<FwdIter1>::value_type,
                typename std::iterator_traits<FwdIter2>::value_type
            >::value
        )>
    // clang-format on
    HPX_DEPRECATED_V(1, 6,
        "hpx::parallel::mismatch is deprecated, use hpx::mismatch instead")
        typename util::detail::algorithm_result<ExPolicy,
            std::pair<FwdIter1, FwdIter2>>::type
        mismatch(ExPolicy&& policy, FwdIter1 first1, FwdIter1 last1,
            FwdIter2 first2, FwdIter2 last2, Pred&& op = Pred())
    {
        static_assert((hpx::traits::is_forward_iterator<FwdIter1>::value),
            "Requires at least forward iterator.");
        static_assert((hpx::traits::is_forward_iterator<FwdIter2>::value),
            "Requires at least forward iterator.");

        using is_seq = execution::is_sequenced_execution_policy<ExPolicy>;

        return detail::get_pair(
            detail::mismatch_binary<util::in_in_result<FwdIter1, FwdIter2>>()
                .call(std::forward<ExPolicy>(policy), is_seq{}, first1, last1,
                    first2, last2, std::forward<Pred>(op)));
    }

    ///////////////////////////////////////////////////////////////////////////
    // mismatch
    namespace detail {

        template <typename IterPair>
        struct mismatch : public detail::algorithm<mismatch<IterPair>, IterPair>
        {
            mismatch()
              : mismatch::algorithm("mismatch")
            {
            }

            template <typename ExPolicy, typename InIter1, typename InIter2,
                typename F>
            static IterPair sequential(
                ExPolicy, InIter1 first1, InIter1 last1, InIter2 first2, F&& f)
            {
                return std::mismatch(first1, last1, first2, std::forward<F>(f));
            }

            template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
                typename F>
            static typename util::detail::algorithm_result<ExPolicy,
                IterPair>::type
            parallel(ExPolicy&& policy, FwdIter1 first1, FwdIter1 last1,
                FwdIter2 first2, F&& f)
            {
                if (first1 == last1)
                {
                    return util::detail::algorithm_result<ExPolicy,
                        IterPair>::get(std::make_pair(first1, first2));
                }

                typedef typename std::iterator_traits<FwdIter1>::difference_type
                    difference_type;
                difference_type count = std::distance(first1, last1);

                typedef hpx::util::zip_iterator<FwdIter1, FwdIter2>
                    zip_iterator;
                typedef typename zip_iterator::reference reference;

                util::cancellation_token<std::size_t> tok(count);

                auto f1 = [tok, f = std::forward<F>(f)](zip_iterator it,
                              std::size_t part_count,
                              std::size_t base_idx) mutable -> void {
                    util::loop_idx_n(base_idx, it, part_count, tok,
                        [&f, &tok](reference t, std::size_t i) {
                            if (!hpx::util::invoke(f, hpx::util::get<0>(t),
                                    hpx::util::get<1>(t)))
                            {
                                tok.cancel(i);
                            }
                        });
                };
                auto f2 = [=](std::vector<hpx::future<void>>&&) mutable
                    -> std::pair<FwdIter1, FwdIter2> {
                    difference_type mismatched =
                        static_cast<difference_type>(tok.get_data());
                    if (mismatched != count)
                        std::advance(first1, mismatched);
                    else
                        first1 = last1;

                    std::advance(first2, mismatched);
                    return std::make_pair(first1, first2);
                };

                return util::partitioner<ExPolicy, IterPair,
                    void>::call_with_index(std::forward<ExPolicy>(policy),
                    hpx::util::make_zip_iterator(first1, first2), count, 1,
                    std::move(f1), std::move(f2));
            }
        };
    }    // namespace detail

    // clang-format off
    template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
        typename Pred = detail::equal_to,
        HPX_CONCEPT_REQUIRES_(
            execution::is_execution_policy<ExPolicy>::value &&
            hpx::traits::is_iterator<FwdIter1>::value &&
            hpx::traits::is_iterator<FwdIter2>::value &&
            hpx::traits::is_invocable<Pred,
                typename std::iterator_traits<FwdIter1>::value_type,
                typename std::iterator_traits<FwdIter2>::value_type
            >::value
        )>
    // clang-format on
    HPX_DEPRECATED_V(1, 6,
        "hpx::parallel::mismatch is deprecated, use hpx::mismatch instead")
        typename util::detail::algorithm_result<ExPolicy,
            std::pair<FwdIter1, FwdIter2>>::type
        mismatch(ExPolicy&& policy, FwdIter1 first1, FwdIter1 last1,
            FwdIter2 first2, Pred&& op = Pred())
    {
        static_assert((hpx::traits::is_forward_iterator<FwdIter1>::value),
            "Requires at least forward iterator.");
        static_assert((hpx::traits::is_forward_iterator<FwdIter2>::value),
            "Requires at least forward iterator.");

        typedef execution::is_sequenced_execution_policy<ExPolicy> is_seq;

        typedef std::pair<FwdIter1, FwdIter2> result_type;
        return detail::mismatch<result_type>().call(
            std::forward<ExPolicy>(policy), is_seq{}, first1, last1, first2,
            std::forward<Pred>(op));
    }

}}}    // namespace hpx::parallel::v1

namespace hpx {

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::mismatch
    HPX_INLINE_CONSTEXPR_VARIABLE struct mismatch_t final
      : hpx::functional::tag<mismatch_t>
    {
    private:
        // clang-format off
        template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
            typename Pred,
            HPX_CONCEPT_REQUIRES_(
                hpx::parallel::execution::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_iterator<FwdIter1>::value &&
                hpx::traits::is_iterator<FwdIter2>::value &&
                hpx::traits::is_invocable<Pred,
                    typename std::iterator_traits<FwdIter1>::value_type,
                    typename std::iterator_traits<FwdIter2>::value_type
                >::value
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            std::pair<FwdIter1, FwdIter2>>::type
        tag_invoke(mismatch_t, ExPolicy&& policy, FwdIter1 first1,
            FwdIter1 last1, FwdIter2 first2, FwdIter2 last2, Pred&& op)
        {
            static_assert((hpx::traits::is_forward_iterator<FwdIter1>::value),
                "Requires at least forward iterator.");
            static_assert((hpx::traits::is_forward_iterator<FwdIter2>::value),
                "Requires at least forward iterator.");

            using is_seq =
                hpx::parallel::execution::is_sequenced_execution_policy<
                    ExPolicy>;

            return hpx::parallel::v1::detail::get_pair(
                hpx::parallel::v1::detail::mismatch_binary<
                    hpx::parallel::util::in_in_result<FwdIter1, FwdIter2>>()
                    .call(std::forward<ExPolicy>(policy), is_seq{}, first1,
                        last1, first2, last2, std::forward<Pred>(op),
                        hpx::parallel::util::projection_identity{},
                        hpx::parallel::util::projection_identity{}));
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
            HPX_CONCEPT_REQUIRES_(
                hpx::parallel::execution::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_iterator<FwdIter1>::value &&
                hpx::traits::is_iterator<FwdIter2>::value
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            std::pair<FwdIter1, FwdIter2>>::type
        tag_invoke(mismatch_t, ExPolicy&& policy, FwdIter1 first1,
            FwdIter1 last1, FwdIter2 first2, FwdIter2 last2)
        {
            static_assert((hpx::traits::is_forward_iterator<FwdIter1>::value),
                "Requires at least forward iterator.");
            static_assert((hpx::traits::is_forward_iterator<FwdIter2>::value),
                "Requires at least forward iterator.");

            using is_seq =
                hpx::parallel::execution::is_sequenced_execution_policy<
                    ExPolicy>;

            return hpx::parallel::v1::detail::get_pair(
                hpx::parallel::v1::detail::mismatch_binary<
                    hpx::parallel::util::in_in_result<FwdIter1, FwdIter2>>()
                    .call(std::forward<ExPolicy>(policy), is_seq{}, first1,
                        last1, first2, last2,
                        hpx::parallel::v1::detail::equal_to{},
                        hpx::parallel::util::projection_identity{},
                        hpx::parallel::util::projection_identity{}));
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
            typename Pred,
            HPX_CONCEPT_REQUIRES_(
                hpx::parallel::execution::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_iterator<FwdIter1>::value &&
                hpx::traits::is_iterator<FwdIter2>::value &&
                hpx::traits::is_invocable<Pred,
                    typename std::iterator_traits<FwdIter1>::value_type,
                    typename std::iterator_traits<FwdIter2>::value_type
                >::value
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            std::pair<FwdIter1, FwdIter2>>::type
        tag_invoke(mismatch_t, ExPolicy&& policy, FwdIter1 first1,
            FwdIter1 last1, FwdIter2 first2, Pred&& op)
        {
            static_assert((hpx::traits::is_forward_iterator<FwdIter1>::value),
                "Requires at least forward iterator.");
            static_assert((hpx::traits::is_forward_iterator<FwdIter2>::value),
                "Requires at least forward iterator.");

            using is_seq =
                hpx::parallel::execution::is_sequenced_execution_policy<
                    ExPolicy>;

            return hpx::parallel::v1::detail::mismatch<
                std::pair<FwdIter1, FwdIter2>>()
                .call(std::forward<ExPolicy>(policy), is_seq{}, first1, last1,
                    first2, std::forward<Pred>(op));
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
            HPX_CONCEPT_REQUIRES_(
                hpx::parallel::execution::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_iterator<FwdIter1>::value &&
                hpx::traits::is_iterator<FwdIter2>::value
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            std::pair<FwdIter1, FwdIter2>>::type
        tag_invoke(mismatch_t, ExPolicy&& policy, FwdIter1 first1,
            FwdIter1 last1, FwdIter2 first2)
        {
            static_assert((hpx::traits::is_forward_iterator<FwdIter1>::value),
                "Requires at least forward iterator.");
            static_assert((hpx::traits::is_forward_iterator<FwdIter2>::value),
                "Requires at least forward iterator.");

            using is_seq =
                hpx::parallel::execution::is_sequenced_execution_policy<
                    ExPolicy>;

            return hpx::parallel::v1::detail::mismatch<
                std::pair<FwdIter1, FwdIter2>>()
                .call(std::forward<ExPolicy>(policy), is_seq{}, first1, last1,
                    first2, hpx::parallel::v1::detail::equal_to{});
        }

        // clang-format off
        template <typename FwdIter1, typename FwdIter2,
            typename Pred,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator<FwdIter1>::value &&
                hpx::traits::is_iterator<FwdIter2>::value &&
                hpx::traits::is_invocable<Pred,
                    typename std::iterator_traits<FwdIter1>::value_type,
                    typename std::iterator_traits<FwdIter2>::value_type
                >::value
            )>
        // clang-format on
        friend std::pair<FwdIter1, FwdIter2> tag_invoke(mismatch_t,
            FwdIter1 first1, FwdIter1 last1, FwdIter2 first2, FwdIter2 last2,
            Pred&& op)
        {
            static_assert((hpx::traits::is_forward_iterator<FwdIter1>::value),
                "Requires at least forward iterator.");
            static_assert((hpx::traits::is_forward_iterator<FwdIter2>::value),
                "Requires at least forward iterator.");

            return hpx::parallel::v1::detail::get_pair(
                hpx::parallel::v1::detail::mismatch_binary<
                    hpx::parallel::util::in_in_result<FwdIter1, FwdIter2>>()
                    .call(hpx::parallel::execution::seq, std::true_type{},
                        first1, last1, first2, last2, std::forward<Pred>(op),
                        hpx::parallel::util::projection_identity{},
                        hpx::parallel::util::projection_identity{}));
        }

        // clang-format off
        template <typename FwdIter1, typename FwdIter2,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator<FwdIter1>::value &&
                hpx::traits::is_iterator<FwdIter2>::value
            )>
        // clang-format on
        friend std::pair<FwdIter1, FwdIter2> tag_invoke(mismatch_t,
            FwdIter1 first1, FwdIter1 last1, FwdIter2 first2, FwdIter2 last2)
        {
            static_assert((hpx::traits::is_forward_iterator<FwdIter1>::value),
                "Requires at least forward iterator.");
            static_assert((hpx::traits::is_forward_iterator<FwdIter2>::value),
                "Requires at least forward iterator.");

            return hpx::parallel::v1::detail::get_pair(
                hpx::parallel::v1::detail::mismatch_binary<
                    hpx::parallel::util::in_in_result<FwdIter1, FwdIter2>>()
                    .call(hpx::parallel::execution::seq, std::true_type{},
                        first1, last1, first2, last2,
                        hpx::parallel::v1::detail::equal_to{},
                        hpx::parallel::util::projection_identity{},
                        hpx::parallel::util::projection_identity{}));
        }

        // clang-format off
        template <typename FwdIter1, typename FwdIter2,
            typename Pred,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator<FwdIter1>::value &&
                hpx::traits::is_iterator<FwdIter2>::value &&
                hpx::traits::is_invocable<Pred,
                    typename std::iterator_traits<FwdIter1>::value_type,
                    typename std::iterator_traits<FwdIter2>::value_type
                >::value
            )>
        // clang-format on
        friend std::pair<FwdIter1, FwdIter2> tag_invoke(mismatch_t,
            FwdIter1 first1, FwdIter1 last1, FwdIter2 first2, Pred&& op)
        {
            static_assert((hpx::traits::is_forward_iterator<FwdIter1>::value),
                "Requires at least forward iterator.");
            static_assert((hpx::traits::is_forward_iterator<FwdIter2>::value),
                "Requires at least forward iterator.");

            return hpx::parallel::v1::detail::mismatch<
                std::pair<FwdIter1, FwdIter2>>()
                .call(hpx::parallel::execution::seq, std::true_type{}, first1,
                    last1, first2, std::forward<Pred>(op));
        }

        // clang-format off
        template <typename FwdIter1, typename FwdIter2,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator<FwdIter1>::value &&
                hpx::traits::is_iterator<FwdIter2>::value
            )>
        // clang-format on
        friend std::pair<FwdIter1, FwdIter2> tag_invoke(
            mismatch_t, FwdIter1 first1, FwdIter1 last1, FwdIter2 first2)
        {
            static_assert((hpx::traits::is_forward_iterator<FwdIter1>::value),
                "Requires at least forward iterator.");
            static_assert((hpx::traits::is_forward_iterator<FwdIter2>::value),
                "Requires at least forward iterator.");

            return hpx::parallel::v1::detail::mismatch<
                std::pair<FwdIter1, FwdIter2>>()
                .call(hpx::parallel::execution::seq, std::true_type{}, first1,
                    last1, first2, hpx::parallel::v1::detail::equal_to{});
        }

    } mismatch;
}    // namespace hpx

#endif    // DOXYGEN
