//  Copyright (c) 2007-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/set_intersection.hpp

#pragma once

#if defined(DOXYGEN)
namespace hpx {
    // clang-format off

    /// Constructs a sorted range beginning at dest consisting of all elements
    /// present in both sorted ranges [first1, last1) and
    /// [first2, last2). This algorithm expects both input ranges to be sorted
    /// with the given binary predicate \a f.
    ///
    /// \note   Complexity: At most 2*(N1 + N2 - 1) comparisons, where \a N1 is
    ///         the length of the first sequence and \a N2 is the length of the
    ///         second sequence.
    ///
    /// If some element is found \a m times in [first1, last1) and \a n times
    /// in [first2, last2), the first std::min(m, n) elements will be copied
    /// from the first range to the destination range. The order of equivalent
    /// elements is preserved. The resulting range cannot overlap with either
    /// of the input ranges.
    ///
    /// The resulting range cannot overlap with either of the input ranges.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it applies user-provided function objects.
    /// \tparam FwdIter1    The type of the source iterators used (deduced)
    ///                     representing the first sequence.
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam FwdIter2    The type of the source iterators used (deduced)
    ///                     representing the first sequence.
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam FwdIter3    The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     output iterator.
    /// \tparam Pred        The type of an optional function/function object to use.
    ///                     Unlike its sequential form, the parallel
    ///                     overload of \a set_intersection requires \a Pred to meet
    ///                     the requirements of \a CopyConstructible. This defaults
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
    /// \param dest         Refers to the beginning of the destination range.
    /// \param op           The binary predicate which returns true if the
    ///                     elements should be treated as equal. The signature
    ///                     of the predicate function should be equivalent to
    ///                     the following:
    ///                     \code
    ///                     bool pred(const Type1 &a, const Type1 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type1 must be such
    ///                     that objects of type \a InIter can
    ///                     be dereferenced and then implicitly converted to
    ///                     \a Type1
    ///
    /// The application of function objects in parallel algorithm
    /// invoked with a sequential execution policy object execute in sequential
    /// order in the calling thread (\a sequenced_policy) or in a
    /// single new thread spawned from the current thread
    /// (for \a sequenced_task_policy).
    ///
    /// The application of function objects in parallel algorithm
    /// invoked with an execution policy object of type
    /// \a parallel_policy or \a parallel_task_policy are
    /// permitted to execute in an unordered fashion in unspecified
    /// threads, and indeterminately sequenced within each thread.
    ///
    /// \returns  The \a set_intersection algorithm returns a \a hpx::future<FwdIter3>
    ///           if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a FwdIter3 otherwise.
    ///           The \a set_intersection algorithm returns the output iterator to the
    ///           element in the destination range, one past the last element
    ///           copied.
    ///
    template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
        typename FwdIter3, typename Pred = detail::less>
    typename util::detail::algorithm_result<ExPolicy, FwdIter3>::type
    set_intersection(ExPolicy&& policy, FwdIter1 first1, FwdIter1 last1,
            FwdIter2 first2, FwdIter2 last2, FwdIter3 dest, Pred&& op = Pred());

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
#include <hpx/parallel/algorithms/detail/set_operation.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/projection_identity.hpp>
#include <hpx/parallel/util/result_types.hpp>

#include <algorithm>
#include <iterator>
#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { inline namespace v1 {
    ///////////////////////////////////////////////////////////////////////////
    // set_intersection
    namespace detail {

        template <typename Iter1, typename Sent1, typename Iter2,
            typename Sent2, typename Iter3, typename Comp, typename Proj1,
            typename Proj2>
        constexpr parallel::util::in_in_out_result<Iter1, Iter2, Iter3>
        sequential_set_intersection(Iter1 first1, Sent1 last1, Iter2 first2,
            Sent2 last2, Iter3 dest, Comp&& comp, Proj1&& proj1, Proj2&& proj2)
        {
            while (first1 != last1 && first2 != last2)
            {
                auto&& value1 = hpx::util::invoke(proj1, *first1);
                auto&& value2 = hpx::util::invoke(proj2, *first2);

                if (hpx::util::invoke(comp, value1, value2))
                {
                    ++first1;
                }
                else
                {
                    if (!hpx::util::invoke(comp, value2, value1))
                    {
                        *dest++ = *first1++;
                    }
                    ++first2;
                }
            }
            return {first1, first2, dest};
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename Result>
        struct set_intersection
          : public detail::algorithm<set_intersection<Result>, Result>
        {
            set_intersection()
              : set_intersection::algorithm("set_intersection")
            {
            }

            template <typename ExPolicy, typename Iter1, typename Sent1,
                typename Iter2, typename Sent2, typename Iter3, typename F,
                typename Proj1, typename Proj2>
            static util::in_in_out_result<Iter1, Iter2, Iter3> sequential(
                ExPolicy, Iter1 first1, Sent1 last1, Iter2 first2, Sent2 last2,
                Iter3 dest, F&& f, Proj1&& proj1, Proj2&& proj2)
            {
                return sequential_set_intersection(first1, last1, first2, last2,
                    dest, std::forward<F>(f), std::forward<Proj1>(proj1),
                    std::forward<Proj2>(proj2));
            }

            template <typename ExPolicy, typename Iter1, typename Sent1,
                typename Iter2, typename Sent2, typename Iter3, typename F,
                typename Proj1, typename Proj2>
            static typename util::detail::algorithm_result<ExPolicy,
                util::in_in_out_result<Iter1, Iter2, Iter3>>::type
            parallel(ExPolicy&& policy, Iter1 first1, Sent1 last1, Iter2 first2,
                Sent2 last2, Iter3 dest, F&& f, Proj1&& proj1, Proj2&& proj2)
            {
                using difference_type1 =
                    typename std::iterator_traits<Iter1>::difference_type;
                using difference_type2 =
                    typename std::iterator_traits<Iter2>::difference_type;

                using result_type = util::in_in_out_result<Iter1, Iter2, Iter3>;
                using result =
                    util::detail::algorithm_result<ExPolicy, result_type>;

                if (first1 == last1 || first2 == last2)
                {
                    return result::get(result_type{
                        std::move(first1), std::move(first2), std::move(dest)});
                }

                using buffer_type = typename set_operations_buffer<Iter3>::type;
                using func_type = typename std::decay<F>::type;

                // calculate approximate destination index
                auto f1 = [](difference_type1 idx1,
                              difference_type2 idx2) -> difference_type1 {
                    return (std::min)(idx1, idx2);
                };

                // perform required set operation for one chunk
                auto f2 = [proj1, proj2](Iter1 part_first1, Sent1 part_last1,
                              Iter2 part_first2, Sent2 part_last2,
                              buffer_type* dest, func_type const& f) {
                    return sequential_set_intersection(part_first1, part_last1,
                        part_first2, part_last2, dest, f, proj1, proj2);
                };

                return set_operation(std::forward<ExPolicy>(policy), first1,
                    last1, first2, last2, dest, std::forward<F>(f),
                    std::forward<Proj1>(proj1), std::forward<Proj2>(proj2),
                    std::move(f1), std::move(f2));
            }
        };
    }    // namespace detail

    // clang-format off
    template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
        typename FwdIter3, typename Pred = detail::less,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy<ExPolicy>::value &&
            hpx::traits::is_iterator<FwdIter1>::value &&
            hpx::traits::is_iterator<FwdIter2>::value &&
            hpx::traits::is_iterator<FwdIter3>::value &&
            hpx::is_invocable_v<Pred,
                typename std::iterator_traits<FwdIter1>::value_type,
                typename std::iterator_traits<FwdIter2>::value_type
            >
        )>
    // clang-format on
    HPX_DEPRECATED_V(1, 6,
        "hpx::parallel::set_intersection is deprecated, use "
        "hpx::set_intersection instead")
        typename util::detail::algorithm_result<ExPolicy, FwdIter3>::type
        set_intersection(ExPolicy&& policy, FwdIter1 first1, FwdIter1 last1,
            FwdIter2 first2, FwdIter2 last2, FwdIter3 dest, Pred&& op = Pred())
    {
        static_assert((hpx::traits::is_forward_iterator<FwdIter1>::value),
            "Requires at least forward iterator.");
        static_assert((hpx::traits::is_forward_iterator<FwdIter2>::value),
            "Requires at least forward iterator.");
        static_assert((hpx::traits::is_forward_iterator<FwdIter3>::value),
            "Requires at least forward iterator.");

        using is_seq = std::integral_constant<bool,
            hpx::is_sequenced_execution_policy<ExPolicy>::value ||
                !hpx::traits::is_random_access_iterator<FwdIter1>::value ||
                !hpx::traits::is_random_access_iterator<FwdIter2>::value>;

        using result_type =
            parallel::util::in_in_out_result<FwdIter1, FwdIter2, FwdIter3>;

#if defined(HPX_GCC_VERSION) && HPX_GCC_VERSION >= 100000
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
        return util::get_third_element(
            detail::set_intersection<result_type>().call(
                std::forward<ExPolicy>(policy), is_seq(), first1, last1, first2,
                last2, dest, std::forward<Pred>(op),
                util::projection_identity(), util::projection_identity()));
#if defined(HPX_GCC_VERSION) && HPX_GCC_VERSION >= 100000
#pragma GCC diagnostic pop
#endif
    }
}}}    // namespace hpx::parallel::v1

namespace hpx {

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::set_intersection
    HPX_INLINE_CONSTEXPR_VARIABLE struct set_intersection_t final
      : hpx::functional::tag<set_intersection_t>
    {
    private:
        // clang-format off
        template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
            typename FwdIter3, typename Pred = hpx::parallel::v1::detail::less,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_iterator<FwdIter1>::value &&
                hpx::traits::is_iterator<FwdIter2>::value &&
                hpx::traits::is_iterator<FwdIter3>::value &&
                hpx::is_invocable_v<Pred,
                    typename std::iterator_traits<FwdIter1>::value_type,
                    typename std::iterator_traits<FwdIter2>::value_type
                >
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            FwdIter3>::type
        tag_invoke(set_intersection_t, ExPolicy&& policy, FwdIter1 first1,
            FwdIter1 last1, FwdIter2 first2, FwdIter2 last2, FwdIter3 dest,
            Pred&& op = Pred())
        {
            static_assert((hpx::traits::is_forward_iterator<FwdIter1>::value),
                "Requires at least forward iterator.");
            static_assert((hpx::traits::is_forward_iterator<FwdIter2>::value),
                "Requires at least forward iterator.");
            static_assert(hpx::traits::is_forward_iterator<FwdIter3>::value ||
                    (hpx::is_sequenced_execution_policy<ExPolicy>::value &&
                        hpx::traits::is_output_iterator<FwdIter3>::value),
                "Requires at least forward iterator or sequential execution.");

            using is_seq = std::integral_constant<bool,
                hpx::is_sequenced_execution_policy<ExPolicy>::value ||
                    !hpx::traits::is_random_access_iterator<FwdIter1>::value ||
                    !hpx::traits::is_random_access_iterator<FwdIter2>::value>;

            using result_type = hpx::parallel::util::in_in_out_result<FwdIter1,
                FwdIter2, FwdIter3>;

            return hpx::parallel::util::get_third_element(
                hpx::parallel::v1::detail::set_intersection<result_type>().call(
                    std::forward<ExPolicy>(policy), is_seq(), first1, last1,
                    first2, last2, dest, std::forward<Pred>(op),
                    hpx::parallel::util::projection_identity(),
                    hpx::parallel::util::projection_identity()));
        }

        // clang-format off
        template <typename FwdIter1, typename FwdIter2, typename FwdIter3,
            typename Pred = hpx::parallel::v1::detail::less,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator<FwdIter1>::value &&
                hpx::traits::is_iterator<FwdIter2>::value &&
                hpx::traits::is_iterator<FwdIter3>::value &&
                hpx::is_invocable_v<Pred,
                    typename std::iterator_traits<FwdIter1>::value_type,
                    typename std::iterator_traits<FwdIter2>::value_type
                >
            )>
        // clang-format on
        friend FwdIter3 tag_invoke(set_intersection_t, FwdIter1 first1,
            FwdIter1 last1, FwdIter2 first2, FwdIter2 last2, FwdIter3 dest,
            Pred&& op = Pred())
        {
            static_assert((hpx::traits::is_input_iterator<FwdIter1>::value),
                "Requires at least input iterator.");
            static_assert((hpx::traits::is_input_iterator<FwdIter2>::value),
                "Requires at least input iterator.");
            static_assert((hpx::traits::is_output_iterator<FwdIter3>::value),
                "Requires at least output iterator.");

            using result_type = hpx::parallel::util::in_in_out_result<FwdIter1,
                FwdIter2, FwdIter3>;

            return hpx::parallel::util::get_third_element(
                hpx::parallel::v1::detail::set_intersection<result_type>().call(
                    hpx::execution::seq, std::true_type(), first1, last1,
                    first2, last2, dest, std::forward<Pred>(op),
                    hpx::parallel::util::projection_identity(),
                    hpx::parallel::util::projection_identity()));
        }
    } set_intersection{};
}    // namespace hpx

#endif    // DOXYGEN
