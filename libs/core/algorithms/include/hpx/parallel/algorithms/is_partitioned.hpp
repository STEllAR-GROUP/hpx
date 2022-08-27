//  Copyright (c) 2020 ETH Zurich
//  Copyright (c) 2015 Daniel Bourgeois
//  Copyright (c) 2017-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#if defined(DOXYGEN)
namespace hpx {
    /// Determines if the range [first, last) is partitioned.
    ///
    /// \note   Complexity: at most (N) predicate evaluations where
    ///         \a N = distance(first, last).
    ///
    /// \tparam FwdIter     The type of the source iterators used for the
    ///                     This iterator type must meet the requirements of a
    ///                     forward iterator.
    /// \tparam Pred        The type of the function/function object to use
    ///                     (deduced).
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     of that the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements of
    ///                     that the algorithm will be applied to.
    /// \param pred         Refers to the unary predicate which returns true
    ///                     for elements expected to be found in the beginning
    ///                     of the range. The signature of the function
    ///                     should be equivalent to
    ///                     \code
    ///                     bool pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that objects of
    ///                     types \a FwdIter can be dereferenced and then
    ///                     implicitly converted to Type.
    ///
    /// \returns  The \a is_partitioned algorithm returns \a bool.
    ///           The \a is_partitioned algorithm returns true if each element
    ///           in the sequence for which pred returns true precedes those for
    ///           which pred returns false. Otherwise is_partitioned returns
    ///           false. If the range [first, last) contains less than two
    ///           elements, the function is always true.
    ///
    template <typename FwdIter, typename Pred>
    bool is_partitioned(FwdIter first, FwdIter last, Pred&& pred);

    /// Determines if the range [first, last) is partitioned.
    ///
    /// \note   Complexity: at most (N) predicate evaluations where
    ///         \a N = distance(first, last).
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter     The type of the source iterators used for the
    ///                     This iterator type must meet the requirements of a
    ///                     forward iterator.
    /// \tparam Pred        The type of the function/function object to use
    ///                     (deduced). \a Pred must be \a CopyConstructible
    ///                     when using a parallel policy.
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     of that the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements of
    ///                     that the algorithm will be applied to.
    /// \param pred         Refers to the unary predicate which returns true
    ///                     for elements expected to be found in the beginning
    ///                     of the range. The signature of the function
    ///                     should be equivalent to
    ///                     \code
    ///                     bool pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that objects of
    ///                     types \a FwdIter can be dereferenced and then
    ///                     implicitly converted to Type.
    ///
    /// The predicate operations in the parallel \a is_partitioned algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// executes in sequential order in the calling thread.
    ///
    /// The comparison operations in the parallel \a is_partitioned algorithm invoked
    /// with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a is_partitioned algorithm returns a \a hpx::future<bool>
    ///           if the execution policy is of type \a task_execution_policy
    ///           and returns \a bool otherwise.
    ///           The \a is_partitioned algorithm returns true if each element
    ///           in the sequence for which pred returns true precedes those for
    ///           which pred returns false. Otherwise is_partitioned returns
    ///           false. If the range [first, last) contains less than two
    ///           elements, the function is always true.
    ///
    template <typename ExPolicy, typename FwdIter, typename Pred>
    typename util::detail::algorithm_result<ExPolicy, bool>::type
    is_partitioned(ExPolicy&& policy, FwdIter first, FwdIter last, Pred&& pred);
}    // namespace hpx
#else

#include <hpx/config.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/functional/invoke.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/util/cancellation_token.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/detail/sender_util.hpp>
#include <hpx/parallel/util/invoke_projected.hpp>
#include <hpx/parallel/util/loop.hpp>
#include <hpx/parallel/util/partitioner.hpp>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace parallel { inline namespace v1 {
    ////////////////////////////////////////////////////////////////////////////
    // is_partitioned
    namespace detail {
        /// \cond NOINTERNAL
        inline bool sequential_is_partitioned(
            std::vector<hpx::future<bool>>&& res)
        {
            std::vector<hpx::future<bool>>::iterator first = res.begin();
            std::vector<hpx::future<bool>>::iterator last = res.end();
            while (first != last && first->get())
            {
                ++first;
            }
            if (first != last)
            {
                ++first;
                while (first != last)
                {
                    if (first->get())
                        return false;
                    ++first;
                }
            }
            return true;
        }

        template <typename Iter, typename Sent>
        struct is_partitioned
          : public detail::algorithm<is_partitioned<Iter, Sent>, bool>
        {
            is_partitioned()
              : is_partitioned::algorithm("is_partitioned")
            {
            }

            template <typename ExPolicy, typename InIter, typename InSent,
                typename Pred, typename Proj>
            static bool sequential(
                ExPolicy, InIter first, InSent last, Pred&& pred, Proj&& proj)
            {
                return std::is_partitioned(first, last,
                    util::invoke_projected<Pred, Proj>(
                        HPX_FORWARD(Pred, pred), HPX_FORWARD(Proj, proj)));
            }

            template <typename ExPolicy, typename Pred, typename Proj>
            static typename util::detail::algorithm_result<ExPolicy, bool>::type
            parallel(ExPolicy&& policy, Iter first, Sent last, Pred&& pred,
                Proj&& proj)
            {
                typedef typename std::iterator_traits<Iter>::difference_type
                    difference_type;
                typedef typename util::detail::algorithm_result<ExPolicy, bool>
                    result;

                difference_type count = std::distance(first, last);
                if (count <= 1)
                    return result::get(true);

                util::invoke_projected<Pred, Proj> pred_projected(
                    HPX_FORWARD(Pred, pred), HPX_FORWARD(Proj, proj));

                util::cancellation_token<> tok;

                // Note: replacing the invoke() with HPX_INVOKE()
                // below makes gcc generate errors
                auto f1 = [tok, pred_projected = HPX_MOVE(pred_projected)](
                              Iter part_begin,
                              std::size_t part_count) mutable -> bool {
                    bool fst_bool = HPX_INVOKE(pred_projected, *part_begin);
                    if (part_count == 1)
                        return fst_bool;

                    util::loop_n<std::decay_t<ExPolicy>>(++part_begin,
                        --part_count, tok,
                        [&fst_bool, &pred_projected, &tok](
                            Iter const& a) mutable -> void {
                            if (fst_bool !=
                                hpx::util::invoke(pred_projected, *a))
                            {
                                if (fst_bool)
                                    fst_bool = false;
                                else
                                    tok.cancel();
                            }
                        });

                    return fst_bool;
                };

                auto f2 = [tok](auto&& results) -> bool {
                    if (tok.was_cancelled())
                        return false;
                    return sequential_is_partitioned(HPX_MOVE(results));
                };

                return util::partitioner<ExPolicy, bool>::call(
                    HPX_FORWARD(ExPolicy, policy), first, count, HPX_MOVE(f1),
                    HPX_MOVE(f2));
            }
        };
        /// \endcond
    }    // namespace detail
}}}      // namespace hpx::parallel::v1

namespace hpx {
    inline constexpr struct is_partitioned_t final
      : hpx::detail::tag_parallel_algorithm<is_partitioned_t>
    {
    private:
        // clang-format off
        template <typename FwdIter, typename Pred,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_forward_iterator<FwdIter>::value
            )>
        // clang-format on
        friend bool tag_fallback_invoke(
            hpx::is_partitioned_t, FwdIter first, FwdIter last, Pred&& pred)
        {
            return hpx::parallel::v1::detail::is_partitioned<FwdIter, FwdIter>()
                .call(hpx::execution::seq, first, last, HPX_FORWARD(Pred, pred),
                    hpx::parallel::util::projection_identity());
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter,
            typename Pred,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_forward_iterator<FwdIter>::value
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            bool>::type
        tag_fallback_invoke(hpx::is_partitioned_t, ExPolicy&& policy,
            FwdIter first, FwdIter last, Pred&& pred)
        {
            return hpx::parallel::v1::detail::is_partitioned<FwdIter, FwdIter>()
                .call(HPX_FORWARD(ExPolicy, policy), first, last,
                    HPX_FORWARD(Pred, pred),
                    hpx::parallel::util::projection_identity());
        }
    } is_partitioned{};
}    // namespace hpx

#endif
