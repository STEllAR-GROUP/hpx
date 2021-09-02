//  Copyright (c) 2014 Grant Mercer
//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/count.hpp

#pragma once

#if defined(DOXYGEN)
namespace hpx {
    // clang-format off

    /// Returns the number of elements in the range [first, last) satisfying
    /// a specific criteria. This version counts the elements that are equal to
    /// the given \a value.
    ///
    /// \note   Complexity: Performs exactly \a last - \a first comparisons.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the comparisons.
    /// \tparam FwdIterB    The type of the source begin iterator used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam FwdIterE    The type of the source end iterator used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam T           The type of the value to search for (deduced).
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param value        The value to search for.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The comparisons in the parallel \a count algorithm invoked with
    /// an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// \note The comparisons in the parallel \a count algorithm invoked with
    ///       an execution policy object of type \a parallel_policy or
    ///       \a parallel_task_policy are permitted to execute in an unordered
    ///       fashion in unspecified threads, and indeterminately sequenced
    ///       within each thread.
    ///
    /// \returns  The \a count algorithm returns a
    ///           \a hpx::future<difference_type> if the execution policy is of
    ///           type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a difference_type otherwise (where \a difference_type
    ///           is defined by \a std::iterator_traits<FwdIterB>::difference_type.
    ///           The \a count algorithm returns the number of elements
    ///           satisfying the given criteria.
    ///
    template <typename ExPolicy, typename FwdIterB, typename FwdIterE,
        typename T, typename Proj = util::projection_identity>
    typename util::detail::algorithm_result<ExPolicy,
        typename std::iterator_traits<FwdIterB>::difference_type>::type
    count(ExPolicy&& policy, FwdIterB first, FwdIterE last, T const& value,
            Proj&& proj = Proj());

    /// Returns the number of elements in the range [first, last) satisfying
    /// a specific criteria. This version counts elements for which predicate
    /// \a f returns true.
    ///
    /// \note   Complexity: Performs exactly \a last - \a first applications of
    ///         the predicate.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the comparisons.
    /// \tparam Iter        The type of the source begin iterator used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Sent        The type of the source end iterator used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a count_if requires \a F to meet the
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
    /// \param f            Specifies the function (or function object) which
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
    ///                     type \a FwdIterB can be dereferenced and then
    ///                     implicitly converted to Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// \note The assignments in the parallel \a count_if algorithm invoked with
    ///       an execution policy object of type \a sequenced_policy
    ///       execute in sequential order in the calling thread.
    /// \note The assignments in the parallel \a count_if algorithm invoked with
    ///       an execution policy object of type \a parallel_policy or
    ///       \a parallel_task_policy are permitted to execute in an unordered
    ///       fashion in unspecified threads, and indeterminately sequenced
    ///       within each thread.
    ///
    /// \returns  The \a count_if algorithm returns
    ///           \a hpx::future<difference_type> if the execution policy is of
    ///           type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a difference_type otherwise (where \a difference_type
    ///           is defined by \a std::iterator_traits<FwdIterB>::difference_type.
    ///           The \a count algorithm returns the number of elements
    ///           satisfying the given criteria.
    ///
    template <typename ExPolicy, typename Iter, typename Sent,
        typename F, typename Proj = util::projection_identity>
    typename util::detail::algorithm_result<ExPolicy,
        typename std::iterator_traits<Iter>::difference_type>::type
    count_if(ExPolicy&& policy, Iter first, Sent last, F&& f,
        Proj&& proj = Proj());

    // clang-format on
}    // namespace hpx

#else    // DOXYGEN

#include <hpx/local/config.hpp>
#include <hpx/algorithms/traits/segmented_iterator_traits.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/functional/bind_back.hpp>
#include <hpx/functional/invoke.hpp>
#include <hpx/iterator_support/range.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/pack_traversal/unwrap.hpp>
#include <hpx/parallel/util/detail/sender_util.hpp>

#include <hpx/algorithms/traits/projected.hpp>
#include <hpx/execution/traits/vector_pack_count_bits.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/detail/distance.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
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
    ///////////////////////////////////////////////////////////////////////////
    // count
    namespace detail {

        /// \cond NOINTERNAL
        template <typename ExPolicy, typename Op, typename Proj>
        struct count_iteration
        {
            using execution_policy_type = typename std::decay<ExPolicy>::type;
            using proj_type = typename std::decay<Proj>::type;
            using op_type = typename std::decay<Op>::type;

            op_type op_;
            proj_type proj_;

            template <typename Op_, typename Proj_,
                typename U = typename std::enable_if<
                    !std::is_same<typename std::decay<Op_>::type,
                        count_iteration>::value>::type>
            HPX_HOST_DEVICE count_iteration(Op_&& op, Proj_&& proj)
              : op_(std::forward<Op_>(op))
              , proj_(std::forward<Proj_>(proj))
            {
            }

#if !defined(__NVCC__) && !defined(__CUDACC__)
            count_iteration(count_iteration const&) = default;
            count_iteration(count_iteration&&) = default;
#else
            HPX_HOST_DEVICE count_iteration(count_iteration const& rhs)
              : op_(rhs.op_)
              , proj_(rhs.proj_)
            {
            }

            HPX_HOST_DEVICE count_iteration(count_iteration&& rhs)
              : op_(std::move(rhs.op_))
              , proj_(std::move(rhs.proj_))
            {
            }
#endif

            count_iteration& operator=(count_iteration const&) = default;
            count_iteration& operator=(count_iteration&&) = default;

            template <typename Iter>
            HPX_HOST_DEVICE HPX_FORCEINLINE constexpr
                typename std::iterator_traits<Iter>::difference_type
                operator()(Iter part_begin, std::size_t part_size)
            {
                typename std::iterator_traits<Iter>::difference_type ret = 0;
                util::loop_n<execution_policy_type>(part_begin, part_size,
                    hpx::util::bind_back(*this, std::ref(ret)));
                return ret;
            }

            template <typename Iter>
            HPX_HOST_DEVICE HPX_FORCEINLINE constexpr void operator()(Iter curr,
                typename std::iterator_traits<Iter>::difference_type& ret)
            {
                ret += traits::count_bits(
                    hpx::util::invoke(op_, hpx::util::invoke(proj_, *curr)));
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Value>
        struct count : public detail::algorithm<count<Value>, Value>
        {
            typedef Value difference_type;

            count()
              : count::algorithm("count")
            {
            }

            template <typename ExPolicy, typename InIterB, typename InIterE,
                typename T, typename Proj>
            static difference_type sequential(ExPolicy&& policy, InIterB first,
                InIterE last, T const& value, Proj&& proj)
            {
                auto f1 =
                    count_iteration<ExPolicy, detail::compare_to<T>, Proj>(
                        detail::compare_to<T>(value), std::forward<Proj>(proj));

                typename std::iterator_traits<InIterB>::difference_type ret = 0;

                util::loop(std::forward<ExPolicy>(policy), first, last,
                    hpx::util::bind_back(std::move(f1), std::ref(ret)));

                return ret;
            }

            template <typename ExPolicy, typename IterB, typename IterE,
                typename T, typename Proj>
            static typename util::detail::algorithm_result<ExPolicy,
                difference_type>::type
            parallel(ExPolicy&& policy, IterB first, IterE last, T const& value,
                Proj&& proj)
            {
                if (first == last)
                {
                    return util::detail::algorithm_result<ExPolicy,
                        difference_type>::get(0);
                }

                auto f1 =
                    count_iteration<ExPolicy, detail::compare_to<T>, Proj>(
                        detail::compare_to<T>(value), std::forward<Proj>(proj));

                return util::partitioner<ExPolicy, difference_type>::call(
                    std::forward<ExPolicy>(policy), first,
                    detail::distance(first, last), std::move(f1),
                    hpx::unwrapping([](std::vector<difference_type>&& results) {
                        return util::accumulate_n(hpx::util::begin(results),
                            hpx::util::size(results), difference_type(0),
                            std::plus<difference_type>());
                    }));
            }
        };
        /// \endcond
    }    // namespace detail

    // clang-format off
    template <typename ExPolicy, typename FwdIterB, typename FwdIterE,
        typename T, typename Proj = util::projection_identity,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy<ExPolicy>::value &&
            traits::is_projected<Proj, FwdIterB>::value &&
            hpx::traits::is_iterator<FwdIterB>::value
        )>
    // clang-format on
    HPX_LOCAL_DEPRECATED_V(
        1, 6, "hpx::parallel::count is deprecated, use hpx::count instead")
        typename util::detail::algorithm_result<ExPolicy,
            typename std::iterator_traits<FwdIterB>::difference_type>::type
        count(ExPolicy&& policy, FwdIterB first, FwdIterE last, T const& value,
            Proj&& proj = Proj())
    {
        static_assert((hpx::traits::is_forward_iterator<FwdIterB>::value),
            "Required at least forward iterator.");

        using difference_type =
            typename std::iterator_traits<FwdIterB>::difference_type;

#if defined(HPX_GCC_VERSION) && HPX_GCC_VERSION >= 100000
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
        return hpx::parallel::v1::detail::count<difference_type>().call(
            std::forward<ExPolicy>(policy), first, last, value,
            std::forward<Proj>(proj));
#if defined(HPX_GCC_VERSION) && HPX_GCC_VERSION >= 100000
#pragma GCC diagnostic pop
#endif
    }

    ///////////////////////////////////////////////////////////////////////////
    // count_if
    namespace detail {
        /// \cond NOINTERNAL
        template <typename Value>
        struct count_if : public detail::algorithm<count_if<Value>, Value>
        {
            typedef Value difference_type;

            count_if()
              : count_if::algorithm("count_if")
            {
            }

            template <typename ExPolicy, typename InIterB, typename InIterE,
                typename Pred, typename Proj>
            static difference_type sequential(ExPolicy&& policy, InIterB first,
                InIterE last, Pred&& op, Proj&& proj)
            {
                auto f1 = count_iteration<ExPolicy, Pred, Proj>(
                    op, std::forward<Proj>(proj));

                typename std::iterator_traits<InIterB>::difference_type ret = 0;

                util::loop(std::forward<ExPolicy>(policy), first, last,
                    hpx::util::bind_back(std::move(f1), std::ref(ret)));

                return ret;
            }

            template <typename ExPolicy, typename IterB, typename IterE,
                typename Pred, typename Proj>
            static typename util::detail::algorithm_result<ExPolicy,
                difference_type>::type
            parallel(ExPolicy&& policy, IterB first, IterE last, Pred&& op,
                Proj&& proj)
            {
                if (first == last)
                {
                    return util::detail::algorithm_result<ExPolicy,
                        difference_type>::get(0);
                }

                auto f1 = count_iteration<ExPolicy, Pred, Proj>(
                    op, std::forward<Proj>(proj));

                return util::partitioner<ExPolicy, difference_type>::call(
                    std::forward<ExPolicy>(policy), first,
                    detail::distance(first, last), std::move(f1),
                    hpx::unwrapping([](std::vector<difference_type>&& results) {
                        return util::accumulate_n(hpx::util::begin(results),
                            hpx::util::size(results), difference_type(0),
                            std::plus<difference_type>());
                    }));
            }
        };
        /// \endcond
    }    // namespace detail

    // clang-format off
    template <typename ExPolicy, typename FwdIterB, typename FwdIterE,
        typename F, typename Proj = util::projection_identity,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy<ExPolicy>::value &&
            hpx::traits::is_iterator<FwdIterB>::value &&
            traits::is_projected<Proj, FwdIterB>::value &&
            traits::is_indirect_callable<ExPolicy, F,
                traits::projected<Proj, FwdIterB>
            >::value
        )>
    // clang-format on
    HPX_LOCAL_DEPRECATED_V(
        1, 6, "hpx::parallel::count is deprecated, use hpx::count instead")
        typename util::detail::algorithm_result<ExPolicy,
            typename std::iterator_traits<FwdIterB>::difference_type>::type
        count_if(ExPolicy&& policy, FwdIterB first, FwdIterE last, F&& f,
            Proj&& proj = Proj())
    {
        static_assert((hpx::traits::is_forward_iterator<FwdIterB>::value),
            "Required at least forward iterator.");

        using difference_type =
            typename std::iterator_traits<FwdIterB>::difference_type;

#if defined(HPX_GCC_VERSION) && HPX_GCC_VERSION >= 100000
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
        return hpx::parallel::v1::detail::count_if<difference_type>().call(
            std::forward<ExPolicy>(policy), first, last, std::forward<F>(f),
            std::forward<Proj>(proj));
#if defined(HPX_GCC_VERSION) && HPX_GCC_VERSION >= 100000
#pragma GCC diagnostic pop
#endif
    }
}}}    // namespace hpx::parallel::v1

namespace hpx {

    ///////////////////////////////////////////////////////////////////////////
    // DPO for hpx::count
    HPX_INLINE_CONSTEXPR_VARIABLE struct count_t final
      : hpx::detail::tag_parallel_algorithm<count_t>
    {
    private:
        // clang-format off
        template <typename ExPolicy, typename FwdIter, typename T,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_iterator<FwdIter>::value
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            typename std::iterator_traits<FwdIter>::difference_type>::type
        tag_fallback_dispatch(count_t, ExPolicy&& policy, FwdIter first,
            FwdIter last, T const& value)
        {
            static_assert((hpx::traits::is_forward_iterator<FwdIter>::value),
                "Required at least forward iterator.");

            using difference_type =
                typename std::iterator_traits<FwdIter>::difference_type;

            return hpx::parallel::v1::detail::count<difference_type>().call(
                std::forward<ExPolicy>(policy), first, last, value,
                hpx::parallel::util::projection_identity{});
        }

        // clang-format off
        template <typename InIter, typename T,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator<InIter>::value
            )>
        // clang-format on
        friend typename std::iterator_traits<InIter>::difference_type
        tag_fallback_dispatch(
            count_t, InIter first, InIter last, T const& value)
        {
            static_assert((hpx::traits::is_input_iterator<InIter>::value),
                "Required at least input iterator.");

            using difference_type =
                typename std::iterator_traits<InIter>::difference_type;

            return hpx::parallel::v1::detail::count<difference_type>().call(
                hpx::execution::seq, first, last, value,
                hpx::parallel::util::projection_identity{});
        }
    } count{};

    ///////////////////////////////////////////////////////////////////////////
    // DPO for hpx::count_if
    HPX_INLINE_CONSTEXPR_VARIABLE struct count_if_t final
      : hpx::detail::tag_parallel_algorithm<count_if_t>
    {
    private:
        // clang-format off
        template <typename ExPolicy, typename FwdIter, typename F,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_iterator<FwdIter>::value &&
                hpx::is_invocable_v<F,
                    typename std::iterator_traits<FwdIter>::value_type
                >
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            typename std::iterator_traits<FwdIter>::difference_type>::type
        tag_fallback_dispatch(
            count_if_t, ExPolicy&& policy, FwdIter first, FwdIter last, F&& f)
        {
            static_assert((hpx::traits::is_forward_iterator<FwdIter>::value),
                "Required at least forward iterator.");

            using difference_type =
                typename std::iterator_traits<FwdIter>::difference_type;

            return hpx::parallel::v1::detail::count_if<difference_type>().call(
                std::forward<ExPolicy>(policy), first, last, std::forward<F>(f),
                hpx::parallel::util::projection_identity{});
        }

        // clang-format off
        template <typename InIter, typename F,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator<InIter>::value &&
                hpx::is_invocable_v<F,
                    typename std::iterator_traits<InIter>::value_type
                >
            )>
        // clang-format on
        friend typename std::iterator_traits<InIter>::difference_type
        tag_fallback_dispatch(count_if_t, InIter first, InIter last, F&& f)
        {
            static_assert((hpx::traits::is_input_iterator<InIter>::value),
                "Required at least input iterator.");

            using difference_type =
                typename std::iterator_traits<InIter>::difference_type;

            return hpx::parallel::v1::detail::count_if<difference_type>().call(
                hpx::execution::seq, first, last, std::forward<F>(f),
                hpx::parallel::util::projection_identity{});
        }
    } count_if{};

}    // namespace hpx

#endif    // DOXYGEN
