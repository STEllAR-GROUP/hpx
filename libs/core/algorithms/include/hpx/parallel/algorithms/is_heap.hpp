//  Copyright (c) 2017 Taeguk Kwon
//  Copyright (c) 2020-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#if defined(DOXYGEN)
namespace hpx {
    // clang-format off

    /// Returns whether the range is max heap. That is, true if the range is
    /// max heap, false otherwise. The function uses the given comparison
    /// function object \a comp (defaults to using operator<()).
    /// Executed according to the policy.
    ///
    /// \note   Complexity:
    ///         Linear in the distance between \a first and \a last.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam RandIter    The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     random access iterator.
    /// \tparam Comp        The type of the function/function object to use
    ///                     (deduced).
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param comp         \a comp is a callable object. The return value of the
    ///                     INVOKE operation applied to an object of type \a Comp,
    ///                     when contextually converted to bool, yields true if
    ///                     the first argument of the call is less than the
    ///                     second, and false otherwise. It is assumed that comp
    ///                     will not apply any non-constant function through the
    ///                     dereferenced iterator.
    ///
    /// \a comp has to induce a strict weak ordering on the values.
    ///
    /// The application of function objects in parallel algorithm
    /// invoked with an execution policy object of type
    /// \a sequenced_policy execute in sequential order in the
    /// calling thread.
    ///
    /// The application of function objects in parallel algorithm
    /// invoked with an execution policy object of type
    /// \a parallel_policy or \a parallel_task_policy are
    /// permitted to execute in an unordered fashion in unspecified
    /// threads, and indeterminately sequenced within each thread.
    ///
    /// \returns  The \a is_heap algorithm returns a \a hpx::future<bool>
    ///           if the execution policy is of type \a sequenced_task_policy or
    ///           \a parallel_task_policy and returns \a bool otherwise.
    ///           The \a is_heap algorithm returns whether the range is max heap.
    ///           That is, true if the range is max heap, false otherwise.
    ///
    template <typename ExPolicy, typename RandIter,
        typename Comp = hpx::parallel::detail::less>
    hpx::parallel::util::detail::algorithm_result_t<ExPolicy, bool>
    is_heap(ExPolicy&& policy, RandIter first, RandIter last,
        Comp&& comp = Comp());

    /// Returns whether the range is max heap. That is, true if the range is
    /// max heap, false otherwise. The function uses the given comparison
    /// function object \a comp (defaults to using operator<()).
    ///
    /// \note   Complexity:
    ///         Linear in the distance between \a first and \a last.
    ///
    /// \tparam RandIter    The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     random access iterator.
    /// \tparam Comp        The type of the function/function object to use
    ///                     (deduced).
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param comp         \a comp is a callable object. The return value of the
    ///                     INVOKE operation applied to an object of type \a Comp,
    ///                     when contextually converted to bool, yields true if
    ///                     the first argument of the call is less than the
    ///                     second, and false otherwise. It is assumed that comp
    ///                     will not apply any non-constant function through the
    ///                     dereferenced iterator.
    ///
    /// \a comp has to induce a strict weak ordering on the values.
    ///
    /// \returns  The \a is_heap a \a bool.
    ///           The \a is_heap algorithm returns whether the range is max heap.
    ///           That is, true if the range is max heap, false otherwise.
    ///
    template <typename RandIter, typename Comp = hpx::parallel::detail::less>
    bool is_heap(RandIter first, RandIter last, Comp&& comp = Comp());

    /// Returns the upper bound of the largest range beginning at \a first
    /// which is a max heap. That is, the last iterator \a it for
    /// which range [first, it) is a max heap. The function
    /// uses the given comparison function object \a comp (defaults to using
    /// operator<()). Executed according to the policy.
    ///
    /// \note   Complexity:
    ///         Linear in the distance between \a first and \a last.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam RandIter    The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     random access iterator.
    /// \tparam Comp        The type of the function/function object to use
    ///                     (deduced).
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param comp         \a comp is a callable object. The return value of the
    ///                     INVOKE operation applied to an object of type \a Comp,
    ///                     when contextually converted to bool, yields true if
    ///                     the first argument of the call is less than the
    ///                     second, and false otherwise. It is assumed that comp
    ///                     will not apply any non-constant function through the
    ///                     dereferenced iterator.
    ///
    /// \a comp has to induce a strict weak ordering on the values.
    ///
    /// The application of function objects in parallel algorithm
    /// invoked with an execution policy object of type
    /// \a sequenced_policy execute in sequential order in the
    /// calling thread.
    ///
    /// The application of function objects in parallel algorithm
    /// invoked with an execution policy object of type
    /// \a parallel_policy or \a parallel_task_policy are
    /// permitted to execute in an unordered fashion in unspecified
    /// threads, and indeterminately sequenced within each thread.
    ///
    /// \returns  The \a is_heap_until algorithm returns a \a hpx::future<RandIter>
    ///           if the execution policy is of type \a sequenced_task_policy or
    ///           \a parallel_task_policy and returns \a RandIter otherwise.
    ///           The \a is_heap_until algorithm returns the upper bound
    ///           of the largest range beginning at first which is a max heap.
    ///           That is, the last iterator \a it for which range [first, it)
    ///           is a max heap.
    ///
    template <typename ExPolicy, typename RandIter,
        typename Comp = hpx::parallel::detail::less>
    hpx::parallel::util::detail::algorithm_result_t<ExPolicy, RandIter>
    is_heap_until(ExPolicy&& policy, RandIter first, RandIter last,
        Comp&& comp = Comp());

    /// Returns the upper bound of the largest range beginning at \a first
    /// which is a max heap. That is, the last iterator \a it for
    /// which range [first, it) is a max heap. The function
    /// uses the given comparison function object \a comp (defaults to using
    /// operator<()).
    ///
    /// \note   Complexity:
    ///         Linear in the distance between \a first and \a last.
    ///
    /// \tparam RandIter    The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     random access iterator.
    /// \tparam Comp        The type of the function/function object to use
    ///                     (deduced).
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param comp         \a comp is a callable object. The return value of the
    ///                     INVOKE operation applied to an object of type \a Comp,
    ///                     when contextually converted to bool, yields true if
    ///                     the first argument of the call is less than the
    ///                     second, and false otherwise. It is assumed that comp
    ///                     will not apply any non-constant function through the
    ///                     dereferenced iterator.
    ///
    /// \a comp has to induce a strict weak ordering on the values.
    ///
    /// \returns  The \a is_heap_until algorithm returns a \a RandIter.
    ///           The \a is_heap_until algorithm returns the upper bound
    ///           of the largest range beginning at first which is a max heap.
    ///           That is, the last iterator \a it for which range [first, it)
    ///           is a max heap.
    ///
    template <typename RandIter, typename Comp = hpx::parallel::detail::less>
    RandIter is_heap_until(RandIter first, RandIter last, Comp&& comp = Comp());

    // clang-format on
}    // namespace hpx

#else    // DOXYGEN

#include <hpx/config.hpp>
#include <hpx/algorithms/traits/projected.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/coroutines/thread_enums.hpp>
#include <hpx/execution/executors/execution.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/functional/invoke.hpp>
#include <hpx/functional/traits/is_invocable.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/detail/distance.hpp>
#include <hpx/parallel/util/adapt_placement_mode.hpp>
#include <hpx/parallel/util/cancellation_token.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/detail/clear_container.hpp>
#include <hpx/parallel/util/detail/sender_util.hpp>
#include <hpx/parallel/util/loop.hpp>
#include <hpx/parallel/util/partitioner.hpp>
#include <hpx/type_support/identity.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>

namespace hpx::parallel {

    ///////////////////////////////////////////////////////////////////////////
    // is_heap
    namespace detail {

        // sequential is_heap with projection function
        template <typename Iter, typename Sent, typename Comp, typename Proj>
        constexpr bool sequential_is_heap(
            Iter first, Sent last, Comp&& comp, Proj&& proj)
        {
            using difference_type =
                typename std::iterator_traits<Iter>::difference_type;

            difference_type count = detail::distance(first, last);

            for (difference_type i = 1; i < count; ++i)
            {
                if (HPX_INVOKE(comp, HPX_INVOKE(proj, *(first + (i - 1) / 2)),
                        HPX_INVOKE(proj, *(first + i))))
                    return false;
            }
            return true;
        }

        struct is_heap_helper
        {
            template <typename ExPolicy, typename Iter, typename Sent,
                typename Comp, typename Proj>
            util::detail::algorithm_result_t<ExPolicy, bool> operator()(
                ExPolicy&& orgpolicy, Iter first, Sent last, Comp&& comp,
                Proj&& proj)
            {
                using result = util::detail::algorithm_result<ExPolicy, bool>;
                using type = typename std::iterator_traits<Iter>::value_type;
                using difference_type =
                    typename std::iterator_traits<Iter>::difference_type;

                difference_type count = detail::distance(first, last);
                if (count <= 1)
                {
                    return result::get(true);
                }

                Iter second = first + 1;
                --count;

                decltype(auto) policy = parallel::util::adapt_placement_mode(
                    HPX_FORWARD(ExPolicy, orgpolicy),
                    hpx::threads::thread_placement_hint::breadth_first);

                using policy_type = std::decay_t<decltype(policy)>;

                hpx::parallel::util::cancellation_token<std::size_t> tok(count);

                // Note: replacing the invoke() with HPX_INVOKE()
                // below makes gcc generate errors
                auto f1 = [tok, first, comp = HPX_FORWARD(Comp, comp),
                              proj = HPX_FORWARD(Proj, proj)](Iter it,
                              std::size_t part_size,
                              std::size_t base_idx) mutable -> void {
                    util::loop_idx_n<policy_type>(base_idx, it, part_size, tok,
                        [&tok, first, &comp, &proj](
                            type const& v, std::size_t i) mutable -> void {
                            if (hpx::invoke(comp,
                                    hpx::invoke(proj, *(first + i / 2)),
                                    hpx::invoke(proj, v)))
                            {
                                tok.cancel(0);
                            }
                        });
                };

                auto f2 = [tok](auto&& data) mutable -> bool {
                    // make sure iterators embedded in function object that is
                    // attached to futures are invalidated
                    util::detail::clear_container(data);

                    difference_type find_res =
                        static_cast<difference_type>(tok.get_data());

                    return find_res != 0;
                };

                using partitioner_type =
                    util::partitioner<policy_type, bool, void>;
                return partitioner_type::call_with_index(
                    HPX_FORWARD(decltype(policy), policy), second, count, 1,
                    HPX_MOVE(f1), HPX_MOVE(f2));
            }
        };

        template <typename RandIter>
        struct is_heap : public algorithm<is_heap<RandIter>, bool>
        {
            constexpr is_heap() noexcept
              : algorithm<is_heap, bool>("is_heap")
            {
            }

            template <typename ExPolicy, typename Iter, typename Sent,
                typename Comp, typename Proj>
            static constexpr bool sequential(
                ExPolicy&&, Iter first, Sent last, Comp&& comp, Proj&& proj)
            {
                return sequential_is_heap(first, last, HPX_FORWARD(Comp, comp),
                    HPX_FORWARD(Proj, proj));
            }

            template <typename ExPolicy, typename Iter, typename Sent,
                typename Comp, typename Proj>
            static util::detail::algorithm_result_t<ExPolicy, bool> parallel(
                ExPolicy&& policy, Iter first, Sent last, Comp&& comp,
                Proj&& proj)
            {
                return is_heap_helper()(HPX_FORWARD(ExPolicy, policy), first,
                    last, HPX_FORWARD(Comp, comp), HPX_FORWARD(Proj, proj));
            }
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    // is_heap_until
    namespace detail {

        // sequential is_heap_until with projection function
        template <typename Iter, typename Sent, typename Comp, typename Proj>
        constexpr Iter sequential_is_heap_until(
            Iter first, Sent last, Comp&& comp, Proj&& proj)
        {
            using difference_type =
                typename std::iterator_traits<Iter>::difference_type;

            difference_type count = detail::distance(first, last);

            for (difference_type i = 1; i < count; ++i)
            {
                if (HPX_INVOKE(comp, HPX_INVOKE(proj, *(first + (i - 1) / 2)),
                        HPX_INVOKE(proj, *(first + i))))
                    return first + i;
            }
            return last;
        }

        struct is_heap_until_helper
        {
            template <typename ExPolicy, typename Iter, typename Sent,
                typename Comp, typename Proj>
            util::detail::algorithm_result_t<ExPolicy, Iter> operator()(
                ExPolicy&& orgpolicy, Iter first, Sent last, Comp comp,
                Proj proj)
            {
                using result = util::detail::algorithm_result<ExPolicy, Iter>;
                using type = typename std::iterator_traits<Iter>::value_type;
                using difference_type =
                    typename std::iterator_traits<Iter>::difference_type;

                difference_type count = detail::distance(first, last);
                if (count <= 1)
                {
                    return result::get(HPX_MOVE(last));
                }

                Iter second = first + 1;
                --count;

                decltype(auto) policy = parallel::util::adapt_placement_mode(
                    HPX_FORWARD(ExPolicy, orgpolicy),
                    hpx::threads::thread_placement_hint::breadth_first);

                using policy_type = std::decay_t<decltype(policy)>;

                hpx::parallel::util::cancellation_token<std::size_t> tok(count);

                // Note: replacing the invoke() with HPX_INVOKE()
                // below makes gcc generate errors
                auto f1 = [tok, first, comp = HPX_FORWARD(Comp, comp),
                              proj = HPX_FORWARD(Proj, proj)](Iter it,
                              std::size_t part_size,
                              std::size_t base_idx) mutable {
                    util::loop_idx_n<policy_type>(base_idx, it, part_size, tok,
                        [&tok, first, &comp, &proj](
                            type const& v, std::size_t i) -> void {
                            if (hpx::invoke(comp,
                                    hpx::invoke(proj, *(first + i / 2)),
                                    hpx::invoke(proj, v)))
                            {
                                tok.cancel(i);
                            }
                        });
                };

                auto f2 = [tok, second](auto&& data) mutable -> Iter {
                    // make sure iterators embedded in function object that is
                    // attached to futures are invalidated
                    util::detail::clear_container(data);

                    difference_type find_res =
                        static_cast<difference_type>(tok.get_data());

                    std::advance(second, find_res);

                    return HPX_MOVE(second);
                };

                using partitioner_type =
                    util::partitioner<policy_type, Iter, void>;
                return partitioner_type::call_with_index(
                    HPX_FORWARD(decltype(policy), policy), second, count, 1,
                    HPX_MOVE(f1), HPX_MOVE(f2));
            }
        };

        template <typename RandIter>
        struct is_heap_until
          : public algorithm<is_heap_until<RandIter>, RandIter>
        {
            constexpr is_heap_until() noexcept
              : algorithm<is_heap_until, RandIter>("is_heap_until")
            {
            }

            template <typename ExPolicy, typename Iter, typename Sent,
                typename Comp, typename Proj>
            static constexpr Iter sequential(
                ExPolicy&&, Iter first, Sent last, Comp&& comp, Proj&& proj)
            {
                return sequential_is_heap_until(first, last,
                    HPX_FORWARD(Comp, comp), HPX_FORWARD(Proj, proj));
            }

            template <typename ExPolicy, typename Iter, typename Sent,
                typename Comp, typename Proj>
            static util::detail::algorithm_result_t<ExPolicy, Iter> parallel(
                ExPolicy&& policy, Iter first, Sent last, Comp&& comp,
                Proj&& proj)
            {
                return is_heap_until_helper()(HPX_FORWARD(ExPolicy, policy),
                    first, last, HPX_FORWARD(Comp, comp),
                    HPX_FORWARD(Proj, proj));
            }
        };
    }    // namespace detail
}    // namespace hpx::parallel

namespace hpx {

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::is_heap
    inline constexpr struct is_heap_t final
      : hpx::detail::tag_parallel_algorithm<is_heap_t>
    {
    private:
        // clang-format off
        template <typename ExPolicy, typename RandIter,
            typename Comp = hpx::parallel::detail::less,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy_v<ExPolicy> &&
                hpx::traits::is_iterator_v<RandIter> &&
                hpx::is_invocable_v<Comp,
                    typename std::iterator_traits<RandIter>::value_type,
                    typename std::iterator_traits<RandIter>::value_type
                >
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            bool>::type
        tag_fallback_invoke(is_heap_t, ExPolicy&& policy, RandIter first,
            RandIter last, Comp comp = Comp())
        {
            static_assert(hpx::traits::is_random_access_iterator_v<RandIter>,
                "Requires a random access iterator.");

            return hpx::parallel::detail::is_heap<RandIter>().call(
                HPX_FORWARD(ExPolicy, policy), first, last, HPX_MOVE(comp),
                hpx::identity_v);
        }

        // clang-format off
        template <typename RandIter,
            typename Comp = hpx::parallel::detail::less,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator_v<RandIter> &&
                hpx::is_invocable_v<Comp,
                    typename std::iterator_traits<RandIter>::value_type,
                    typename std::iterator_traits<RandIter>::value_type
                >
            )>
        // clang-format on
        friend bool tag_fallback_invoke(
            is_heap_t, RandIter first, RandIter last, Comp comp = Comp())
        {
            static_assert(hpx::traits::is_random_access_iterator_v<RandIter>,
                "Requires a random access iterator.");

            return hpx::parallel::detail::is_heap<RandIter>().call(
                hpx::execution::seq, first, last, HPX_MOVE(comp),
                hpx::identity_v);
        }
    } is_heap{};

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::is_heap_until
    inline constexpr struct is_heap_until_t final
      : hpx::detail::tag_parallel_algorithm<is_heap_until_t>
    {
    private:
        // clang-format off
        template <typename ExPolicy, typename RandIter,
            typename Comp = hpx::parallel::detail::less,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy_v<ExPolicy> &&
                hpx::traits::is_iterator_v<RandIter> &&
                hpx::is_invocable_v<Comp,
                    typename std::iterator_traits<RandIter>::value_type,
                    typename std::iterator_traits<RandIter>::value_type
                >
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            RandIter>::type
        tag_fallback_invoke(is_heap_until_t, ExPolicy&& policy, RandIter first,
            RandIter last, Comp comp = Comp())
        {
            static_assert(hpx::traits::is_random_access_iterator_v<RandIter>,
                "Requires a random access iterator.");

            return hpx::parallel::detail::is_heap_until<RandIter>().call(
                HPX_FORWARD(ExPolicy, policy), first, last, HPX_MOVE(comp),
                hpx::identity_v);
        }

        // clang-format off
        template <typename RandIter,
            typename Comp = hpx::parallel::detail::less,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator_v<RandIter> &&
                hpx::is_invocable_v<Comp,
                    typename std::iterator_traits<RandIter>::value_type,
                    typename std::iterator_traits<RandIter>::value_type
                >
            )>
        // clang-format on
        friend RandIter tag_fallback_invoke(
            is_heap_until_t, RandIter first, RandIter last, Comp comp = Comp())
        {
            static_assert(hpx::traits::is_random_access_iterator_v<RandIter>,
                "Requires a random access iterator.");

            return hpx::parallel::detail::is_heap_until<RandIter>().call(
                hpx::execution::seq, first, last, HPX_MOVE(comp),
                hpx::identity_v);
        }
    } is_heap_until{};
}    // namespace hpx

#endif    // DOXYGEN
