//  Copyright (c) 2007-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/all_any_none.hpp

#pragma once

#if defined(DOXYGEN)
namespace hpx {
    // clang-format off

    ///  Checks if unary predicate \a f returns true for no elements in the
    ///  range [first, last).
    ///
    /// \note   Complexity: At most \a last - \a first applications of the
    ///         predicate \a f
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it applies user-provided function objects.
    /// \tparam FwdIter     The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a none_of requires \a F to meet the
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
    ///                     sequence specified by [first, last).
    ///                     The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     bool pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed
    ///                     to it. The type \a Type must be such that an object
    ///                     of type \a FwdIter can be dereferenced and then
    ///                     implicitly converted to Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
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
    /// \returns  The \a none_of algorithm returns a \a hpx::future<bool> if
    ///           the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and returns \a bool
    ///           otherwise.
    ///           The \a none_of algorithm returns true if the unary predicate
    ///           \a f returns true for no elements in the range, false
    ///           otherwise. It returns true if the range is empty.
    ///
    template <typename ExPolicy, typename FwdIter, typename F,
        typename Proj = util::projection_identity>
    typename util::detail::algorithm_result<ExPolicy, bool>::type
    none_of(ExPolicy&& policy, FwdIter first, FwdIter last, F&& f,
        Proj&& proj = Proj());

    ///  Checks if unary predicate \a f returns true for at least one element
    ///  in the range [first, last).
    ///
    /// \note   Complexity: At most \a last - \a first applications of the
    ///         predicate \a f
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it applies user-provided function objects.
    /// \tparam FwdIter     The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a any_of requires \a F to meet the
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
    ///                     sequence specified by [first, last).
    ///                     The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     bool pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed
    ///                     to it. The type \a Type must be such that an object
    ///                     of type \a FwdIter can be dereferenced and then
    ///                     implicitly converted to Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
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
    /// \returns  The \a any_of algorithm returns a \a hpx::future<bool> if
    ///           the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a bool otherwise.
    ///           The \a any_of algorithm returns true if the unary predicate
    ///           \a f returns true for at least one element in the range,
    ///           false otherwise. It returns false if the range is empty.
    ///
    template <typename ExPolicy, typename FwdIter, typename F,
        typename Proj = util::projection_identity>
    typename util::detail::algorithm_result<ExPolicy, bool>::type
    any_of(ExPolicy&& policy, FwdIter first, FwdIter last, F&& f,
        Proj&& proj = Proj());

    /// Checks if unary predicate \a f returns true for all elements in the
    /// range [first, last).
    ///
    /// \note   Complexity: At most \a last - \a first applications of the
    ///         predicate \a f
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it applies user-provided function objects.
    /// \tparam FwdIter     The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a all_of requires \a F to meet the
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
    ///                     sequence specified by [first, last).
    ///                     The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     bool pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed
    ///                     to it. The type \a Type must be such that an object
    ///                     of type \a FwdIter can be dereferenced and then
    ///                     implicitly converted to Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
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
    /// \returns  The \a all_of algorithm returns a \a hpx::future<bool> if
    ///           the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a bool otherwise.
    ///           The \a all_of algorithm returns true if the unary predicate
    ///           \a f returns true for all elements in the range, false
    ///           otherwise. It returns true if the range is empty.
    ///
    template <typename ExPolicy, typename FwdIter, typename F,
        typename Proj = util::projection_identity>
    typename util::detail::algorithm_result<ExPolicy, bool>::type
    all_of(ExPolicy&& policy, FwdIter first, FwdIter last, F&& f,
        Proj&& proj = Proj());

    // clang-format on
}    // namespace hpx
#else    // DOXYGEN

#include <hpx/local/config.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/functional/invoke.hpp>
#include <hpx/parallel/util/detail/sender_util.hpp>
#include <hpx/iterator_support/range.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/type_support/void_guard.hpp>

#include <hpx/algorithms/traits/projected.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/detail/distance.hpp>
#include <hpx/parallel/algorithms/detail/find.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/invoke_projected.hpp>
#include <hpx/parallel/util/loop.hpp>
#include <hpx/parallel/util/partitioner.hpp>
#include <hpx/type_support/unused.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace parallel { inline namespace v1 {

    ///////////////////////////////////////////////////////////////////////////
    // none_of
    namespace detail {
        /// \cond NOINTERNAL
        struct none_of : public detail::algorithm<none_of, bool>
        {
            none_of()
              : none_of::algorithm("none_of")
            {
            }

            template <typename ExPolicy, typename Iter, typename Sent,
                typename F, typename Proj>
            static bool sequential(
                ExPolicy, Iter first, Sent last, F&& f, Proj&& proj)
            {
                return detail::sequential_find_if(first, last,
                           util::invoke_projected<F, Proj>(std::forward<F>(f),
                               std::forward<Proj>(proj))) == last;
            }

            template <typename ExPolicy, typename FwdIter, typename Sent,
                typename F, typename Proj>
            static typename util::detail::algorithm_result<ExPolicy, bool>::type
            parallel(ExPolicy&& policy, FwdIter first, Sent last, F&& op,
                Proj&& proj)
            {
                if (first == last)
                {
                    return util::detail::algorithm_result<ExPolicy, bool>::get(
                        true);
                }

                util::cancellation_token<> tok;
                auto f1 = [op = std::forward<F>(op), tok,
                              proj = std::forward<Proj>(proj)](
                              FwdIter part_begin,
                              std::size_t part_count) mutable -> bool {
                    util::loop_n<std::decay_t<ExPolicy>>(part_begin, part_count,
                        tok, [&op, &tok, &proj](FwdIter const& curr) {
                            if (hpx::util::invoke(
                                    op, hpx::util::invoke(proj, *curr)))
                            {
                                tok.cancel();
                            }
                        });

                    return !tok.was_cancelled();
                };

                return util::partitioner<ExPolicy, bool>::call(
                    std::forward<ExPolicy>(policy), first,
                    detail::distance(first, last), std::move(f1),
                    [](std::vector<hpx::future<bool>>&& results) {
                        return detail::sequential_find_if_not(
                                   hpx::util::begin(results),
                                   hpx::util::end(results),
                                   [](hpx::future<bool>& val) {
                                       return val.get();
                                   }) == hpx::util::end(results);
                    });
            }
        };
        /// \endcond
    }    // namespace detail

    // clang-format off
    template <typename ExPolicy, typename FwdIter, typename F,
        typename Proj = util::projection_identity,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy<ExPolicy>::value &&
            hpx::traits::is_iterator<FwdIter>::value &&
            traits::is_projected<Proj, FwdIter>::value &&
            traits::is_indirect_callable<ExPolicy, F,
                traits::projected<Proj, FwdIter>
            >::value
        )>
    // clang-format on
    HPX_LOCAL_DEPRECATED_V(
        1, 6, "hpx::parallel::none_of is deprecated, use hpx::none_of instead")
        typename util::detail::algorithm_result<ExPolicy, bool>::type
        none_of(ExPolicy&& policy, FwdIter first, FwdIter last, F&& f,
            Proj&& proj = Proj())
    {
        static_assert(hpx::traits::is_forward_iterator<FwdIter>::value,
            "Required at least forward iterator.");

#if defined(HPX_GCC_VERSION) && HPX_GCC_VERSION >= 100000
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
        return hpx::parallel::v1::detail::none_of().call(
            std::forward<ExPolicy>(policy), first, last, std::forward<F>(f),
            std::forward<Proj>(proj));
#if defined(HPX_GCC_VERSION) && HPX_GCC_VERSION >= 100000
#pragma GCC diagnostic pop
#endif
    }

    ///////////////////////////////////////////////////////////////////////////
    // any_of
    namespace detail {
        /// \cond NOINTERNAL
        struct any_of : public detail::algorithm<any_of, bool>
        {
            any_of()
              : any_of::algorithm("any_of")
            {
            }

            template <typename ExPolicy, typename Iter, typename Sent,
                typename F, typename Proj>
            static bool sequential(
                ExPolicy, Iter first, Sent last, F&& f, Proj&& proj)
            {
                return detail::sequential_find_if(first, last,
                           util::invoke_projected<F, Proj>(std::forward<F>(f),
                               std::forward<Proj>(proj))) != last;
            }

            template <typename ExPolicy, typename FwdIter, typename Sent,
                typename F, typename Proj>
            static typename util::detail::algorithm_result<ExPolicy, bool>::type
            parallel(ExPolicy&& policy, FwdIter first, Sent last, F&& op,
                Proj&& proj)
            {
                if (first == last)
                {
                    return util::detail::algorithm_result<ExPolicy, bool>::get(
                        false);
                }

                util::cancellation_token<> tok;
                auto f1 = [op = std::forward<F>(op), tok,
                              proj = std::forward<Proj>(proj)](
                              FwdIter part_begin,
                              std::size_t part_count) mutable -> bool {
                    util::loop_n<std::decay_t<ExPolicy>>(part_begin, part_count,
                        tok, [&op, &tok, &proj](FwdIter const& curr) {
                            if (hpx::util::invoke(
                                    op, hpx::util::invoke(proj, *curr)))
                            {
                                tok.cancel();
                            }
                        });

                    return tok.was_cancelled();
                };

                return util::partitioner<ExPolicy, bool>::call(
                    std::forward<ExPolicy>(policy), first,
                    detail::distance(first, last), std::move(f1),
                    [](std::vector<hpx::future<bool>>&& results) {
                        return detail::sequential_find_if(
                                   hpx::util::begin(results),
                                   hpx::util::end(results),
                                   [](hpx::future<bool>& val) {
                                       return val.get();
                                   }) != hpx::util::end(results);
                    });
            }
        };
        /// \endcond
    }    // namespace detail

    // clang-format off
    template <typename ExPolicy, typename FwdIter, typename F,
        typename Proj = util::projection_identity,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy<ExPolicy>::value &&
            hpx::traits::is_iterator<FwdIter>::value &&
            traits::is_projected<Proj, FwdIter>::value &&
            traits::is_indirect_callable<ExPolicy, F,
                traits::projected<Proj, FwdIter>
            >::value
        )>
    // clang-format on
    HPX_LOCAL_DEPRECATED_V(
        1, 6, "hpx::parallel::any_of is deprecated, use hpx::any_of instead")
        typename util::detail::algorithm_result<ExPolicy, bool>::type
        any_of(ExPolicy&& policy, FwdIter first, FwdIter last, F&& f,
            Proj&& proj = Proj())
    {
        static_assert(hpx::traits::is_forward_iterator<FwdIter>::value,
            "Required at least forward iterator.");

#if defined(HPX_GCC_VERSION) && HPX_GCC_VERSION >= 100000
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
        return hpx::parallel::v1::detail::any_of().call(
            std::forward<ExPolicy>(policy), first, last, std::forward<F>(f),
            std::forward<Proj>(proj));
#if defined(HPX_GCC_VERSION) && HPX_GCC_VERSION >= 100000
#pragma GCC diagnostic pop
#endif
    }

    ///////////////////////////////////////////////////////////////////////////
    // all_of
    namespace detail {
        /// \cond NOINTERNAL
        struct all_of : public detail::algorithm<all_of, bool>
        {
            all_of()
              : all_of::algorithm("all_of")
            {
            }

            template <typename ExPolicy, typename Iter, typename Sent,
                typename F, typename Proj>
            static bool sequential(
                ExPolicy, Iter first, Sent last, F&& f, Proj&& proj)
            {
                return detail::sequential_find_if_not(first, last,
                           std::forward<F>(f),
                           std::forward<Proj>(proj)) == last;
            }

            template <typename ExPolicy, typename FwdIter, typename Sent,
                typename F, typename Proj>
            static typename util::detail::algorithm_result<ExPolicy, bool>::type
            parallel(ExPolicy&& policy, FwdIter first, Sent last, F&& op,
                Proj&& proj)
            {
                if (first == last)
                {
                    return util::detail::algorithm_result<ExPolicy, bool>::get(
                        true);
                }

                util::cancellation_token<> tok;
                auto f1 = [op = std::forward<F>(op), tok,
                              proj = std::forward<Proj>(proj)](
                              FwdIter part_begin,
                              std::size_t part_count) mutable -> bool {
                    util::loop_n<std::decay_t<ExPolicy>>(part_begin, part_count,
                        tok, [&op, &tok, &proj](FwdIter const& curr) {
                            if (!hpx::util::invoke(
                                    op, hpx::util::invoke(proj, *curr)))
                            {
                                tok.cancel();
                            }
                        });

                    return !tok.was_cancelled();
                };

                return util::partitioner<ExPolicy, bool>::call(
                    std::forward<ExPolicy>(policy), first,
                    detail::distance(first, last), std::move(f1),
                    [](std::vector<hpx::future<bool>>&& results) {
                        return detail::sequential_find_if_not(
                                   hpx::util::begin(results),
                                   hpx::util::end(results),
                                   [](hpx::future<bool>& val) {
                                       return val.get();
                                   }) == hpx::util::end(results);
                    });
            }
        };
        /// \endcond
    }    // namespace detail

    // clang-format off
    template <typename ExPolicy, typename FwdIter, typename F,
        typename Proj = util::projection_identity,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy<ExPolicy>::value&&
            hpx::traits::is_iterator<FwdIter>::value&&
            traits::is_projected<Proj, FwdIter>::value&&
            traits::is_indirect_callable<ExPolicy, F,
                traits::projected<Proj, FwdIter>
            >::value
        )>
    // clang-format on
    HPX_LOCAL_DEPRECATED_V(
        1, 6, "hpx::parallel::all_of is deprecated, use hpx::all_of instead")
        typename util::detail::algorithm_result<ExPolicy, bool>::type
        all_of(ExPolicy&& policy, FwdIter first, FwdIter last, F&& f,
            Proj&& proj = Proj())
    {
        static_assert(hpx::traits::is_forward_iterator<FwdIter>::value,
            "Required at least forward iterator.");

#if defined(HPX_GCC_VERSION) && HPX_GCC_VERSION >= 100000
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
        return hpx::parallel::v1::detail::all_of().call(
            std::forward<ExPolicy>(policy), first, last, std::forward<F>(f),
            std::forward<Proj>(proj));
#if defined(HPX_GCC_VERSION) && HPX_GCC_VERSION >= 100000
#pragma GCC diagnostic pop
#endif
    }
}}}    // namespace hpx::parallel::v1

namespace hpx {

    ///////////////////////////////////////////////////////////////////////////
    // DPO for hpx::none_of
    HPX_INLINE_CONSTEXPR_VARIABLE struct none_of_t final
      : hpx::detail::tag_parallel_algorithm<none_of_t>
    {
    private:
        // clang-format off
        template <typename ExPolicy, typename FwdIter, typename F,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_iterator<FwdIter>::value
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            bool>::type
        tag_fallback_dispatch(
            none_of_t, ExPolicy&& policy, FwdIter first, FwdIter last, F&& f)
        {
            static_assert(hpx::traits::is_forward_iterator<FwdIter>::value,
                "Required at least forward iterator.");

            return hpx::parallel::v1::detail::none_of().call(
                std::forward<ExPolicy>(policy), first, last, std::forward<F>(f),
                hpx::parallel::util::projection_identity{});
        }

        // clang-format off
        template <typename InIter, typename F,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator<InIter>::value
            )>
        // clang-format on
        friend bool tag_fallback_dispatch(
            none_of_t, InIter first, InIter last, F&& f)
        {
            static_assert(hpx::traits::is_input_iterator<InIter>::value,
                "Required at least input iterator.");

            return hpx::parallel::v1::detail::none_of().call(
                hpx::execution::seq, first, last, std::forward<F>(f),
                hpx::parallel::util::projection_identity{});
        }
    } none_of{};

    ///////////////////////////////////////////////////////////////////////////
    // DPO for hpx::any_of
    HPX_INLINE_CONSTEXPR_VARIABLE struct any_of_t final
      : hpx::detail::tag_parallel_algorithm<any_of_t>
    {
    private:
        // clang-format off
        template <typename ExPolicy, typename FwdIter, typename F,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_iterator<FwdIter>::value
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            bool>::type
        tag_fallback_dispatch(
            any_of_t, ExPolicy&& policy, FwdIter first, FwdIter last, F&& f)
        {
            static_assert(hpx::traits::is_forward_iterator<FwdIter>::value,
                "Required at least forward iterator.");

            return hpx::parallel::v1::detail::any_of().call(
                std::forward<ExPolicy>(policy), first, last, std::forward<F>(f),
                hpx::parallel::util::projection_identity{});
        }

        // clang-format off
        template <typename InIter, typename F,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator<InIter>::value
            )>
        // clang-format on
        friend bool tag_fallback_dispatch(
            any_of_t, InIter first, InIter last, F&& f)
        {
            static_assert(hpx::traits::is_input_iterator<InIter>::value,
                "Required at least input iterator.");

            return hpx::parallel::v1::detail::any_of().call(hpx::execution::seq,
                first, last, std::forward<F>(f),
                hpx::parallel::util::projection_identity{});
        }
    } any_of{};

    ///////////////////////////////////////////////////////////////////////////
    // DPO for hpx::all_of
    HPX_INLINE_CONSTEXPR_VARIABLE struct all_of_t final
      : hpx::detail::tag_parallel_algorithm<all_of_t>
    {
    private:
        // clang-format off
        template <typename ExPolicy, typename FwdIter, typename F,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_iterator<FwdIter>::value
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            bool>::type
        tag_fallback_dispatch(
            all_of_t, ExPolicy&& policy, FwdIter first, FwdIter last, F&& f)
        {
            static_assert(hpx::traits::is_forward_iterator<FwdIter>::value,
                "Required at least forward iterator.");

            return hpx::parallel::v1::detail::all_of().call(
                std::forward<ExPolicy>(policy), first, last, std::forward<F>(f),
                hpx::parallel::util::projection_identity());
        }

        // clang-format off
        template <typename InIter, typename F,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator<InIter>::value
            )>
        // clang-format on
        friend bool tag_fallback_dispatch(
            all_of_t, InIter first, InIter last, F&& f)
        {
            static_assert(hpx::traits::is_input_iterator<InIter>::value,
                "Required at least input iterator.");

            return hpx::parallel::v1::detail::all_of().call(hpx::execution::seq,
                first, last, std::forward<F>(f),
                hpx::parallel::util::projection_identity{});
        }
    } all_of{};

}    // namespace hpx

#endif    // DOXYGEN
