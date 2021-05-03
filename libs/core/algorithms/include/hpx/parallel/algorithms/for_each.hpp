//  Copyright (c) 2007-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/for_each.hpp

#pragma once

#if defined(DOXYGEN)
namespace hpx {
    /// Applies \a f to the result of dereferencing every iterator in the
    /// range [first, last).
    ///
    /// \note   Complexity: Applies \a f exactly \a last - \a first times.
    ///
    /// If \a f returns a result, the result is ignored.
    ///
    /// If the type of \a first satisfies the requirements of a mutable
    /// iterator, \a f may apply non-constant functions through the
    /// dereferenced iterator.
    ///
    /// \tparam InIter      The type of the source begin and end iterator used
    ///                     (deduced). This iterator type must meet the
    ///                     requirements of an input iterator.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). F must meet requirements of
    ///                     \a MoveConstructible.
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
    ///                     <ignored> pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&. The
    ///                     type \a Type must be such that an object of
    ///                     type \a FwdIter can be dereferenced and then
    ///                     implicitly converted to Type.
    ///
    /// \returns            \a f.
    template <typename InIter, typename F>
    F for_each(InIter first, InIter last, F&& f);

    /// Applies \a f to the result of dereferencing every iterator in the
    /// range [first, last).
    ///
    /// \note   Complexity: Applies \a f exactly \a last - \a first times.
    ///
    /// If \a f returns a result, the result is ignored.
    ///
    /// If the type of \a first satisfies the requirements of a mutable
    /// iterator, \a f may apply non-constant functions through the
    /// dereferenced iterator.
    ///
    /// Unlike its sequential form, the parallel overload of
    /// \a for_each does not return a copy of its \a Function parameter,
    /// since parallelization may not permit efficient state
    /// accumulation.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it applies user-provided function objects.
    /// \tparam FwdIte      The type of the source begin and end iterator used
    ///                     (deduced). This iterator type must meet the
    ///                     requirements of an forward iterator.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a for_each requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
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
    ///                     <ignored> pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&. The
    ///                     type \a Type must be such that an object of
    ///                     type \a FwdIter can be dereferenced and then
    ///                     implicitly converted to Type.
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
    /// \returns  The \a for_each algorithm returns a
    ///           \a hpx::future<void> if the execution policy is of
    ///           type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and returns void otherwise.
    template <typename ExPolicy, typename FwdIter, typename F>
    typename util::detail::algorithm_result<ExPolicy, void>::type for_each(
        ExPolicy&& policy, FwdIter first, FwdIter last, F&& f);

    /// Applies \a f to the result of dereferencing every iterator in the range
    /// [first, first + count), starting from first and proceeding to
    /// first + count - 1.
    ///
    /// \note   Complexity: Applies \a f exactly \a count times.
    ///
    /// If \a f returns a result, the result is ignored.
    ///
    /// If the type of \a first satisfies the requirements of a mutable
    /// iterator, \a f may apply non-constant functions through the
    /// dereferenced iterator.
    ///
    /// \tparam InIter      The type of the source begin and end iterator used
    ///                     (deduced). This iterator type must meet the
    ///                     requirements of an input iterator.
    /// \tparam Size        The type of the argument specifying the number of
    ///                     elements to apply \a f to.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). F must meet requirements of
    ///                     \a MoveConstructible.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param count        Refers to the number of elements starting at
    ///                     \a first the algorithm will be applied to.
    /// \param f            Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last).
    ///                     The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     <ignored> pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&. The
    ///                     type \a Type must be such that an object of
    ///                     type \a FwdIter can be dereferenced and then
    ///                     implicitly converted to Type.
    ///
    /// \returns            \a first + \a count for non-negative values of
    ///                     \a count and \a first for negative values.
    template <typename InIter, typename Size, typename F>
    InIter for_each_n(InIter first, Size count, F&& f);

    /// Applies \a f to the result of dereferencing every iterator in the range
    /// [first, first + count), starting from first and proceeding to
    /// first + count - 1.
    ///
    /// \note   Complexity: Applies \a f exactly \a count times.
    ///
    /// If \a f returns a result, the result is ignored.
    ///
    /// If the type of \a first satisfies the requirements of a mutable
    /// iterator, \a f may apply non-constant functions through the
    /// dereferenced iterator.
    ///
    /// Unlike its sequential form, the parallel overload of
    /// \a for_each does not return a copy of its \a Function parameter,
    /// since parallelization may not permit efficient state
    /// accumulation.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it applies user-provided function objects.
    /// \tparam FwdIter     The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Size        The type of the argument specifying the number of
    ///                     elements to apply \a f to.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a for_each requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param count        Refers to the number of elements starting at
    ///                     \a first the algorithm will be applied to.
    /// \param f            Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last).
    ///                     The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     <ignored> pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&. The
    ///                     type \a Type must be such that an object of
    ///                     type \a FwdIter can be dereferenced and then
    ///                     implicitly converted to Type.
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
    /// \returns  The \a for_each_n algorithm returns a
    ///           \a hpx::future<FwdIter> if the execution policy is of
    ///           type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and returns \a FwdIter
    ///           otherwise.
    ///           It returns \a first + \a count for non-negative values of
    ///           \a count and \a first for negative values.
    template <typename ExPolicy, typename FwdIter, typename Size, typename F>
    typename util::detail::algorithm_result<ExPolicy, FwdIter>::type for_each_n(
        ExPolicy&& policy, FwdIter first, Size count, F&& f);

}    // namespace hpx

#else

#include <hpx/config.hpp>
#include <hpx/algorithms/traits/is_value_proxy.hpp>
#include <hpx/algorithms/traits/projected.hpp>
#include <hpx/algorithms/traits/segmented_iterator_traits.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/functional/detail/invoke.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/modules/execution.hpp>
#include <hpx/modules/executors.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/detail/distance.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/detail/sender_util.hpp>
#include <hpx/parallel/util/foreach_partitioner.hpp>
#include <hpx/parallel/util/loop.hpp>
#include <hpx/parallel/util/projection_identity.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { inline namespace v1 {

    ///////////////////////////////////////////////////////////////////////////
    // for_each_n
    namespace detail {

        /// \cond NOINTERNAL
        template <typename F, typename Proj = util::projection_identity>
        struct invoke_projected
        {
            F& f_;
            Proj& proj_;

            template <typename T>
            HPX_HOST_DEVICE HPX_FORCEINLINE constexpr typename std::enable_if<
                !hpx::traits::is_value_proxy<T>::value>::type
            call(T&& t)
            {
                HPX_INVOKE(f_, HPX_INVOKE(proj_, std::forward<T>(t)));
            }

            template <typename T>
            HPX_HOST_DEVICE HPX_FORCEINLINE constexpr typename std::enable_if<
                hpx::traits::is_value_proxy<T>::value>::type
            call(T&& t)
            {
                auto tmp = HPX_INVOKE(proj_, std::forward<T>(t));
                HPX_INVOKE(f_, tmp);
            }

            template <typename Iter>
            HPX_HOST_DEVICE HPX_FORCEINLINE constexpr void operator()(Iter curr)
            {
                call(*curr);
            }
        };

        template <typename F>
        struct invoke_projected<F, util::projection_identity>
        {
            HPX_HOST_DEVICE constexpr invoke_projected(F& f) noexcept
              : f_(f)
            {
            }

            HPX_HOST_DEVICE constexpr invoke_projected(
                F& f, util::projection_identity) noexcept
              : f_(f)
            {
            }

            F& f_;

            template <typename T>
            HPX_HOST_DEVICE HPX_FORCEINLINE constexpr typename std::enable_if<
                !hpx::traits::is_value_proxy<T>::value>::type
            call(T&& t)
            {
                HPX_INVOKE(f_, std::forward<T>(t));
            }

            template <typename T>
            HPX_HOST_DEVICE HPX_FORCEINLINE constexpr typename std::enable_if<
                hpx::traits::is_value_proxy<T>::value>::type
            call(T&& t)
            {
                auto tmp = std::forward<T>(t);
                HPX_INVOKE(f_, tmp);
            }

            template <typename Iter>
            HPX_HOST_DEVICE HPX_FORCEINLINE constexpr void operator()(Iter curr)
            {
                call(*curr);
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename ExPolicy, typename F, typename Proj>
        struct for_each_iteration
        {
            using execution_policy_type = std::decay_t<ExPolicy>;
            using fun_type = std::decay_t<F>;
            using proj_type = Proj;

            fun_type f_;
            proj_type proj_;

            template <typename Iter>
            HPX_HOST_DEVICE HPX_FORCEINLINE constexpr void execute(
                Iter part_begin, std::size_t part_size)
            {
                util::loop_n<execution_policy_type>(part_begin, part_size,
                    invoke_projected<fun_type, proj_type>{f_, proj_});
            }

            template <typename F_, typename Proj_>
            HPX_HOST_DEVICE for_each_iteration(F_&& f, Proj_&& proj)
              : f_(std::forward<F_>(f))
              , proj_(std::forward<Proj_>(proj))
            {
            }

#if !defined(__NVCC__) && !defined(__CUDACC__)
            for_each_iteration(for_each_iteration const&) = default;
            for_each_iteration(for_each_iteration&&) = default;
#else
            HPX_HOST_DEVICE for_each_iteration(for_each_iteration const& rhs)
              : f_(rhs.f_)
              , proj_(rhs.proj_)
            {
            }

            HPX_HOST_DEVICE for_each_iteration(for_each_iteration&& rhs)
              : f_(std::move(rhs.f_))
              , proj_(std::move(rhs.proj_))
            {
            }
#endif

            for_each_iteration& operator=(for_each_iteration const&) = default;
            for_each_iteration& operator=(for_each_iteration&&) = default;

            template <typename Iter>
            HPX_HOST_DEVICE HPX_FORCEINLINE void operator()(Iter part_begin,
                std::size_t part_size, std::size_t /*part_index*/)
            {
                execute(part_begin, part_size);
            }
        };

        template <typename ExPolicy, typename F>
        struct for_each_iteration<ExPolicy, F, util::projection_identity>
        {
            using execution_policy_type = std::decay_t<ExPolicy>;
            using fun_type = std::decay_t<F>;

            fun_type f_;

            template <typename Iter>
            HPX_HOST_DEVICE HPX_FORCEINLINE constexpr void execute(
                Iter part_begin, std::size_t part_size, std::true_type)
            {
                util::loop_n<execution_policy_type>(
                    part_begin, part_size, invoke_projected<F>(f_));
            }

            template <typename Iter>
            HPX_HOST_DEVICE HPX_FORCEINLINE constexpr void execute(
                Iter part_begin, std::size_t part_size, std::false_type)
            {
                util::loop_n_ind<execution_policy_type>(
                    part_begin, part_size, f_);
            }

            template <typename F_, typename Proj_>
            HPX_HOST_DEVICE for_each_iteration(F_&& f, Proj_&&)
              : f_(std::forward<F_>(f))
            {
            }

#if !defined(__NVCC__) && !defined(__CUDACC__)
            for_each_iteration(for_each_iteration const&) = default;
            for_each_iteration(for_each_iteration&&) = default;
#else
            HPX_HOST_DEVICE for_each_iteration(for_each_iteration const& rhs)
              : f_(rhs.f_)
            {
            }

            HPX_HOST_DEVICE for_each_iteration(for_each_iteration&& rhs)
              : f_(std::move(rhs.f_))
            {
            }
#endif

            for_each_iteration& operator=(for_each_iteration const&) = default;
            for_each_iteration& operator=(for_each_iteration&&) = default;

            template <typename Iter>
            HPX_HOST_DEVICE HPX_FORCEINLINE void operator()(Iter part_begin,
                std::size_t part_size, std::size_t /*part_index*/)
            {
                using pred = hpx::traits::is_value_proxy<std::decay_t<
                    typename std::iterator_traits<Iter>::reference>>;

                hpx::util::annotate_function annotate(f_);
                execute(part_begin, part_size, pred());
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Iter>
        struct for_each_n : public detail::algorithm<for_each_n<Iter>, Iter>
        {
            for_each_n()
              : for_each_n::algorithm("for_each_n")
            {
            }

            template <typename ExPolicy, typename InIter, typename F,
                typename Proj>
            HPX_HOST_DEVICE static constexpr Iter sequential(
                ExPolicy&&, InIter first, std::size_t count, F&& f, Proj&& proj)
            {
                return util::loop_n<std::decay_t<ExPolicy>>(first, count,
                    invoke_projected<F, std::decay_t<Proj>>{f, proj});
            }

            template <typename ExPolicy, typename InIter, typename F>
            HPX_HOST_DEVICE static constexpr Iter sequential(ExPolicy&&,
                InIter first, std::size_t count, F&& f,
                util::projection_identity)
            {
                return util::loop_n_ind<std::decay_t<ExPolicy>>(
                    first, count, std::forward<F>(f));
            }

            template <typename ExPolicy, typename FwdIter, typename F,
                typename Proj = util::projection_identity>
            static constexpr
                typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
                parallel(ExPolicy&& policy, FwdIter first, std::size_t count,
                    F&& f, Proj&& proj /* = Proj()*/)
            {
                if (count != 0)
                {
                    auto f1 =
                        for_each_iteration<ExPolicy, F, std::decay_t<Proj>>(
                            std::forward<F>(f), std::forward<Proj>(proj));

                    return util::foreach_partitioner<ExPolicy>::call(
                        std::forward<ExPolicy>(policy), first, count,
                        std::move(f1), util::projection_identity());
                }

                return util::detail::algorithm_result<ExPolicy, FwdIter>::get(
                    std::move(first));
            }
        };
        /// \endcond
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    // for_each
    namespace detail {
        /// \cond NOINTERNAL
        template <typename Iter>
        struct for_each : public detail::algorithm<for_each<Iter>, Iter>
        {
            for_each()
              : for_each::algorithm("for_each")
            {
            }

            template <typename ExPolicy, typename InIterB, typename InIterE,
                typename F, typename Proj>
            static constexpr typename std::enable_if<
                hpx::traits::is_random_access_iterator<InIterB>::value,
                InIterB>::type
            sequential(
                ExPolicy&&, InIterB first, InIterE last, F&& f, Proj&& proj)
            {
                return util::loop_n<std::decay_t<ExPolicy>>(first,
                    static_cast<std::size_t>(detail::distance(first, last)),
                    invoke_projected<F, std::decay_t<Proj>>{f, proj});
            }

            template <typename ExPolicy, typename InIterB, typename InIterE,
                typename F, typename Proj>
            static constexpr typename std::enable_if<
                !hpx::traits::is_random_access_iterator<InIterB>::value,
                InIterB>::type
            sequential(ExPolicy&& policy, InIterB first, InIterE last, F&& f,
                Proj&& proj)
            {
                return util::loop(std::forward<ExPolicy>(policy), first, last,
                    invoke_projected<F, std::decay_t<Proj>>{f, proj});
            }

            template <typename ExPolicy, typename InIterB, typename InIterE,
                typename F>
            static constexpr typename std::enable_if<
                hpx::traits::is_random_access_iterator<InIterB>::value,
                InIterB>::type
            sequential(ExPolicy&&, InIterB first, InIterE last, F&& f,
                util::projection_identity)
            {
                return util::loop_n_ind<std::decay_t<ExPolicy>>(first,
                    static_cast<std::size_t>(detail::distance(first, last)),
                    std::forward<F>(f));
            }

            template <typename ExPolicy, typename InIterB, typename InIterE,
                typename F>
            static constexpr typename std::enable_if<
                !hpx::traits::is_random_access_iterator<InIterB>::value,
                InIterB>::type
            sequential(ExPolicy&& policy, InIterB first, InIterE last, F&& f,
                util::projection_identity)
            {
                return util::loop_ind(std::forward<ExPolicy>(policy), first,
                    last, std::forward<F>(f));
            }

            template <typename ExPolicy, typename FwdIterB, typename FwdIterE,
                typename F, typename Proj>
            static constexpr typename util::detail::algorithm_result<ExPolicy,
                FwdIterB>::type
            parallel(ExPolicy&& policy, FwdIterB first, FwdIterE last, F&& f,
                Proj&& proj)
            {
                if (first != last)
                {
                    auto f1 =
                        for_each_iteration<ExPolicy, F, std::decay_t<Proj>>(
                            std::forward<F>(f), std::forward<Proj>(proj));

                    return util::foreach_partitioner<ExPolicy>::call(
                        std::forward<ExPolicy>(policy), first,
                        detail::distance(first, last), std::move(f1),
                        util::projection_identity());
                }

                return util::detail::algorithm_result<ExPolicy, FwdIterB>::get(
                    std::move(first));
            }
        };
        /// \endcond
    }    // namespace detail

    // clang-format off
    template <typename ExPolicy, typename FwdIter, typename Size, typename F,
        typename Proj = util::projection_identity,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy<ExPolicy>::value&&
            hpx::traits::is_iterator<FwdIter>::value&&
            hpx::parallel::traits::is_projected<Proj, FwdIter>::value&&
            hpx::parallel::traits::is_indirect_callable<ExPolicy, F,
            hpx::parallel::traits::projected<Proj,
            FwdIter>>::value
        )>
    // clang-format on
    HPX_DEPRECATED_V(1, 6,
        "hpx::parallel::for_each_n is deprecated, use hpx::for_each_n "
        "instead")
        typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
        for_each_n(ExPolicy&& policy, FwdIter first, Size count, F&& f,
            Proj&& proj = Proj())
    {
        static_assert((hpx::traits::is_forward_iterator<FwdIter>::value),
            "Requires at least forward iterator.");

        // if count is representing a negative value or zero, we do nothing
        if (detail::is_negative(count) || count == 0)
        {
            using result = util::detail::algorithm_result<ExPolicy, FwdIter>;
            return result::get(std::move(first));
        }

#if defined(HPX_GCC_VERSION) && HPX_GCC_VERSION >= 100000
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
        return parallel::v1::detail::for_each_n<FwdIter>().call(
            std::forward<ExPolicy>(policy), first, std::size_t(count),
            std::forward<F>(f), std::forward<Proj>(proj));
#if defined(HPX_GCC_VERSION) && HPX_GCC_VERSION >= 100000
#pragma GCC diagnostic pop
#endif
    }

    // clang-format off
    template <typename ExPolicy, typename FwdIterB, typename FwdIterE,
        typename F, typename Proj = util::projection_identity,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy<ExPolicy>::value &&
            hpx::traits::is_iterator<FwdIterB>::value &&
            hpx::traits::is_sentinel_for<FwdIterE, FwdIterB>::value &&
            hpx::parallel::traits::is_projected<Proj, FwdIterB>::value
        )
#if (!defined(__NVCC__) && !defined(__CUDACC__)) || defined(__CUDA_ARCH__)
            ,
        HPX_CONCEPT_REQUIRES_(hpx::parallel::traits::is_indirect_callable<ExPolicy,
            F, traits::projected<Proj, FwdIterB>>::value)
#endif
        >
    // clang-format on
    HPX_DEPRECATED_V(1, 6,
        "hpx::parallel::for_each is deprecated, use hpx::for_each instead")
        typename util::detail::algorithm_result<ExPolicy, FwdIterB>::type
        for_each(ExPolicy&& policy, FwdIterB first, FwdIterE last, F&& f,
            Proj&& proj = Proj())
    {
        static_assert((hpx::traits::is_forward_iterator<FwdIterB>::value),
            "Requires at least forward iterator.");

        if (first == last)
        {
            using result =
                parallel::util::detail::algorithm_result<ExPolicy, FwdIterB>;
            return result::get(std::move(first));
        }

        return parallel::v1::detail::for_each<FwdIterB>().call(
            std::forward<ExPolicy>(policy), first, last, std::forward<F>(f),
            std::forward<Proj>(proj));
    }
}}}    // namespace hpx::parallel::v1

namespace hpx {

    // Note: The implementation of the non-segmented algorithms here relies on
    //       tag_fallback_dispatch. For this reason the tag_dispatch overloads for
    //       the segmented algorithms (and other specializations) take
    //       precedence over the implementations here. This has the advantage
    //       that the non-segmented algorithms do not need to be explicitly
    //       disabled for other, possibly external specializations.
    //
    HPX_INLINE_CONSTEXPR_VARIABLE struct for_each_t final
      : hpx::detail::tag_parallel_algorithm<for_each_t>
    {
    private:
        // clang-format off
        template <typename InIter,
            typename F,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator<InIter>::value
            )>
        // clang-format on
        friend F tag_fallback_dispatch(
            hpx::for_each_t, InIter first, InIter last, F&& f)
        {
            static_assert((hpx::traits::is_input_iterator<InIter>::value),
                "Requires at least input iterator.");

            if (first != last)
            {
                hpx::parallel::v1::detail::for_each<InIter>().call(
                    hpx::execution::seq, first, last, f,
                    hpx::parallel::util::projection_identity());
            }
            return std::forward<F>(f);
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter,
            typename F,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_iterator<FwdIter>::value
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<
            ExPolicy>::type
        tag_fallback_dispatch(hpx::for_each_t, ExPolicy&& policy, FwdIter first,
            FwdIter last, F&& f)
        {
            static_assert((hpx::traits::is_forward_iterator<FwdIter>::value),
                "Requires at least forward iterator.");

            if (first == last)
            {
                using result =
                    hpx::parallel::util::detail::algorithm_result<ExPolicy>;
                return result::get();
            }

            return hpx::parallel::util::detail::algorithm_result<ExPolicy>::get(
                hpx::parallel::v1::detail::for_each<FwdIter>().call(
                    std::forward<ExPolicy>(policy), first, last,
                    std::forward<F>(f),
                    hpx::parallel::util::projection_identity()));
        }
    } for_each{};

    ///////////////////////////////////////////////////////////////////////////
    HPX_INLINE_CONSTEXPR_VARIABLE struct for_each_n_t final
      : hpx::detail::tag_parallel_algorithm<for_each_n_t>
    {
    private:
        // clang-format off
        template <typename InIter, typename Size, typename F,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_input_iterator<InIter>::value
            )>
        // clang-format on
        friend InIter tag_fallback_dispatch(
            hpx::for_each_n_t, InIter first, Size count, F&& f)
        {
            static_assert((hpx::traits::is_input_iterator<InIter>::value),
                "Requires at least input iterator.");

            // if count is representing a negative value, we do nothing
            if (parallel::v1::detail::is_negative(count))
            {
                return first;
            }

            return hpx::parallel::v1::detail::for_each_n<InIter>().call(
                hpx::execution::seq, first, std::size_t(count),
                std::forward<F>(f), parallel::util::projection_identity());
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter, typename Size, typename F,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_forward_iterator<FwdIter>::value
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            FwdIter>::type
        tag_fallback_dispatch(hpx::for_each_n_t, ExPolicy&& policy,
            FwdIter first, Size count, F&& f)
        {
            static_assert((hpx::traits::is_forward_iterator<FwdIter>::value),
                "Requires at least forward iterator.");

            // if count is representing a negative value, we do nothing
            if (parallel::v1::detail::is_negative(count))
            {
                using result =
                    parallel::util::detail::algorithm_result<ExPolicy, FwdIter>;
                return result::get(std::move(first));
            }

            return hpx::parallel::v1::detail::for_each_n<FwdIter>().call(
                std::forward<ExPolicy>(policy), first, std::size_t(count),
                std::forward<F>(f), parallel::util::projection_identity());
        }
    } for_each_n{};
}    // namespace hpx

#if defined(HPX_HAVE_THREAD_DESCRIPTION)
namespace hpx { namespace traits {
    template <typename ExPolicy, typename F, typename Proj>
    struct get_function_address<
        parallel::v1::detail::for_each_iteration<ExPolicy, F, Proj>>
    {
        static constexpr std::size_t call(
            parallel::v1::detail::for_each_iteration<ExPolicy, F, Proj> const&
                f) noexcept
        {
            return get_function_address<std::decay_t<F>>::call(f.f_);
        }
    };

    template <typename ExPolicy, typename F, typename Proj>
    struct get_function_annotation<
        parallel::v1::detail::for_each_iteration<ExPolicy, F, Proj>>
    {
        static constexpr char const* call(
            parallel::v1::detail::for_each_iteration<ExPolicy, F, Proj> const&
                f) noexcept
        {
            return get_function_annotation<std::decay_t<F>>::call(f.f_);
        }
    };

#if HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX)
    template <typename ExPolicy, typename F, typename Proj>
    struct get_function_annotation_itt<
        parallel::v1::detail::for_each_iteration<ExPolicy, F, Proj>>
    {
        static util::itt::string_handle call(
            parallel::v1::detail::for_each_iteration<ExPolicy, F, Proj> const&
                f) noexcept
        {
            return get_function_annotation_itt<std::decay_t<F>>::call(f.f_);
        }
    };
#endif
}}    // namespace hpx::traits
#endif
#endif
