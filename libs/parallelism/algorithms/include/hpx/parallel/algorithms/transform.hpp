//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c) 2021 Giannis Gonidelis
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/transform.hpp

#pragma once

#if defined(DOXYGEN)
namespace hpx {

    /// Applies the given function \a f to the range [first, last) and stores
    /// the result in another range, beginning at dest.
    ///
    /// \note   Complexity: Exactly \a last - \a first applications of \a f
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the invocations of \a f.
    /// \tparam FwdIter1    The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam FwdIter2    The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a transform requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
    /// \param f            Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last).This is an
    ///                     unary predicate. The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     Ret fun(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&.
    ///                     The type \a Type must be such that an object of
    ///                     type \a FwdIter1 can be dereferenced and then
    ///                     implicitly converted to \a Type. The type \a Ret
    ///                     must be such that an object of type \a FwdIter2 can
    ///                     be dereferenced and assigned a value of type
    ///                     \a Ret.
    ///
    /// The invocations of \a f in the parallel \a transform algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The invocations of \a f in the parallel \a transform algorithm invoked
    /// with an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a transform algorithm returns a
    /// \a hpx::future<in_out_result<FwdIter1, FwdIter2> >
    ///           if the execution policy is of type \a parallel_task_policy
    ///           and returns
    /// \a in_out_result<FwdIter1, FwdIter2> otherwise.
    ///           The \a transform algorithm returns a tuple holding an iterator
    ///           referring to the first element after the input sequence and
    ///           the output iterator to the
    ///           element in the destination range, one past the last element
    ///           copied.
    ///
    template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
        typename F>
    typename util::detail::algorithm_result<ExPolicy,
        util::in_out_result<FwdIter1, FwdIter2>>::type
    transform(
        ExPolicy&& policy, FwdIter1 first, FwdIter1 last, FwdIter2 dest, F&& f);

    /// Applies the given function \a f to pairs of elements from two ranges:
    /// one defined by [first1, last1) and the other beginning at first2, and
    /// stores the result in another range, beginning at dest.
    ///
    /// \note   Complexity: Exactly \a last - \a first applications of \a f
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the invocations of \a f.
    /// \tparam FwdIter1    The type of the source iterators for the first
    ///                     range used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam FwdIter2    The type of the source iterators for the second
    ///                     range used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam FwdIter3    The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a transform requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first1       Refers to the beginning of the first sequence of
    ///                     elements the algorithm will be applied to.
    /// \param last1        Refers to the end of the first sequence of elements
    ///                     the algorithm will be applied to.
    /// \param first2       Refers to the beginning of the second sequence of
    ///                     elements the algorithm will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
    /// \param f            Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last).This is a
    ///                     binary predicate. The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     Ret fun(const Type1 &a, const Type2 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const&.
    ///                     The types \a Type1 and \a Type2 must be such that
    ///                     objects of types FwdIter1 and FwdIter2 can be
    ///                     dereferenced and then implicitly converted to
    ///                     \a Type1 and \a Type2 respectively. The type \a Ret
    ///                     must be such that an object of type \a FwdIter3 can
    ///                     be dereferenced and assigned a value of type
    ///                     \a Ret.
    ///
    /// The invocations of \a f in the parallel \a transform algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The invocations of \a f in the parallel \a transform algorithm invoked
    /// with an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a transform algorithm returns a
    /// \a hpx::future<in_in_out_result<FwdIter1, FwdIter2, FwdIter3> >
    ///           if the execution policy is of type \a parallel_task_policy
    ///           and returns
    /// \a in_in_out_result<FwdIter1, FwdIter2, FwdIter3>
    ///           otherwise.
    ///           The \a transform algorithm returns a tuple holding an iterator
    ///           referring to the first element after the first input sequence,
    ///           an iterator referring to the first element after the second
    ///           input sequence, and the output iterator referring to the
    ///           element in the destination range, one past the last element
    ///           copied.
    ///
    template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
        typename FwdIter3, typename F>
    typename util::detail::algorithm_result<ExPolicy,
        util::in_in_out_result<FwdIter1, FwdIter2, FwdIter3>>::type
    transform(ExPolicy&& policy, FwdIter1 first1, FwdIter1 last1,
        FwdIter2 first2, FwdIter3 dest, F&& f);

}    // namespace hpx

#else    // DOXYGEN

#include <hpx/config.hpp>
#include <hpx/concepts/concepts.hpp>
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
#include <hpx/functional/traits/get_function_address.hpp>
#include <hpx/functional/traits/get_function_annotation.hpp>
#endif
#include <hpx/algorithms/traits/segmented_iterator_traits.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/functional/invoke.hpp>
#include <hpx/functional/traits/is_invocable.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/parallel/util/result_types.hpp>
#include <hpx/threading_base/annotated_function.hpp>

#include <hpx/algorithms/traits/projected.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/detail/distance.hpp>
#include <hpx/parallel/tagspec.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/foreach_partitioner.hpp>
#include <hpx/parallel/util/projection_identity.hpp>
#include <hpx/parallel/util/transform_loop.hpp>
#include <hpx/parallel/util/zip_iterator.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { inline namespace v1 {
    ///////////////////////////////////////////////////////////////////////////
    // transform
    namespace detail {
        /// \cond NOINTERNAL
        template <typename F, typename Proj>
        struct transform_projected
        {
            typename std::decay<F>::type& f_;
            typename std::decay<Proj>::type& proj_;

            template <typename Iter>
            HPX_HOST_DEVICE HPX_FORCEINLINE auto operator()(Iter curr)
                -> decltype(
                    hpx::util::invoke(f_, hpx::util::invoke(proj_, *curr)))
            {
                return hpx::util::invoke(f_, hpx::util::invoke(proj_, *curr));
            }
        };

        template <typename ExPolicy, typename F, typename Proj>
        struct transform_iteration
        {
            typedef typename std::decay<ExPolicy>::type execution_policy_type;
            typedef typename std::decay<F>::type fun_type;
            typedef typename std::decay<Proj>::type proj_type;

            fun_type f_;
            proj_type proj_;

            template <typename F_, typename Proj_>
            HPX_HOST_DEVICE transform_iteration(F_&& f, Proj_&& proj)
              : f_(std::forward<F_>(f))
              , proj_(std::forward<Proj_>(proj))
            {
            }

#if !defined(__NVCC__) && !defined(__CUDACC__)
            transform_iteration(transform_iteration const&) = default;
            transform_iteration(transform_iteration&&) = default;
#else
            HPX_HOST_DEVICE transform_iteration(transform_iteration const& rhs)
              : f_(rhs.f_)
              , proj_(rhs.proj_)
            {
            }

            HPX_HOST_DEVICE transform_iteration(transform_iteration&& rhs)
              : f_(std::move(rhs.f_))
              , proj_(std::move(rhs.proj_))
            {
            }
#endif

            transform_iteration& operator=(
                transform_iteration const&) = default;
            transform_iteration& operator=(transform_iteration&&) = default;

            template <typename Iter>
            HPX_HOST_DEVICE HPX_FORCEINLINE
                std::pair<typename hpx::tuple_element<0,
                              typename Iter::iterator_tuple_type>::type,
                    typename hpx::tuple_element<1,
                        typename Iter::iterator_tuple_type>::type>
                execute(Iter part_begin, std::size_t part_size)
            {
                auto iters = part_begin.get_iterator_tuple();
                return util::transform_loop_n<execution_policy_type>(
                    hpx::get<0>(iters), part_size, hpx::get<1>(iters),
                    transform_projected<F, Proj>{f_, proj_});
            }

            template <typename Iter>
            HPX_HOST_DEVICE HPX_FORCEINLINE
                std::pair<typename hpx::tuple_element<0,
                              typename Iter::iterator_tuple_type>::type,
                    typename hpx::tuple_element<1,
                        typename Iter::iterator_tuple_type>::type>
                operator()(Iter part_begin, std::size_t part_size,
                    std::size_t /*part_index*/)
            {
                hpx::util::annotate_function annotate(f_);
                return execute(part_begin, part_size);
            }
        };

        template <typename IterPair>
        struct transform
          : public detail::algorithm<transform<IterPair>, IterPair>
        {
            transform()
              : transform::algorithm("transform")
            {
            }

            template <typename ExPolicy, typename InIterB, typename InIterE,
                typename OutIter, typename F, typename Proj>
            HPX_HOST_DEVICE static util::in_out_result<InIterB, OutIter>
            sequential(ExPolicy&& policy, InIterB first, InIterE last,
                OutIter dest, F&& f, Proj&& proj)
            {
                return util::transform_loop(std::forward<ExPolicy>(policy),
                    first, last, dest, transform_projected<F, Proj>{f, proj});
            }

            template <typename ExPolicy, typename FwdIter1B, typename FwdIter1E,
                typename FwdIter2, typename F, typename Proj>
            static typename util::detail::algorithm_result<ExPolicy,
                util::in_out_result<FwdIter1B, FwdIter2>>::type
            parallel(ExPolicy&& policy, FwdIter1B first, FwdIter1E last,
                FwdIter2 dest, F&& f, Proj&& proj)
            {
                if (first != last)
                {
                    auto f1 = transform_iteration<ExPolicy, F, Proj>(
                        std::forward<F>(f), std::forward<Proj>(proj));

                    return util::detail::get_in_out_result(
                        util::foreach_partitioner<ExPolicy>::call(
                            std::forward<ExPolicy>(policy),
                            hpx::util::make_zip_iterator(first, dest),
                            detail::distance(first, last), std::move(f1),
                            util::projection_identity()));
                }

                using result_type = util::in_out_result<FwdIter1B, FwdIter2>;

                return util::detail::algorithm_result<ExPolicy,
                    result_type>::get(result_type{
                    std::move(first), std::move(dest)});
            }
        };

        /// non_segmented version
        template <typename ExPolicy, typename FwdIter1B, typename FwdIter1E,
            typename FwdIter2, typename F, typename Proj>
        typename util::detail::algorithm_result<ExPolicy,
            util::in_out_result<FwdIter1B, FwdIter2>>::type
        transform_(ExPolicy&& policy, FwdIter1B first, FwdIter1E last,
            FwdIter2 dest, F&& f, Proj&& proj, std::false_type)
        {
            static_assert((hpx::traits::is_forward_iterator<FwdIter1B>::value),
                "Requires at least forward iterator.");
            static_assert((hpx::traits::is_forward_iterator<FwdIter2>::value),
                "Requires at least forward iterator.");

            typedef hpx::is_sequenced_execution_policy<ExPolicy> is_seq;

            return detail::transform<util::in_out_result<FwdIter1B, FwdIter2>>()
                .call(std::forward<ExPolicy>(policy), is_seq(), first, last,
                    dest, std::forward<F>(f), std::forward<Proj>(proj));
        }

        /// forward declare the segmented version
        template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
            typename F, typename Proj>
        typename util::detail::algorithm_result<ExPolicy,
            util::in_out_result<FwdIter1, FwdIter2>>::type
        transform_(ExPolicy&& policy, FwdIter1 first, FwdIter1 last,
            FwdIter2 dest, F&& f, Proj&& proj, std::true_type);
        /// \endcond
    }    // namespace detail

    // clang-format off
    template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
        typename F, typename Proj = util::projection_identity,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy<ExPolicy>::value &&
            hpx::traits::is_iterator<FwdIter1>::value &&
            hpx::traits::is_iterator<FwdIter2>::value &&
            traits::is_projected<Proj, FwdIter1>::value &&
            traits::is_indirect_callable<ExPolicy, F,
                traits::projected<Proj, FwdIter1>>::value
        )>
    // clang-format on
    HPX_DEPRECATED_V(1, 6,
        "hpx::parallel::transform is deprecated, use hpx::transform instead")
        typename util::detail::algorithm_result<ExPolicy,
            util::in_out_result<FwdIter1, FwdIter2>>::type
        transform(ExPolicy&& policy, FwdIter1 first, FwdIter1 last,
            FwdIter2 dest, F&& f, Proj&& proj = Proj())
    {
        typedef hpx::traits::is_segmented_iterator<FwdIter1> is_segmented;
        return detail::transform_(std::forward<ExPolicy>(policy), first, last,
            dest, std::forward<F>(f), std::forward<Proj>(proj), is_segmented());
    }

    ///////////////////////////////////////////////////////////////////////////
    // transform binary predicate
    namespace detail {
        /// \cond NOINTERNAL
        template <typename F, typename Proj1, typename Proj2>
        struct transform_binary_projected
        {
            typename std::decay<F>::type& f_;
            typename std::decay<Proj1>::type& proj1_;
            typename std::decay<Proj2>::type& proj2_;

            template <typename Iter1, typename Iter2>
            HPX_HOST_DEVICE HPX_FORCEINLINE auto operator()(
                Iter1 curr1, Iter2 curr2) -> decltype(hpx::util::invoke(f_,
                hpx::util::invoke(proj1_, *curr1),
                hpx::util::invoke(proj2_, *curr2)))
            {
                return hpx::util::invoke(f_, hpx::util::invoke(proj1_, *curr1),
                    hpx::util::invoke(proj2_, *curr2));
            }
        };

        template <typename ExPolicy, typename F, typename Proj1, typename Proj2>
        struct transform_binary_iteration
        {
            typedef typename std::decay<ExPolicy>::type execution_policy_type;
            typedef typename std::decay<F>::type fun_type;
            typedef typename std::decay<Proj1>::type proj1_type;
            typedef typename std::decay<Proj2>::type proj2_type;

            fun_type f_;
            proj1_type proj1_;
            proj2_type proj2_;

            template <typename F_, typename Proj1_, typename Proj2_>
            HPX_HOST_DEVICE transform_binary_iteration(
                F_&& f, Proj1_&& proj1, Proj2_&& proj2)
              : f_(std::forward<F_>(f))
              , proj1_(std::forward<Proj1_>(proj1))
              , proj2_(std::forward<Proj2_>(proj2))
            {
            }

#if !defined(__NVCC__) && !defined(__CUDACC__)
            transform_binary_iteration(
                transform_binary_iteration const&) = default;
            transform_binary_iteration(transform_binary_iteration&&) = default;
#else
            HPX_HOST_DEVICE
            transform_binary_iteration(transform_binary_iteration const& rhs)
              : f_(rhs.f_)
              , proj1_(rhs.proj1_)
              , proj2_(rhs.proj2_)
            {
            }

            HPX_HOST_DEVICE
            transform_binary_iteration(transform_binary_iteration&& rhs)
              : f_(std::move(rhs.f_))
              , proj1_(std::move(rhs.proj1_))
              , proj2_(std::move(rhs.proj2_))
            {
            }
#endif

            transform_binary_iteration& operator=(
                transform_binary_iteration const&) = default;
            transform_binary_iteration& operator=(
                transform_binary_iteration&&) = default;

            template <typename Iter>
            HPX_HOST_DEVICE HPX_FORCEINLINE
                hpx::tuple<typename hpx::tuple_element<0,
                               typename Iter::iterator_tuple_type>::type,
                    typename hpx::tuple_element<1,
                        typename Iter::iterator_tuple_type>::type,
                    typename hpx::tuple_element<2,
                        typename Iter::iterator_tuple_type>::type>
                operator()(Iter part_begin, std::size_t part_size,
                    std::size_t /*part_index*/)
            {
                auto iters = part_begin.get_iterator_tuple();
                return util::transform_binary_loop_n<execution_policy_type>(
                    hpx::get<0>(iters), part_size, hpx::get<1>(iters),
                    hpx::get<2>(iters),
                    transform_binary_projected<F, Proj1, Proj2>{
                        f_, proj1_, proj2_});
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename IterTuple>
        struct transform_binary
          : public detail::algorithm<transform_binary<IterTuple>, IterTuple>
        {
            transform_binary()
              : transform_binary::algorithm("transform_binary")
            {
            }

            template <typename ExPolicy, typename InIter1, typename InIter2,
                typename OutIter, typename F, typename Proj1, typename Proj2>
            static util::in_in_out_result<InIter1, InIter2, OutIter> sequential(
                ExPolicy&&, InIter1 first1, InIter1 last1, InIter2 first2,
                OutIter dest, F&& f, Proj1&& proj1, Proj2&& proj2)
            {
                return util::transform_binary_loop<ExPolicy>(first1, last1,
                    first2, dest,
                    transform_binary_projected<F, Proj1, Proj2>{
                        f, proj1, proj2});
            }

            template <typename ExPolicy, typename FwdIter1B, typename FwdIter1E,
                typename FwdIter2, typename FwdIter3, typename F,
                typename Proj1, typename Proj2>
            static typename util::detail::algorithm_result<ExPolicy,
                util::in_in_out_result<FwdIter1B, FwdIter2, FwdIter3>>::type
            parallel(ExPolicy&& policy, FwdIter1B first1, FwdIter1E last1,
                FwdIter2 first2, FwdIter3 dest, F&& f, Proj1&& proj1,
                Proj2&& proj2)
            {
                if (first1 != last1)
                {
                    auto f1 =
                        transform_binary_iteration<ExPolicy, F, Proj1, Proj2>(
                            std::forward<F>(f), std::forward<Proj1>(proj1),
                            std::forward<Proj2>(proj2));

                    return util::detail::get_in_in_out_result(
                        util::foreach_partitioner<ExPolicy>::call(
                            std::forward<ExPolicy>(policy),
                            hpx::util::make_zip_iterator(first1, first2, dest),
                            detail::distance(first1, last1), std::move(f1),
                            util::projection_identity()));
                }

                using result_type =
                    util::in_in_out_result<FwdIter1B, FwdIter2, FwdIter3>;

                return util::detail::algorithm_result<ExPolicy,
                    result_type>::get(result_type{
                    std::move(first1), std::move(first2), std::move(dest)});
            }
        };

        template <typename ExPolicy, typename FwdIter1B, typename FwdIter1E,
            typename FwdIter2, typename FwdIter3, typename F, typename Proj1,
            typename Proj2>
        typename util::detail::algorithm_result<ExPolicy,
            util::in_in_out_result<FwdIter1B, FwdIter2, FwdIter3>>::type
        transform_(ExPolicy&& policy, FwdIter1B first1, FwdIter1E last1,
            FwdIter2 first2, FwdIter3 dest, F&& f, Proj1&& proj1, Proj2&& proj2,
            std::false_type)
        {
            static_assert((hpx::traits::is_forward_iterator<FwdIter1B>::value),
                "Requires at least forward iterator.");
            static_assert((hpx::traits::is_forward_iterator<FwdIter2>::value),
                "Requires at least forward iterator.");
            static_assert((hpx::traits::is_forward_iterator<FwdIter3>::value),
                "Requires at least forward iterator.");

            typedef hpx::is_sequenced_execution_policy<ExPolicy> is_seq;
            typedef util::in_in_out_result<FwdIter1B, FwdIter2, FwdIter3>
                result_type;

            return detail::transform_binary<result_type>().call(
                std::forward<ExPolicy>(policy), is_seq(), first1, last1, first2,
                dest, std::forward<F>(f), std::forward<Proj1>(proj1),
                std::forward<Proj2>(proj2));
        }

        template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
            typename FwdIter3, typename F, typename Proj1, typename Proj2>
        typename util::detail::algorithm_result<ExPolicy,
            util::in_in_out_result<FwdIter1, FwdIter2, FwdIter3>>::type
        transform_(ExPolicy&& policy, FwdIter1 first1, FwdIter1 last1,
            FwdIter2 first2, FwdIter3 dest, F&& f, Proj1&& proj1, Proj2&& proj2,
            std::true_type);
        /// \endcond
    }    // namespace detail

    // clang-format off
    template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
        typename FwdIter3, typename F,
        typename Proj1 = util::projection_identity,
        typename Proj2 = util::projection_identity,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy<
                ExPolicy>::value&& hpx::traits::is_iterator<FwdIter1>::value &&
            hpx::traits::is_iterator<FwdIter2>::value &&
            hpx::traits::is_iterator<FwdIter3>::value &&
            traits::is_projected<Proj1, FwdIter1>::value &&
            traits::is_projected<Proj2, FwdIter2>::value &&
            traits::is_indirect_callable<ExPolicy, F,
                traits::projected<Proj1, FwdIter1>,
                traits::projected<Proj2, FwdIter2>>::value
        )>
    // clang-format on
    HPX_DEPRECATED_V(1, 6,
        "hpx::parallel::transform is deprecated, use hpx::transform instead")
        typename util::detail::algorithm_result<ExPolicy,
            util::in_in_out_result<FwdIter1, FwdIter2, FwdIter3>>::type
        transform(ExPolicy&& policy, FwdIter1 first1, FwdIter1 last1,
            FwdIter2 first2, FwdIter3 dest, F&& f, Proj1&& proj1 = Proj1(),
            Proj2&& proj2 = Proj2())
    {
        typedef hpx::traits::is_segmented_iterator<FwdIter1> is_segmented;

        return detail::transform_(std::forward<ExPolicy>(policy), first1, last1,
            first2, dest, std::forward<F>(f), std::forward<Proj1>(proj1),
            std::forward<Proj2>(proj2), is_segmented());
    }

    ///////////////////////////////////////////////////////////////////////////
    // transform binary predicate
    namespace detail {
        /// \cond NOINTERNAL
        template <typename IterTuple>
        struct transform_binary2
          : public detail::algorithm<transform_binary2<IterTuple>, IterTuple>
        {
            transform_binary2()
              : transform_binary2::algorithm("transform_binary")
            {
            }

            template <typename ExPolicy, typename InIter1, typename InIter2,
                typename OutIter, typename F, typename Proj1, typename Proj2>
            static util::in_in_out_result<InIter1, InIter2, OutIter> sequential(
                ExPolicy&&, InIter1 first1, InIter1 last1, InIter2 first2,
                InIter2 last2, OutIter dest, F&& f, Proj1&& proj1,
                Proj2&& proj2)
            {
                return util::transform_binary_loop<ExPolicy>(first1, last1,
                    first2, last2, dest,
                    transform_binary_projected<F, Proj1, Proj2>{
                        f, proj1, proj2});
            }

            template <typename ExPolicy, typename FwdIter1B, typename FwdIter1E,
                typename FwdIter2B, typename FwdIter2E, typename FwdIter3,
                typename F, typename Proj1, typename Proj2>
            static typename util::detail::algorithm_result<ExPolicy,
                util::in_in_out_result<FwdIter1B, FwdIter2B, FwdIter3>>::type
            parallel(ExPolicy&& policy, FwdIter1B first1, FwdIter1E last1,
                FwdIter2B first2, FwdIter2E last2, FwdIter3 dest, F&& f,
                Proj1&& proj1, Proj2&& proj2)
            {
                if (first1 != last1 && first2 != last2)
                {
                    auto f1 =
                        transform_binary_iteration<ExPolicy, F, Proj1, Proj2>(
                            std::forward<F>(f), std::forward<Proj1>(proj1),
                            std::forward<Proj2>(proj2));

                    return util::detail::get_in_in_out_result(
                        util::foreach_partitioner<ExPolicy>::call(
                            std::forward<ExPolicy>(policy),
                            hpx::util::make_zip_iterator(first1, first2, dest),
                            (std::min)(detail::distance(first1, last1),
                                detail::distance(first2, last2)),
                            std::move(f1), util::projection_identity()));
                }

                using result_type =
                    util::in_in_out_result<FwdIter1B, FwdIter2B, FwdIter3>;

                return util::detail::algorithm_result<ExPolicy,
                    result_type>::get(result_type{
                    std::move(first1), std::move(first2), std::move(dest)});
            }
        };

        template <typename ExPolicy, typename FwdIter1B, typename FwdIter1E,
            typename FwdIter2B, typename FwdIter2E, typename FwdIter3,
            typename F, typename Proj1, typename Proj2>
        typename util::detail::algorithm_result<ExPolicy,
            util::in_in_out_result<FwdIter1B, FwdIter2B, FwdIter3>>::type
        transform_(ExPolicy&& policy, FwdIter1B first1, FwdIter1E last1,
            FwdIter2B first2, FwdIter2E last2, FwdIter3 dest, F&& f,
            Proj1&& proj1, Proj2&& proj2, std::false_type)
        {
            static_assert((hpx::traits::is_forward_iterator<FwdIter1B>::value),
                "Requires at least forward iterator.");
            static_assert((hpx::traits::is_forward_iterator<FwdIter2E>::value),
                "Requires at least forward iterator.");
            static_assert((hpx::traits::is_forward_iterator<FwdIter3>::value),
                "Requires at least forward iterator.");

            typedef hpx::is_sequenced_execution_policy<ExPolicy> is_seq;
            typedef util::in_in_out_result<FwdIter1B, FwdIter2E, FwdIter3>
                result_type;

            return detail::transform_binary2<result_type>().call(
                std::forward<ExPolicy>(policy), is_seq(), first1, last1, first2,
                last2, dest, std::forward<F>(f), std::forward<Proj1>(proj1),
                std::forward<Proj2>(proj2));
        }
        template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
            typename FwdIter3, typename F, typename Proj1, typename Proj2>
        typename util::detail::algorithm_result<ExPolicy,
            util::in_in_out_result<FwdIter1, FwdIter2, FwdIter3>>::type
        transform_(ExPolicy&& policy, FwdIter1 first1, FwdIter1 last1,
            FwdIter2 first2, FwdIter2 last2, FwdIter3 dest, F&& f,
            Proj1&& proj1, Proj2&& proj2, std::true_type);
        /// \endcond
    }    // namespace detail

    // clang-format off
    template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
        typename FwdIter3, typename F,
        typename Proj1 = util::projection_identity,
        typename Proj2 = util::projection_identity,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy<
                ExPolicy>::value&& hpx::traits::is_iterator<FwdIter1>::value &&
            hpx::traits::is_iterator<FwdIter2>::value &&
            hpx::traits::is_iterator<FwdIter3>::value &&
            traits::is_projected<Proj1, FwdIter1>::value &&
            traits::is_projected<Proj2, FwdIter2>::value &&
            traits::is_indirect_callable<ExPolicy, F,
                traits::projected<Proj1, FwdIter1>,
                traits::projected<Proj2, FwdIter2>>::value
        )>
    // clang-format on
    HPX_DEPRECATED_V(1, 6,
        "hpx::parallel::transform is deprecated, use hpx::transform instead")
        typename util::detail::algorithm_result<ExPolicy,
            util::in_in_out_result<FwdIter1, FwdIter2, FwdIter3>>::type
        transform(ExPolicy&& policy, FwdIter1 first1, FwdIter1 last1,
            FwdIter2 first2, FwdIter2 last2, FwdIter3 dest, F&& f,
            Proj1&& proj1 = Proj1(), Proj2&& proj2 = Proj2())
    {
        typedef hpx::traits::is_segmented_iterator<FwdIter1> is_segmented;

        return detail::transform_(std::forward<ExPolicy>(policy), first1, last1,
            first2, last2, dest, std::forward<F>(f), std::forward<Proj1>(proj1),
            std::forward<Proj2>(proj2), is_segmented());
    }
}}}    // namespace hpx::parallel::v1

#if defined(HPX_HAVE_THREAD_DESCRIPTION)
namespace hpx { namespace traits {
    template <typename ExPolicy, typename F, typename Proj>
    struct get_function_address<
        parallel::v1::detail::transform_iteration<ExPolicy, F, Proj>>
    {
        static std::size_t call(
            parallel::v1::detail::transform_iteration<ExPolicy, F, Proj> const&
                f) noexcept
        {
            return get_function_address<typename std::decay<F>::type>::call(
                f.f_);
        }
    };

    template <typename ExPolicy, typename F, typename Proj>
    struct get_function_annotation<
        parallel::v1::detail::transform_iteration<ExPolicy, F, Proj>>
    {
        static char const* call(
            parallel::v1::detail::transform_iteration<ExPolicy, F, Proj> const&
                f) noexcept
        {
            return get_function_annotation<typename std::decay<F>::type>::call(
                f.f_);
        }
    };

#if HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX)
    template <typename ExPolicy, typename F, typename Proj>
    struct get_function_annotation_itt<
        parallel::v1::detail::transform_iteration<ExPolicy, F, Proj>>
    {
        static util::itt::string_handle call(
            parallel::v1::detail::transform_iteration<ExPolicy, F, Proj> const&
                f) noexcept
        {
            return get_function_annotation_itt<
                typename std::decay<F>::type>::call(f.f_);
        }
    };
#endif
}}    // namespace hpx::traits
#endif

namespace hpx {
    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::transform
    HPX_INLINE_CONSTEXPR_VARIABLE struct transform_t final
      : hpx::functional::tag<transform_t>
    {
    private:
        // clang-format off
        template <typename FwdIter1, typename FwdIter2,
            typename F, HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator<FwdIter1>::value &&
                hpx::traits::is_iterator<FwdIter2>::value
            )>
        // clang-format on
        friend FwdIter2 tag_invoke(hpx::transform_t, FwdIter1 first,
            FwdIter1 last, FwdIter2 dest, F&& f)
        {
            return parallel::util::get_second_element(
                parallel::v1::detail::transform_(hpx::execution::seq, first,
                    last, dest, std::forward<F>(f),
                    hpx::parallel::util::projection_identity(),
                    std::false_type{}));
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
            typename F, HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_iterator<FwdIter1>::value &&
                hpx::traits::is_iterator<FwdIter2>::value
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            FwdIter2>::type
        tag_invoke(hpx::transform_t, ExPolicy&& policy, FwdIter1 first,
            FwdIter1 last, FwdIter2 dest, F&& f)
        {
            typedef hpx::traits::is_segmented_iterator<FwdIter1> is_segmented;

            return parallel::util::get_second_element(
                parallel::v1::detail::transform_(std::forward<ExPolicy>(policy),
                    first, last, dest, std::forward<F>(f),
                    hpx::parallel::util::projection_identity(),
                    is_segmented()));
        }

        // clang-format off
        template <typename FwdIter1, typename FwdIter2, typename FwdIter3,
            typename F, HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator<FwdIter1>::value &&
                hpx::traits::is_iterator<FwdIter2>::value &&
                hpx::traits::is_iterator<FwdIter3>::value
            )>
        // clang-format on
        friend FwdIter3 tag_invoke(hpx::transform_t, FwdIter1 first1,
            FwdIter1 last1, FwdIter2 first2, FwdIter3 dest, F&& f)
        {
            using proj_id = hpx::parallel::util::projection_identity;

            return parallel::util::get_third_element(
                parallel::v1::detail::transform_(hpx::execution::seq, first1,
                    last1, first2, dest, std::forward<F>(f), proj_id(),
                    proj_id(), std::false_type{}));
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter1,
            typename FwdIter2, typename FwdIter3,
            typename F, HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_iterator<FwdIter1>::value &&
                hpx::traits::is_iterator<FwdIter2>::value &&
                hpx::traits::is_iterator<FwdIter3>::value
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            FwdIter3>::type
        tag_invoke(hpx::transform_t, ExPolicy&& policy, FwdIter1 first1,
            FwdIter1 last1, FwdIter2 first2, FwdIter3 dest, F&& f)
        {
            using is_segmented = std::integral_constant<bool,
                hpx::traits::is_segmented_iterator<FwdIter1>::value ||
                    hpx::traits::is_segmented_iterator<FwdIter2>::value>;
            using proj_id = hpx::parallel::util::projection_identity;

            return parallel::util::get_third_element(
                parallel::v1::detail::transform_(std::forward<ExPolicy>(policy),
                    first1, last1, first2, dest, std::forward<F>(f), proj_id(),
                    proj_id(), is_segmented()));
        }

    } transform{};
}    // namespace hpx

#endif    // DOXYGEN
