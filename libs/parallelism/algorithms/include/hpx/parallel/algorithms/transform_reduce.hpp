//  Copyright (c) 2015 Daniel Bourgeois
//  Copyright (c) 2017-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/transform_reduce.hpp

#pragma once

#if defined(DOXYGEN)
namespace hpx {
    // clang-format off

    /// Returns GENERALIZED_SUM(red_op, init, conv_op(*first), ...,
    /// conv_op(*(first + (last - first) - 1))).
    ///
    /// \note   Complexity: O(\a last - \a first) applications of the
    ///         predicates \a red_op and \a conv_op.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter      The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a copy_if requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    /// \tparam T           The type of the value to be used as initial (and
    ///                     intermediate) values (deduced).
    /// \tparam Reduce      The type of the binary function object used for
    ///                     the reduction operation.
    /// \tparam Convert     The type of the unary function object used to
    ///                     transform the elements of the input sequence before
    ///                     invoking the reduce function.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param conv_op      Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last). This is a
    ///                     unary predicate. The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     R fun(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that an object of
    ///                     type \a FwdIter can be dereferenced and then
    ///                     implicitly converted to Type.
    ///                     The type \a R must be such that an object of this
    ///                     type can be implicitly converted to \a T.
    /// \param init         The initial value for the generalized sum.
    /// \param red_op       Specifies the function (or function object) which
    ///                     will be invoked for each of the values returned
    ///                     from the invocation of \a conv_op. This is a
    ///                     binary predicate. The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     Ret fun(const Type1 &a, const Type2 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it.
    ///                     The types \a Type1, \a Type2, and \a Ret must be
    ///                     such that an object of a type as returned from
    ///                     \a conv_op can be implicitly converted to any
    ///                     of those types.
    ///
    /// The reduce operations in the parallel \a transform_reduce algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The reduce operations in the parallel \a transform_reduce algorithm invoked
    /// with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a transform_reduce algorithm returns a \a hpx::future<T> if the
    ///           execution policy is of type \a parallel_task_policy and
    ///           returns \a T otherwise.
    ///           The \a transform_reduce algorithm returns the result of the
    ///           generalized sum over the values returned from \a conv_op when
    ///           applied to the elements given by the input range
    ///           [first, last).
    ///
    /// \note   GENERALIZED_SUM(op, a1, ..., aN) is defined as follows:
    ///         * a1 when N is 1
    ///         * op(GENERALIZED_SUM(op, b1, ..., bK), GENERALIZED_SUM(op, bM, ..., bN)),
    ///           where:
    ///           * b1, ..., bN may be any permutation of a1, ..., aN and
    ///           * 1 < K+1 = M <= N.
    ///
    /// The difference between \a transform_reduce and \a accumulate is
    /// that the behavior of transform_reduce may be non-deterministic for
    /// non-associative or non-commutative binary predicate.
    ///
    template <typename ExPolicy, typename FwdIter, typename T, typename Reduce,
        typename Convert>
    typename util::detail::algorithm_result<ExPolicy, T>::type
    transform_reduce(ExPolicy&& policy, FwdIter first, FwdIter last, T init,
        Reduce&& red_op, Convert&& conv_op);

    ///////////////////////////////////////////////////////////////////////////
    /// Returns the result of accumulating init with the inner products of the
    /// pairs formed by the elements of two ranges starting at first1 and
    /// first2.
    ///
    /// \note   Complexity: O(\a last - \a first) applications of the
    ///         predicate \a op2.
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter1    The type of the first source iterators used
    ///                     (deduced). This iterator type must meet the
    ///                     requirements of an forward iterator.
    /// \tparam FwdIter2    The type of the second source iterators used
    ///                     (deduced). This iterator type must meet the
    ///                     requirements of an forward iterator.
    /// \tparam T           The type of the value to be used as return)
    ///                     values (deduced).
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first1       Refers to the beginning of the first sequence of
    ///                     elements the result will be calculated with.
    /// \param last1        Refers to the end of the first sequence of elements
    ///                     the algorithm will be applied to.
    /// \param first2       Refers to the beginning of the second sequence of
    ///                     elements the result will be calculated with.
    /// \param init         The initial value for the sum.
    ///
    /// The operations in the parallel \a transform_reduce algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The operations in the parallel \a transform_reduce algorithm invoked
    /// with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a transform_reduce algorithm returns a \a hpx::future<T> if
    ///           the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a T otherwise.
    ///
    template <typename ExPolicy, typename FwdIter1, typename FwdIter2, typename T>
    typename util::detail::algorithm_result<ExPolicy, T>::type
    transform_reduce(ExPolicy&& policy, FwdIter1 first1, FwdIter1 last1,
        FwdIter2 first2, T init);

    ///////////////////////////////////////////////////////////////////////////
    /// Returns the result of accumulating init with the inner products of the
    /// pairs formed by the elements of two ranges starting at first1 and
    /// first2.
    ///
    /// \note   Complexity: O(\a last - \a first) applications of the
    ///         predicate \a op2.
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter1    The type of the first source iterators used
    ///                     (deduced). This iterator type must meet the
    ///                     requirements of an forward iterator.
    /// \tparam FwdIter2    The type of the second source iterators used
    ///                     (deduced). This iterator type must meet the
    ///                     requirements of an forward iterator.
    /// \tparam T           The type of the value to be used as return)
    ///                     values (deduced).
    /// \tparam Reduce      The type of the binary function object used for
    ///                     the multiplication operation.
    /// \tparam Convert     The type of the unary function object used to
    ///                     transform the elements of the input sequence before
    ///                     invoking the reduce function.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first1       Refers to the beginning of the first sequence of
    ///                     elements the result will be calculated with.
    /// \param last1        Refers to the end of the first sequence of elements
    ///                     the algorithm will be applied to.
    /// \param first2       Refers to the beginning of the second sequence of
    ///                     elements the result will be calculated with.
    /// \param init         The initial value for the sum.
    /// \param red_op       Specifies the function (or function object) which
    ///                     will be invoked for the initial value and each
    ///                     of the return values of \a op2.
    ///                     This is a binary predicate. The
    ///                     signature of this predicate should be equivalent to
    ///                     should be equivalent to:
    ///                     \code
    ///                     Ret fun(const Type1 &a, const Type1 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it.
    ///                     The type \a Ret must be
    ///                     such that it can be implicitly converted to a type
    ///                     of \a T.
    /// \param conv_op      Specifies the function (or function object) which
    ///                     will be invoked for each of the input values
    ///                     of the sequence. This is a binary predicate. The
    ///                     signature of this predicate should be equivalent to
    ///                     \code
    ///                     Ret fun(const Type1 &a, const Type2 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it.
    ///                     The type \a Ret must be such that it can be
    ///                     implicitly converted to an object for the second
    ///                     argument type of \a op1.
    ///
    /// The operations in the parallel \a transform_reduce algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The operations in the parallel \a transform_reduce algorithm invoked
    /// with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a transform_reduce algorithm returns a \a hpx::future<T>
    ///           if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a T otherwise.
    ///
    template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
        typename T, typename Reduce, typename Convert>
    typename util::detail::algorithm_result<ExPolicy, T>::type
    transform_reduce(ExPolicy&& policy, FwdIter1 first1, FwdIter1 last1,
        FwdIter2 first2, T init, Reduce&& red_op, Convert&& conv_op);

    // clang-format on
}    // namespace hpx

#else    // DOXYGEN

#include <hpx/config.hpp>
#include <hpx/algorithms/traits/segmented_iterator_traits.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/functional/invoke.hpp>
#include <hpx/functional/invoke_result.hpp>
#include <hpx/functional/tag_invoke.hpp>
#include <hpx/functional/traits/is_invocable.hpp>
#include <hpx/iterator_support/range.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/pack_traversal/unwrap.hpp>

#include <hpx/execution/algorithms/detail/predicates.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/accumulate.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/detail/distance.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/loop.hpp>
#include <hpx/parallel/util/partitioner.hpp>
#include <hpx/parallel/util/zip_iterator.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace parallel { inline namespace v1 {

    ///////////////////////////////////////////////////////////////////////////
    // transform_reduce
    namespace detail {

        template <typename T>
        struct transform_reduce
          : public detail::algorithm<transform_reduce<T>, T>
        {
            transform_reduce()
              : transform_reduce::algorithm("transform_reduce")
            {
            }

            template <typename ExPolicy, typename Iter, typename Sent,
                typename T_, typename Reduce, typename Convert>
            static T sequential(ExPolicy, Iter first, Sent last, T_&& init,
                Reduce&& r, Convert&& conv)
            {
                using value_type =
                    typename std::iterator_traits<Iter>::value_type;

                return detail::accumulate(first, last, std::forward<T_>(init),
                    [&r, &conv](T const& res, value_type const& next) -> T {
                        return hpx::util::invoke(
                            r, res, hpx::util::invoke(conv, next));
                    });
            }

            template <typename ExPolicy, typename Iter, typename Sent,
                typename T_, typename Reduce, typename Convert>
            static typename util::detail::algorithm_result<ExPolicy, T>::type
            parallel(ExPolicy&& policy, Iter first, Sent last, T_&& init,
                Reduce&& r, Convert&& conv)
            {
                if (first == last)
                {
                    T init_ = init;
                    return util::detail::algorithm_result<ExPolicy, T>::get(
                        std::move(init_));
                }

                using reference =
                    typename std::iterator_traits<Iter>::reference;

                auto f1 = [r, conv = std::forward<Convert>(conv)](
                              Iter part_begin, std::size_t part_size) -> T {
                    T val = hpx::util::invoke(conv, *part_begin);
                    return util::accumulate_n(++part_begin, --part_size,
                        std::move(val), [=](T const& res, reference next) -> T {
                            return hpx::util::invoke(
                                r, res, hpx::util::invoke(conv, next));
                        });
                };

                return util::partitioner<ExPolicy, T>::call(
                    std::forward<ExPolicy>(policy), first,
                    detail::distance(first, last), std::move(f1),
                    hpx::util::unwrapping([init = std::forward<T_>(init),
                                              r = std::forward<Reduce>(r)](
                                              std::vector<T>&& results) -> T {
                        return util::accumulate_n(hpx::util::begin(results),
                            hpx::util::size(results), init, r);
                    }));
            }
        };

        template <typename ExPolicy, typename Iter, typename Sent, typename T,
            typename Reduce, typename Convert>
        inline typename util::detail::algorithm_result<ExPolicy,
            typename std::decay<T>::type>::type
        transform_reduce_(ExPolicy&& policy, Iter first, Sent last, T&& init,
            Reduce&& red_op, Convert&& conv_op, std::false_type)
        {
            using is_seq = hpx::is_sequenced_execution_policy<ExPolicy>;
            using init_type = typename std::decay<T>::type;

            return transform_reduce<init_type>().call(
                std::forward<ExPolicy>(policy), is_seq(), first, last,
                std::forward<T>(init), std::forward<Reduce>(red_op),
                std::forward<Convert>(conv_op));
        }

        // forward declare the segmented version of this algorithm
        template <typename ExPolicy, typename FwdIter, typename T,
            typename Reduce, typename Convert>
        typename util::detail::algorithm_result<ExPolicy,
            typename std::decay<T>::type>::type
        transform_reduce_(ExPolicy&& policy, FwdIter first, FwdIter last,
            T&& init, Reduce&& red_op, Convert&& conv_op, std::true_type);

    }    // namespace detail

    // clang-format off
    template <typename ExPolicy, typename FwdIter, typename T, typename Reduce,
        typename Convert,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy<ExPolicy>::value &&
            hpx::traits::is_iterator<FwdIter>::value &&
            hpx::is_invocable_v<Convert,
                typename std::iterator_traits<FwdIter>::value_type> &&
            hpx::is_invocable_v<Reduce,
                typename hpx::util::invoke_result<
                    Convert, typename std::iterator_traits<FwdIter>::value_type
                >::type,
                typename hpx::util::invoke_result<
                    Convert, typename std::iterator_traits<FwdIter>::value_type
                >::type
            >
        )>
    // clang-format on
    HPX_DEPRECATED_V(1, 6,
        "hpx::parallel::transform_reduce is deprecated, use "
        "hpx::transform_reduce instead")
        typename util::detail::algorithm_result<ExPolicy, T>::type
        transform_reduce(ExPolicy&& policy, FwdIter first, FwdIter last, T init,
            Reduce&& red_op, Convert&& conv_op)
    {
        static_assert((hpx::traits::is_forward_iterator<FwdIter>::value),
            "Requires at least forward iterator.");

        typedef hpx::traits::is_segmented_iterator<FwdIter> is_segmented;

#if defined(HPX_GCC_VERSION) && HPX_GCC_VERSION >= 100000
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
        return detail::transform_reduce_(std::forward<ExPolicy>(policy), first,
            last, std::move(init), std::forward<Reduce>(red_op),
            std::forward<Convert>(conv_op), is_segmented());
#if defined(HPX_GCC_VERSION) && HPX_GCC_VERSION >= 100000
#pragma GCC diagnostic pop
#endif
    }

    ///////////////////////////////////////////////////////////////////////////
    // transform_reduce_binary
    namespace detail {

        template <typename F>
        struct transform_reduce_binary_indirect
        {
            F f_;

            template <typename Iter1, typename Iter2>
            HPX_HOST_DEVICE HPX_FORCEINLINE constexpr auto operator()(Iter1 it1,
                Iter2 it2) -> decltype(hpx::util::invoke(f_, *it1, *it2))
            {
                return hpx::util::invoke(f_, *it1, *it2);
            }
        };

        template <typename Op1, typename Op2, typename T>
        struct transform_reduce_binary_partition
        {
            typedef typename std::decay<T>::type value_type;

            Op1 op1_;
            Op2 op2_;
            value_type& part_sum_;

            template <typename Iter1, typename Iter2>
            HPX_HOST_DEVICE HPX_FORCEINLINE constexpr void operator()(
                Iter1 it1, Iter2 it2)
            {
                part_sum_ = hpx::util::invoke(
                    op1_, part_sum_, hpx::util::invoke(op2_, *it1, *it2));
            }
        };

        template <typename T>
        struct transform_reduce_binary
          : public detail::algorithm<transform_reduce_binary<T>, T>
        {
            transform_reduce_binary()
              : transform_reduce_binary::algorithm("transform_reduce_binary")
            {
            }

            template <typename ExPolicy, typename Iter, typename Sent,
                typename Iter2, typename T_, typename Op1, typename Op2>
            static T sequential(ExPolicy&& /* policy */, Iter first1,
                Sent last1, Iter2 first2, T_ init, Op1&& op1, Op2&& op2)
            {
                if (first1 == last1)
                {
                    return init;
                }

                // check whether we should apply vectorization
                if (!util::loop_optimization<ExPolicy>(first1, last1))
                {
                    util::loop2<ExPolicy>(std::false_type(), first1, last1,
                        first2,
                        transform_reduce_binary_partition<Op1, Op2, T>{
                            std::forward<Op1>(op1), std::forward<Op2>(op2),
                            init});
                    return init;
                }

                // loop_step properly advances the iterators
                auto part_sum = util::loop_step<ExPolicy>(std::true_type(),
                    transform_reduce_binary_indirect<Op2>{op2}, first1, first2);

                std::pair<Iter, Iter2> p = util::loop2<ExPolicy>(
                    std::true_type(), first1, last1, first2,
                    transform_reduce_binary_partition<Op1, Op2,
                        decltype(part_sum)>{op1, op2, part_sum});

                // this is to support vectorization, it will call op1 for each
                // of the elements of a value-pack
                auto result = util::detail::accumulate_values<ExPolicy>(
                    [&op1](T const& sum, T&& val) -> T {
                        return hpx::util::invoke(op1, sum, val);
                    },
                    std::move(part_sum), std::move(init));

                // the vectorization might not cover all of the sequences,
                // handle the remainder directly
                if (p.first != last1)
                {
                    util::loop2<ExPolicy>(std::false_type(), p.first, last1,
                        p.second,
                        transform_reduce_binary_partition<Op1, Op2,
                            decltype(result)>{std::forward<Op1>(op1),
                            std::forward<Op2>(op2), result});
                }

                return util::detail::extract_value<ExPolicy>(result);
            }

            template <typename ExPolicy, typename Iter, typename Sent,
                typename Iter2, typename T_, typename Op1, typename Op2>
            static typename util::detail::algorithm_result<ExPolicy, T>::type
            parallel(ExPolicy&& policy, Iter first1, Sent last1, Iter2 first2,
                T_&& init, Op1&& op1, Op2&& op2)
            {
                typedef util::detail::algorithm_result<ExPolicy, T> result;
                typedef hpx::util::zip_iterator<Iter, Iter2> zip_iterator;
                typedef typename std::iterator_traits<Iter>::difference_type
                    difference_type;

                if (first1 == last1)
                {
                    return result::get(std::forward<T_>(init));
                }

                difference_type count = detail::distance(first1, last1);

                auto f1 = [op1, op2 = std::forward<Op2>(op2)](
                              zip_iterator part_begin,
                              std::size_t part_size) mutable -> T {
                    auto iters = part_begin.get_iterator_tuple();
                    Iter it1 = hpx::get<0>(iters);
                    Iter2 it2 = hpx::get<1>(iters);

                    Iter last1 = it1;
                    std::advance(last1, part_size);

                    if (!util::loop_optimization<ExPolicy>(it1, last1))
                    {
                        // loop_step properly advances the iterators
                        auto result =
                            util::loop_step<ExPolicy>(std::false_type(),
                                transform_reduce_binary_indirect<Op2>{op2}, it1,
                                it2);

                        util::loop2<ExPolicy>(std::false_type(), it1, last1,
                            it2,
                            transform_reduce_binary_partition<Op1, Op2,
                                decltype(result)>{std::forward<Op1>(op1),
                                std::forward<Op2>(op2), result});

                        return util::detail::extract_value<ExPolicy>(result);
                    }

                    // loop_step properly advances the iterators
                    auto part_sum = util::loop_step<ExPolicy>(std::true_type(),
                        transform_reduce_binary_indirect<Op2>{op2}, it1, it2);

                    std::pair<Iter, Iter2> p =
                        util::loop2<ExPolicy>(std::true_type(), it1, last1, it2,
                            transform_reduce_binary_partition<Op1, Op2,
                                decltype(part_sum)>{op1, op2, part_sum});

                    // this is to support vectorization, it will call op1
                    // for each of the elements of a value-pack
                    auto result = util::detail::accumulate_values<ExPolicy>(
                        [&op1](T const& sum, T&& val) -> T {
                            return hpx::util::invoke(op1, sum, val);
                        },
                        part_sum);

                    // the vectorization might not cover all of the sequences,
                    // handle the remainder directly
                    if (p.first != last1)
                    {
                        util::loop2<ExPolicy>(std::false_type(), p.first, last1,
                            p.second,
                            transform_reduce_binary_partition<Op1, Op2,
                                decltype(result)>{std::forward<Op1>(op1),
                                std::forward<Op2>(op2), result});
                    }

                    return util::detail::extract_value<ExPolicy>(result);
                };

                using hpx::util::make_zip_iterator;

                return util::partitioner<ExPolicy, T>::call(
                    std::forward<ExPolicy>(policy),
                    make_zip_iterator(first1, first2), count, std::move(f1),
                    [init = std::forward<T_>(init),
                        op1 = std::forward<Op1>(op1)](
                        std::vector<hpx::future<T>>&& results) mutable -> T {
                        T ret = std::move(init);
                        for (auto&& fut : results)
                        {
                            ret = hpx::util::invoke(
                                op1, std::move(ret), fut.get());
                        }
                        return ret;
                    });
            }
        };

        template <typename ExPolicy, typename Iter, typename Sent,
            typename Iter2, typename T, typename Reduce, typename Convert>
        typename util::detail::algorithm_result<ExPolicy, T>::type
        transform_reduce_(ExPolicy&& policy, Iter first1, Sent last1,
            Iter2 first2, T init, Reduce&& red_op, Convert&& conv_op,
            std::false_type)
        {
            static_assert((hpx::traits::is_forward_iterator<Iter>::value),
                "Requires at least forward iterator.");
            static_assert((hpx::traits::is_forward_iterator<Iter>::value),
                "Requires at least forward iterator.");

            using is_seq = hpx::is_sequenced_execution_policy<ExPolicy>;

            return detail::transform_reduce_binary<T>().call(
                std::forward<ExPolicy>(policy), is_seq(), first1, last1, first2,
                std::move(init), std::forward<Reduce>(red_op),
                std::forward<Convert>(conv_op));
        }

        template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
            typename T, typename Reduce, typename Convert>
        typename util::detail::algorithm_result<ExPolicy, T>::type
        transform_reduce_(ExPolicy&& policy, FwdIter1 first1, FwdIter1 last1,
            FwdIter2 first2, T init, Reduce&& red_op, Convert&& conv_op,
            std::true_type);
    }    // namespace detail

    // clang-format off
    template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
        typename T,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy<ExPolicy>::value &&
            hpx::traits::is_iterator<FwdIter1>::value &&
            hpx::traits::is_iterator<FwdIter2>::value
        )>
    // clang-format on
    HPX_DEPRECATED_V(1, 6,
        "hpx::parallel::transform_reduce is deprecated, use "
        "hpx::transform_reduce instead")
        typename util::detail::algorithm_result<ExPolicy, T>::type
        transform_reduce(ExPolicy&& policy, FwdIter1 first1, FwdIter1 last1,
            FwdIter2 first2, T init)
    {
        typedef hpx::traits::is_segmented_iterator<FwdIter1> is_segmented;
        return detail::transform_reduce_(std::forward<ExPolicy>(policy), first1,
            last1, first2, std::move(init), detail::plus(),
            detail::multiplies(), is_segmented());
    }

    // clang-format off
    template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
        typename T, typename Reduce, typename Convert,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy<ExPolicy>::value &&
            hpx::traits::is_iterator<FwdIter1>::value &&
            hpx::traits::is_iterator<FwdIter2>::value &&
            hpx::is_invocable_v<Convert,
                typename std::iterator_traits<FwdIter1>::value_type,
                typename std::iterator_traits<FwdIter2>::value_type
            > &&
            hpx::is_invocable_v<Reduce,
                typename hpx::util::invoke_result<Convert,
                    typename std::iterator_traits<FwdIter1>::value_type,
                    typename std::iterator_traits<FwdIter2>::value_type
                >::type,
                typename hpx::util::invoke_result<Convert,
                    typename std::iterator_traits<FwdIter1>::value_type,
                    typename std::iterator_traits<FwdIter2>::value_type
                >::type
            >
        )>
    // clang-format on
    HPX_DEPRECATED_V(1, 6,
        "hpx::parallel::transform_reduce is deprecated, use "
        "hpx::transform_reduce instead")
        typename util::detail::algorithm_result<ExPolicy, T>::type
        transform_reduce(ExPolicy&& policy, FwdIter1 first1, FwdIter1 last1,
            FwdIter2 first2, T init, Reduce&& red_op, Convert&& conv_op)
    {
        typedef hpx::traits::is_segmented_iterator<FwdIter1> is_segmented;
        return detail::transform_reduce_(std::forward<ExPolicy>(policy), first1,
            last1, first2, std::move(init), std::forward<Reduce>(red_op),
            std::forward<Convert>(conv_op), is_segmented());
    }

}}}    // namespace hpx::parallel::v1

namespace hpx {

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::transform_reduce
    HPX_INLINE_CONSTEXPR_VARIABLE struct transform_reduce_t final
      : hpx::functional::tag<transform_reduce_t>
    {
    private:
        // clang-format off
        template <typename ExPolicy, typename FwdIter, typename T,
            typename Reduce, typename Convert,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_iterator<FwdIter>::value &&
                hpx::is_invocable_v<Convert,
                   typename std::iterator_traits<FwdIter>::value_type> &&
                hpx::is_invocable_v<Reduce,
                   typename hpx::util::invoke_result<Convert,
                       typename std::iterator_traits<FwdIter>::value_type
                   >::type,
                   typename hpx::util::invoke_result<Convert,
                       typename std::iterator_traits<FwdIter>::value_type
                   >::type
                >
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            T>::type
        tag_invoke(transform_reduce_t, ExPolicy&& policy, FwdIter first,
            FwdIter last, T init, Reduce&& red_op, Convert&& conv_op)
        {
            static_assert((hpx::traits::is_forward_iterator<FwdIter>::value),
                "Requires at least forward iterator.");

            using is_segmented = hpx::traits::is_segmented_iterator<FwdIter>;

            return hpx::parallel::v1::detail::transform_reduce_(
                std::forward<ExPolicy>(policy), first, last, std::move(init),
                std::forward<Reduce>(red_op), std::forward<Convert>(conv_op),
                is_segmented());
        }

        // clang-format off
        template <typename FwdIter, typename T, typename Reduce,
            typename Convert,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator<FwdIter>::value &&
                hpx::is_invocable_v<Convert,
                   typename std::iterator_traits<FwdIter>::value_type> &&
                hpx::is_invocable_v<Reduce,
                   typename hpx::util::invoke_result<Convert,
                       typename std::iterator_traits<FwdIter>::value_type
                   >::type,
                   typename hpx::util::invoke_result<Convert,
                       typename std::iterator_traits<FwdIter>::value_type
                   >::type
                >
            )>
        // clang-format on
        friend T tag_invoke(transform_reduce_t, FwdIter first, FwdIter last,
            T init, Reduce&& red_op, Convert&& conv_op)
        {
            static_assert((hpx::traits::is_forward_iterator<FwdIter>::value),
                "Requires at least forward iterator.");

            using is_segmented = hpx::traits::is_segmented_iterator<FwdIter>;

            return hpx::parallel::v1::detail::transform_reduce_(
                hpx::execution::seq, first, last, std::move(init),
                std::forward<Reduce>(red_op), std::forward<Convert>(conv_op),
                is_segmented());
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
            typename T,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_iterator<FwdIter1>::value &&
                hpx::traits::is_iterator<FwdIter2>::value
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            T>::type
        tag_invoke(transform_reduce_t, ExPolicy&& policy, FwdIter1 first1,
            FwdIter1 last1, FwdIter2 first2, T init)
        {
            using is_segmented = hpx::traits::is_segmented_iterator<FwdIter1>;

            return hpx::parallel::v1::detail::transform_reduce_(
                std::forward<ExPolicy>(policy), first1, last1, first2,
                std::move(init), hpx::parallel::v1::detail::plus(),
                hpx::parallel::v1::detail::multiplies(), is_segmented());
        }

        // clang-format off
        template <typename FwdIter1, typename FwdIter2, typename T,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator<FwdIter1>::value &&
                hpx::traits::is_iterator<FwdIter2>::value
            )>
        // clang-format on
        friend T tag_invoke(transform_reduce_t, FwdIter1 first1, FwdIter1 last1,
            FwdIter2 first2, T init)
        {
            using is_segmented = hpx::traits::is_segmented_iterator<FwdIter1>;

            return hpx::parallel::v1::detail::transform_reduce_(
                hpx::execution::seq, first1, last1, first2, std::move(init),
                hpx::parallel::v1::detail::plus(),
                hpx::parallel::v1::detail::multiplies(), is_segmented());
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
            typename T, typename Reduce, typename Convert,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_iterator<FwdIter1>::value &&
                hpx::traits::is_iterator<FwdIter2>::value &&
                hpx::is_invocable_v<Convert,
                    typename std::iterator_traits<FwdIter1>::value_type,
                    typename std::iterator_traits<FwdIter2>::value_type
                > &&
                hpx::is_invocable_v<Reduce,
                    typename hpx::util::invoke_result<Convert,
                        typename std::iterator_traits<FwdIter1>::value_type,
                        typename std::iterator_traits<FwdIter2>::value_type
                    >::type,
                    typename hpx::util::invoke_result<Convert,
                        typename std::iterator_traits<FwdIter1>::value_type,
                        typename std::iterator_traits<FwdIter2>::value_type
                    >::type
                >
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            T>::type
        tag_invoke(transform_reduce_t, ExPolicy&& policy, FwdIter1 first1,
            FwdIter1 last1, FwdIter2 first2, T init, Reduce&& red_op,
            Convert&& conv_op)
        {
            using is_segmented = hpx::traits::is_segmented_iterator<FwdIter1>;

            return hpx::parallel::v1::detail::transform_reduce_(
                std::forward<ExPolicy>(policy), first1, last1, first2,
                std::move(init), std::forward<Reduce>(red_op),
                std::forward<Convert>(conv_op), is_segmented());
        }

        // clang-format off
        template <typename FwdIter1, typename FwdIter2, typename T,
            typename Reduce, typename Convert,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator<FwdIter1>::value &&
                hpx::traits::is_iterator<FwdIter2>::value &&
                hpx::is_invocable_v<Convert,
                    typename std::iterator_traits<FwdIter1>::value_type,
                    typename std::iterator_traits<FwdIter2>::value_type
                > &&
                hpx::is_invocable_v<Reduce,
                    typename hpx::util::invoke_result<Convert,
                        typename std::iterator_traits<FwdIter1>::value_type,
                        typename std::iterator_traits<FwdIter2>::value_type
                    >::type,
                    typename hpx::util::invoke_result<Convert,
                        typename std::iterator_traits<FwdIter1>::value_type,
                        typename std::iterator_traits<FwdIter2>::value_type
                    >::type
                >
            )>
        // clang-format on
        friend T tag_invoke(transform_reduce_t, FwdIter1 first1, FwdIter1 last1,
            FwdIter2 first2, T init, Reduce&& red_op, Convert&& conv_op)
        {
            using is_segmented = hpx::traits::is_segmented_iterator<FwdIter1>;

            return hpx::parallel::v1::detail::transform_reduce_(
                hpx::execution::seq, first1, last1, first2, std::move(init),
                std::forward<Reduce>(red_op), std::forward<Convert>(conv_op),
                is_segmented());
        }
    } transform_reduce{};
}    // namespace hpx

#endif    // DOXYGEN
