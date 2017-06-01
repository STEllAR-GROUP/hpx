//  Copyright (c) 2015 Daniel Bourgeois
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/tranform_reduce_binary.hpp

#if !defined(HPX_PARALLEL_ALGORITHM_TRANFORM_REDUCE_BINARY_JUL_15_2015_0730AM)
#define HPX_PARALLEL_ALGORITHM_TRANFORM_REDUCE_BINARY_JUL_15_2015_0730AM

#include <hpx/config.hpp>
#include <hpx/traits/concepts.hpp>
#include <hpx/traits/is_callable.hpp>
#include <hpx/traits/is_iterator.hpp>
#include <hpx/util/invoke.hpp>
#include <hpx/util/result_of.hpp>
#include <hpx/util/zip_iterator.hpp>

#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/loop.hpp>
#include <hpx/parallel/util/partitioner.hpp>
#include <hpx/util/unused.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace parallel { inline namespace v1
{
    ///////////////////////////////////////////////////////////////////////////
    // transform_reduce
    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename F>
        struct transform_reduce_binary_indirect
        {
            typename hpx::util::decay<F>::type& f_;

            template <typename Iter1, typename Iter2>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            auto operator()(Iter1 it1, Iter2 it2)
            ->  decltype(hpx::util::invoke(f_, *it1, *it2))
            {
                return hpx::util::invoke(f_, *it1, *it2);
            }
        };

        template <typename Op1, typename Op2, typename T>
        struct transform_reduce_binary_partition
        {
            typedef typename hpx::util::decay<Op1>::type op1_type;
            typedef typename hpx::util::decay<Op2>::type op2_type;
            typedef typename hpx::util::decay<T>::type value_type;

            op1_type& op1_;
            op2_type& op2_;
            value_type& part_sum_;

            template <typename Iter1, typename Iter2>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            void operator()(Iter1 it1, Iter2 it2)
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
            {}

            template <typename ExPolicy, typename InIter1, typename InIter2,
                typename T_, typename Op1, typename Op2>
            static T
            sequential(ExPolicy && policy, InIter1 first1, InIter1 last1,
                InIter2 first2, T_ init, Op1 && op1, Op2 && op2)
            {
                if (first1 == last1)
                    return std::move(init);

                // check whether we should apply vectorization
                if (!util::loop_optimization<ExPolicy>(first1, last1))
                {
                    util::loop2<ExPolicy>(
                        std::false_type(), first1, last1, first2,
                        transform_reduce_binary_partition<Op1, Op2, T>{
                            op1, op2, init
                        });

                    return init;
                }

                // loop_step properly advances the iterators
                auto part_sum = util::loop_step<ExPolicy>(std::true_type(),
                    transform_reduce_binary_indirect<Op2>{op2}, first1, first2);

                std::pair<InIter1, InIter2> p = util::loop2<ExPolicy>(
                    std::true_type(), first1, last1, first2,
                    transform_reduce_binary_partition<
                            Op1, Op2, decltype(part_sum)
                        >{op1, op2, part_sum});

                // this is to support vectorization, it will call op1 for each
                // of the elements of a value-pack
                auto result =
                    util::detail::accumulate_values<ExPolicy>(
                        [&op1](T const& sum, T const& val)
                        {
                            return hpx::util::invoke(op1, sum, val);
                        },
                        std::move(part_sum),
                        std::move(init));

                // the vectorization might not cover all of the sequences,
                // handle the remainder directly
                if (p.first != last1)
                {
                    util::loop2<ExPolicy>(
                        std::false_type(), p.first, last1, p.second,
                        transform_reduce_binary_partition<
                                Op1, Op2, decltype(result)
                            >{op1, op2, result});
                }

                return util::detail::extract_value<ExPolicy>(result);
            }

            template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
                typename T_, typename Op1, typename Op2>
            static typename util::detail::algorithm_result<
                ExPolicy, T
            >::type
            parallel(ExPolicy && policy, FwdIter1 first1, FwdIter1 last1,
                 FwdIter2 first2, T_ && init, Op1 && op1, Op2 && op2)
            {
                typedef util::detail::algorithm_result<ExPolicy, T> result;
                typedef hpx::util::zip_iterator<FwdIter1, FwdIter2>
                    zip_iterator;
                typedef typename std::iterator_traits<FwdIter1>::difference_type
                    difference_type;

                if (first1 == last1)
                    return result::get(std::forward<T_>(init));

                difference_type count = std::distance(first1, last1);

                auto f1 =
                    [op1, op2, policy](
                        zip_iterator part_begin, std::size_t part_size
                    ) mutable -> T
                    {
                        HPX_UNUSED(policy);

                        auto iters = part_begin.get_iterator_tuple();
                        FwdIter1 it1 = hpx::util::get<0>(iters);
                        FwdIter2 it2 = hpx::util::get<1>(iters);

                        FwdIter1 last1 = it1;
                        std::advance(last1, part_size);

                        if (!util::loop_optimization<ExPolicy>(it1, last1))
                        {
                            // loop_step properly advances the iterators
                            auto result = util::loop_step<ExPolicy>(
                                std::false_type(),
                                transform_reduce_binary_indirect<Op2>{op2},
                                it1, it2);

                            util::loop2<ExPolicy>(
                                std::false_type(), it1, last1, it2,
                                transform_reduce_binary_partition<
                                        Op1, Op2, decltype(result)
                                    >{op1, op2, result});

                            return util::detail::extract_value<ExPolicy>(result);
                        }

                        // loop_step properly advances the iterators
                        auto part_sum = util::loop_step<ExPolicy>(
                            std::true_type(),
                            transform_reduce_binary_indirect<Op2>{op2}, it1, it2);

                        std::pair<FwdIter1, FwdIter2> p =
                            util::loop2<ExPolicy>(
                                std::true_type(), it1, last1, it2,
                                transform_reduce_binary_partition<
                                        Op1, Op2, decltype(part_sum)
                                    >{op1, op2, part_sum});

                        // this is to support vectorization, it will call op1
                        // for each of the elements of a value-pack
                        auto result = util::detail::accumulate_values<ExPolicy>(
                            [&op1](T const& sum, T const& val)
                            {
                                return hpx::util::invoke(op1, sum, val);
                            },
                            part_sum);

                        // the vectorization might not cover all of the sequences,
                        // handle the remainder directly
                        if (p.first != last1)
                        {
                            util::loop2<ExPolicy>(
                                std::false_type(), p.first, last1, p.second,
                                transform_reduce_binary_partition<
                                        Op1, Op2, decltype(result)
                                    >{op1, op2, result});
                        }

                        return util::detail::extract_value<ExPolicy>(result);
                    };

                using hpx::util::make_zip_iterator;
                return util::partitioner<ExPolicy, T>::call(
                    std::forward<ExPolicy>(policy),
                    make_zip_iterator(first1, first2), count,
                    std::move(f1),
                    [init, op1](std::vector<hpx::future<T> > && results) -> T
                    {
                        T ret = init;
                        for(auto && fut : results)
                        {
                            ret = hpx::util::invoke(op1, ret, fut.get());
                        }
                        return ret;
                    });
            }
        };
        /// \endcond
    }

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
    /// \tparam InIter1     The type of the first source iterators used
    ///                     (deduced). This iterator type must meet the
    ///                     requirements of an input iterator.
    /// \tparam InIter2     The type of the second source iterators used
    ///                     (deduced). This iterator type must meet the
    ///                     requirements of an input iterator.
    /// \tparam OutIter     The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     output iterator.
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
    ///           returns \a OutIter otherwise.
    ///
    template <typename ExPolicy, typename InIter1, typename InIter2, typename T,
    HPX_CONCEPT_REQUIRES_(
        execution::is_execution_policy<ExPolicy>::value &&
        hpx::traits::is_iterator<InIter1>::value &&
        hpx::traits::is_iterator<InIter2>::value)>
    typename util::detail::algorithm_result<ExPolicy, T>::type
    transform_reduce(ExPolicy && policy, InIter1 first1, InIter1 last1,
        InIter2 first2, T init)
    {
        static_assert(
            (hpx::traits::is_input_iterator<InIter1>::value),
            "Requires at least input iterator.");
        static_assert(
            (hpx::traits::is_input_iterator<InIter2>::value),
            "Requires at least input iterator.");

        typedef std::integral_constant<bool,
                execution::is_sequential_execution_policy<ExPolicy>::value ||
               !hpx::traits::is_forward_iterator<InIter1>::value ||
               !hpx::traits::is_forward_iterator<InIter2>::value
            > is_seq;

        return detail::transform_reduce_binary<T>().call(
            std::forward<ExPolicy>(policy), is_seq(), first1, last1, first2,
            std::move(init), detail::plus(), detail::multiplies());
    }

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
    /// \tparam InIter1     The type of the first source iterators used
    ///                     (deduced). This iterator type must meet the
    ///                     requirements of an input iterator.
    /// \tparam InIter2     The type of the second source iterators used
    ///                     (deduced). This iterator type must meet the
    ///                     requirements of an input iterator.
    /// \tparam OutIter     The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     output iterator.
    /// \tparam T           The type of the value to be used as return)
    ///                     values (deduced).
    /// \tparam Reduce      The type of the binary function object used for
    ///                     the multiplication operation.
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
    ///           returns \a OutIter otherwise.
    ///
    template <typename ExPolicy, typename InIter1, typename InIter2, typename T,
        typename Reduce, typename Convert,
    HPX_CONCEPT_REQUIRES_(
        execution::is_execution_policy<ExPolicy>::value &&
        hpx::traits::is_iterator<InIter1>::value &&
        hpx::traits::is_iterator<InIter2>::value &&
        hpx::traits::is_invocable<Convert,
                typename std::iterator_traits<InIter1>::value_type,
                typename std::iterator_traits<InIter2>::value_type
            >::value &&
        hpx::traits::is_invocable<Reduce,
                typename hpx::util::invoke_result<Convert,
                    typename std::iterator_traits<InIter1>::value_type,
                    typename std::iterator_traits<InIter2>::value_type
                >::type,
                typename hpx::util::invoke_result<Convert,
                    typename std::iterator_traits<InIter1>::value_type,
                    typename std::iterator_traits<InIter2>::value_type
                >::type
            >::value)>
    typename util::detail::algorithm_result<ExPolicy, T>::type
    transform_reduce(ExPolicy && policy, InIter1 first1, InIter1 last1,
        InIter2 first2, T init, Reduce && red_op, Convert && conv_op)
    {
        static_assert(
            (hpx::traits::is_input_iterator<InIter1>::value),
            "Requires at least input iterator.");
        static_assert(
            (hpx::traits::is_input_iterator<InIter2>::value),
            "Requires at least input iterator.");

        typedef std::integral_constant<bool,
                is_sequential_execution_policy<ExPolicy>::value ||
               !hpx::traits::is_forward_iterator<InIter1>::value ||
               !hpx::traits::is_forward_iterator<InIter2>::value
            > is_seq;

        return detail::transform_reduce_binary<T>().call(
            std::forward<ExPolicy>(policy), is_seq(), first1, last1, first2,
            std::move(init),
            std::forward<Reduce>(red_op), std::forward<Convert>(conv_op));
    }
}}}

#endif
