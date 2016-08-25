//  Copyright (c) 2015 Daniel Bourgeois
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/inner_product.hpp

#if !defined(HPX_PARALLEL_ALGORITHM_INNER_PRODUCT_JUL_15_2015_0730AM)
#define HPX_PARALLEL_ALGORITHM_INNER_PRODUCT_JUL_15_2015_0730AM

#include <hpx/config.hpp>
#include <hpx/traits/is_iterator.hpp>
#include <hpx/util/zip_iterator.hpp>

#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/loop.hpp>
#include <hpx/parallel/util/partitioner.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
    ///////////////////////////////////////////////////////////////////////////
    // inner_product
    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename T>
        struct inner_product
          : public detail::algorithm<inner_product<T>, T>
        {
            inner_product()
              : inner_product::algorithm("inner_product")
            {}

            template <typename ExPolicy, typename InIter1, typename InIter2,
                typename T_, typename Op1, typename Op2>
            static T
            sequential(ExPolicy, InIter1 first1, InIter1 last1, InIter2 first2,
                T_ && init, Op1 && op1, Op2 && op2)
            {
                return std::inner_product(
                    first1, last1, first2, std::forward<T_>(init),
                    std::forward<Op1>(op1), std::forward<Op2>(op2));
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

                using hpx::util::make_zip_iterator;
                return util::partitioner<ExPolicy, T>::call(
                    std::forward<ExPolicy>(policy),
                    make_zip_iterator(first1, first2), count,
                    [op1, op2](zip_iterator part_begin, std::size_t part_size) ->T
                    {
                        using hpx::util::get;
                        T part_sum = op2(
                            get<0>(*part_begin), get<1>(*part_begin));
                        ++part_begin;

                        // VS2015RC bails out when op is captured by ref
                        util::loop_n(part_begin, part_size - 1,
                            [=, &part_sum](zip_iterator it)
                            {
                                part_sum = op1(
                                    part_sum, op2(get<0>(*it), get<1>(*it)));
                            });
                        return part_sum;
                    },
                    [init, op1](std::vector<hpx::future<T> > && results) -> T
                    {
                        T ret = init;
                        for(auto && fut : results)
                        {
                            ret = op1(ret, fut.get());
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
    /// The operations in the parallel \a inner_product algorithm invoked
    /// with an execution policy object of type \a sequential_execution_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The operations in the parallel \a inner_product algorithm invoked
    /// with an execution policy object of type \a parallel_execution_policy
    /// or \a parallel_task_execution_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a inner_product algorithm returns a \a hpx::future<T> if
    ///           the execution policy is of type
    ///           \a sequential_task_execution_policy or
    ///           \a parallel_task_execution_policy and
    ///           returns \a OutIter otherwise.
    ///

    template <typename ExPolicy, typename InIter1, typename InIter2, typename T>
    inline typename std::enable_if<
        is_execution_policy<ExPolicy>::value,
        typename util::detail::algorithm_result<ExPolicy, T>::type
    >::type
    inner_product(ExPolicy&& policy, InIter1 first1, InIter1 last1,
        InIter2 first2, T init)
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

        return detail::inner_product<T>().call(
            std::forward<ExPolicy>(policy), is_seq(), first1, last1, first2,
            std::move(init), std::plus<T>(), std::multiplies<T>());
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
    /// \tparam Op1         The type of the binary function object used for
    ///                     the summation operation.
    /// \tparam Op2         The type of the binary function object used for
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
    /// \param op1          Specifies the function (or function object) which
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
    ///                     argument type of \a op2.
    /// \param op2          Specifies the function (or function object) which
    ///                     will be invoked for the initial value and each
    ///                     of the return values of \a op1.
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
    ///
    /// The operations in the parallel \a inner_product algorithm invoked
    /// with an execution policy object of type \a sequential_execution_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The operations in the parallel \a inner_product algorithm invoked
    /// with an execution policy object of type \a parallel_execution_policy
    /// or \a parallel_task_execution_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a inner_product algorithm returns a \a hpx::future<T> if
    ///           the execution policy is of type
    ///           \a sequential_task_execution_policy or
    ///           \a parallel_task_execution_policy and
    ///           returns \a OutIter otherwise.
    ///

    template <typename ExPolicy, typename InIter1, typename InIter2, typename T,
        typename Op1, typename Op2>
    inline typename std::enable_if<
        is_execution_policy<ExPolicy>::value,
        typename util::detail::algorithm_result<ExPolicy, T>::type
    >::type
    inner_product(ExPolicy&& policy, InIter1 first1, InIter1 last1,
        InIter2 first2, T init, Op1 && op1, Op2 && op2)
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

        return detail::inner_product<T>().call(
            std::forward<ExPolicy>(policy), is_seq(), first1, last1, first2,
            std::move(init), std::forward<Op1>(op1), std::forward<Op2>(op2));
    }
}}}

#endif
