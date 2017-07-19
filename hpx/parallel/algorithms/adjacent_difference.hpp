//  Copyright (c) 2015 Daniel Bourgeois
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/adjacent_difference.hpp

#if !defined(HPX_PARALLEL_ALGORITHM_ADJACENT_DIF_JUL_15)
#define HPX_PARALLEL_ALGORITHM_ADJACENT_DIF_JUL_15

#include <hpx/config.hpp>
#include <hpx/traits/is_iterator.hpp>
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
    // adjacent_difference
    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename Iter>
        struct adjacent_difference
          : public detail::algorithm<adjacent_difference<Iter>, Iter>
        {
            adjacent_difference()
              : adjacent_difference::algorithm("adjacent_difference")
            {}

            template <typename ExPolicy, typename InIter, typename Op>
            static Iter
            sequential(ExPolicy, InIter first, InIter last, Iter dest,
                Op && op)
            {
                return std::adjacent_difference(
                    first, last, dest, std::forward<Op>(op));
            }

            template <typename ExPolicy, typename FwdIter, typename Op>
            static typename util::detail::algorithm_result<
                ExPolicy, Iter
            >::type
            parallel(ExPolicy && policy, FwdIter first, FwdIter last,
                Iter dest, Op && op)
            {
                typedef hpx::util::zip_iterator<FwdIter, FwdIter, Iter>
                    zip_iterator;
                typedef util::detail::algorithm_result<ExPolicy, Iter> result;
                typedef typename std::iterator_traits<FwdIter>::difference_type
                    difference_type;

                if (first == last)
                    return result::get(std::move(dest));

                difference_type count = std::distance(first, last) - 1;

                FwdIter prev = first;
                *dest++ = *first++;

                if (count == 0)
                {
                    return result::get(std::move(dest));
                }

                auto f1 =
                    [op, policy](
                        zip_iterator part_begin, std::size_t part_size
                    ) mutable
                    {
                        HPX_UNUSED(policy);

                        // VS2015RC bails out when op is captured by ref
                        using hpx::util::get;
                        util::loop_n<ExPolicy>(
                            part_begin, part_size,
                            [op](zip_iterator it)
                            {
                                get<2>(*it) = hpx::util::invoke(
                                    op, get<0>(*it), get<1>(*it));
                            });
                    };

                using hpx::util::make_zip_iterator;
                return util::partitioner<ExPolicy, Iter, void>::call(
                    std::forward<ExPolicy>(policy),
                    make_zip_iterator(first, prev, dest), count,
                    std::move(f1),
                    [dest, count](std::vector<hpx::future<void> > &&)
                        mutable -> Iter
                    {
                        std::advance(dest, count);
                        return dest;
                    });
            }
        };
        /// \endcond
    }
    ////////////////////////////////////////////////////////////////////////////
    /// Assigns each value in the range given by result its corresponding
    /// element in the range [first, last] and the one preceding it except
    /// *result, which is assigned *first
    ///
    /// \note   Complexity: Exactly (last - first) - 1 application of the
    ///                     binary operator and (last - first) assignments.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter1    The type of the source iterators used for the
    ///                     input range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam FwdIter2    The type of the source iterators used for the
    ///                     output range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     of the range the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements of
    ///                     the range the algorithm will be applied to.
    /// \param dest         Refers to the beginning of the sequence of elements
    ///                     the results will be assigned to.
    ///
    /// The difference operations in the parallel \a adjacent_difference invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The difference operations in the parallel \a adjacent_difference invoked
    /// with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an
    /// unordered fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a adjacent_difference algorithm returns a
    ///           \a hpx::future<FwdIter2> if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a FwdIter2 otherwise.
    ///           The \a adjacent_find algorithm returns an iterator to the
    ///           last element in the output range.
    ///
    ///           This overload of \a adjacent_find is available if the user
    ///           decides to provide their algorithm their own binary
    ///           predicate \a op.
    ///

    template <typename ExPolicy, typename FwdIter1, typename FwdIter2>
    inline typename std::enable_if<
        execution::is_execution_policy<ExPolicy>::value,
        typename util::detail::algorithm_result<ExPolicy, FwdIter2>::type
    >::type
    adjacent_difference(ExPolicy&& policy, FwdIter1 first, FwdIter1 last,
        FwdIter2 dest)
    {
        typedef typename std::iterator_traits<FwdIter1>::value_type value_type;

#if defined(HPX_HAVE_ALGORITHM_INPUT_ITERATOR_SUPPORT)
        static_assert(
            (hpx::traits::is_input_iterator<FwdIter1>::value),
            "Requires at least input iterator.");
        static_assert(
            (hpx::traits::is_output_iterator<FwdIter2>::value ||
                hpx::traits::is_forward_iterator<FwdIter2>::value),
            "Requires at least output iterator.");

        typedef std::integral_constant<bool,
                execution::is_sequenced_execution_policy<ExPolicy>::value ||
               !hpx::traits::is_forward_iterator<FwdIter2>::value ||
               !hpx::traits::is_forward_iterator<FwdIter1>::value
            > is_seq;
#else
        static_assert(
            (hpx::traits::is_forward_iterator<FwdIter1>::value),
            "Requires at least forward iterator.");
        static_assert(
            (hpx::traits::is_forward_iterator<FwdIter2>::value),
            "Requires at least forward iterator.");

        typedef execution::is_sequenced_execution_policy<ExPolicy> is_seq;
#endif

        return detail::adjacent_difference<FwdIter2>().call(
            std::forward<ExPolicy>(policy), is_seq(), first, last, dest,
            std::minus<value_type>());
    }

    ////////////////////////////////////////////////////////////////////////////
    /// Assigns each value in the range given by result its corresponding
    /// element in the range [first, last] and the one preceding it except
    /// *result, which is assigned *first
    ///
    /// \note   Complexity: Exactly (last - first) - 1 application of the
    ///                     binary operator and (last - first) assignments.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter1    The type of the source iterators used for the
    ///                     input range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam FwdIter2    The type of the source iterators used for the
    ///                     output range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Op          The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a adjacent_difference requires \a Op
    ///                     to meet the requirements of \a CopyConstructible.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     of the range the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements of
    ///                     the range the algorithm will be applied to.
    /// \param dest         Refers to the beginning of the sequence of elements
    ///                     the results will be assigned to.
    /// \param op           The binary operator which returns the difference
    ///                     of elements. The signature should be equivalent
    ///                     to the following:
    ///                     \code
    ///                     bool op(const Type1 &a, const Type1 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The types \a Type1  must be such
    ///                     that objects of type \a FwdIter1 can be dereferenced
    ///                     and then implicitly converted to the dereferenced
    ///                     type of \a dest.
    ///
    /// The difference operations in the parallel \a adjacent_difference invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The difference operations in the parallel \a adjacent_difference invoked
    /// with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an
    /// unordered fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a adjacent_difference algorithm returns a
    ///           \a hpx::future<FwdIter2> if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a FwdIter2 otherwise.
    ///           The \a adjacent_find algorithm returns an iterator to the
    ///           last element in the output range.
    ///
    ///
    template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
        typename Op>
    inline typename std::enable_if<
        execution::is_execution_policy<ExPolicy>::value,
        typename util::detail::algorithm_result<ExPolicy, FwdIter2>::type
    >::type
    adjacent_difference(ExPolicy&& policy, FwdIter1 first, FwdIter1 last,
        FwdIter2 dest, Op && op)
    {
#if defined(HPX_HAVE_ALGORITHM_INPUT_ITERATOR_SUPPORT)
        static_assert(
            (hpx::traits::is_input_iterator<FwdIter1>::value),
            "Requires at least input iterator.");
        static_assert(
            (hpx::traits::is_output_iterator<FwdIter2>::value ||
                hpx::traits::is_forward_iterator<FwdIter2>::value),
            "Requires at least output iterator.");

        typedef std::integral_constant<bool,
                execution::is_sequenced_execution_policy<ExPolicy>::value ||
               !hpx::traits::is_forward_iterator<FwdIter2>::value ||
               !hpx::traits::is_forward_iterator<FwdIter1>::value
            > is_seq;
#else
        static_assert(
            (hpx::traits::is_forward_iterator<FwdIter1>::value),
            "Requires at least forward iterator.");
        static_assert(
            (hpx::traits::is_forward_iterator<FwdIter2>::value),
            "Requires at least forward iterator.");

        typedef execution::is_sequenced_execution_policy<ExPolicy> is_seq;
#endif

        return detail::adjacent_difference<FwdIter2>().call(
            std::forward<ExPolicy>(policy), is_seq(), first, last, dest,
            std::forward<Op>(op));
    }
}}}

#endif
