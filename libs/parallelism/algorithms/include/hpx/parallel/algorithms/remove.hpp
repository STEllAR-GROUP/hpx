//  Copyright (c) 2017 Taeguk Kwon
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/remove.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/functional/invoke.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/parallel/util/tagged_pair.hpp>
#include <hpx/type_support/unused.hpp>

#include <hpx/algorithms/traits/projected.hpp>
#include <hpx/execution/algorithms/detail/is_negative.hpp>
#include <hpx/execution/algorithms/detail/predicates.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/detail/transfer.hpp>
#include <hpx/parallel/tagspec.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/foreach_partitioner.hpp>
#include <hpx/parallel/util/invoke_projected.hpp>
#include <hpx/parallel/util/loop.hpp>
#include <hpx/parallel/util/projection_identity.hpp>
#include <hpx/parallel/util/scan_partitioner.hpp>
#include <hpx/parallel/util/transfer.hpp>
#include <hpx/parallel/util/zip_iterator.hpp>

#if !defined(HPX_HAVE_CXX17_SHARED_PTR_ARRAY)
#include <boost/shared_array.hpp>
#endif

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <iterator>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace parallel { inline namespace v1 {
    /////////////////////////////////////////////////////////////////////////////
    // remove_if
    namespace detail {
        /// \cond NOINTERNAL

        // sequential remove_if with projection function
        template <typename FwdIter, typename Pred, typename Proj>
        FwdIter sequential_remove_if(
            FwdIter first, FwdIter last, Pred&& pred, Proj&& proj)
        {
            return std::remove_if(first, last,
                util::invoke_projected<Pred, Proj>(
                    std::forward<Pred>(pred), std::forward<Proj>(proj)));
        }

        template <typename FwdIter>
        struct remove_if : public detail::algorithm<remove_if<FwdIter>, FwdIter>
        {
            remove_if()
              : remove_if::algorithm("remove_if")
            {
            }

            template <typename ExPolicy, typename Pred, typename Proj>
            static FwdIter sequential(
                ExPolicy, FwdIter first, FwdIter last, Pred&& pred, Proj&& proj)
            {
                return sequential_remove_if(first, last,
                    std::forward<Pred>(pred), std::forward<Proj>(proj));
            }

            template <typename ExPolicy, typename Pred, typename Proj>
            static
                typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
                parallel(ExPolicy&& policy, FwdIter first, FwdIter last,
                    Pred&& pred, Proj&& proj)
            {
                typedef hpx::util::zip_iterator<FwdIter, bool*> zip_iterator;
                typedef util::detail::algorithm_result<ExPolicy, FwdIter>
                    algorithm_result;
                typedef typename std::iterator_traits<FwdIter>::difference_type
                    difference_type;

                difference_type count = std::distance(first, last);

                if (count == 0)
                    return algorithm_result::get(std::move(last));

#if defined(HPX_HAVE_CXX17_SHARED_PTR_ARRAY)
                std::shared_ptr<bool[]> flags(new bool[count]);
#else
                boost::shared_array<bool> flags(new bool[count]);
#endif
                std::size_t init = 0u;

                using hpx::get;
                using hpx::util::make_zip_iterator;
                typedef util::scan_partitioner<ExPolicy, FwdIter, std::size_t,
                    void, util::scan_partitioner_sequential_f3_tag>
                    scan_partitioner_type;

                auto f1 = [pred = std::forward<Pred>(pred),
                              proj = std::forward<Proj>(proj)](
                              zip_iterator part_begin,
                              std::size_t part_size) -> std::size_t {
                    // MSVC complains if pred or proj is captured by ref below
                    util::loop_n<ExPolicy>(part_begin, part_size,
                        [pred, proj](zip_iterator it) mutable {
                            bool f = hpx::util::invoke(
                                pred, hpx::util::invoke(proj, get<0>(*it)));

                            get<1>(*it) = f;
                        });

                    // There is no need to return the partition result.
                    // But, the scan_partitioner doesn't support 'void' as
                    // Result1. So, unavoidably return non-meaning value.
                    return 0u;
                };

                auto f2 = hpx::util::unwrapping(
                    [](std::size_t, std::size_t) -> std::size_t {
                        // There is no need to propagate the partition
                        // results. But, the scan_partitioner doesn't
                        // support 'void' as Result1. So, unavoidably
                        // return non-meaning value.
                        return 0u;
                    });

                std::shared_ptr<FwdIter> dest_ptr =
                    std::make_shared<FwdIter>(first);
                auto f3 =
                    [dest_ptr, flags](zip_iterator part_begin,
                        std::size_t part_size,
                        hpx::shared_future<std::size_t> curr,
                        hpx::shared_future<std::size_t> next) mutable -> void {
                    HPX_UNUSED(flags);

                    curr.get();    // rethrow exceptions
                    next.get();    // rethrow exceptions

                    FwdIter& dest = *dest_ptr;

                    if (dest == get<0>(part_begin.get_iterator_tuple()))
                    {
                        // Self-assignment must be detected.
                        util::loop_n<ExPolicy>(
                            part_begin, part_size, [&dest](zip_iterator it) {
                                if (!get<1>(*it))
                                {
                                    if (dest != get<0>(it.get_iterator_tuple()))
                                        *dest++ = std::move(get<0>(*it));
                                    else
                                        ++dest;
                                }
                            });
                    }
                    else
                    {
                        // Self-assignment can't be performed.
                        util::loop_n<ExPolicy>(
                            part_begin, part_size, [&dest](zip_iterator it) {
                                if (!get<1>(*it))
                                    *dest++ = std::move(get<0>(*it));
                            });
                    }
                };

                auto f4 =
                    [dest_ptr, flags](
                        std::vector<hpx::shared_future<std::size_t>>&&,
                        std::vector<hpx::future<void>>&&) mutable -> FwdIter {
                    HPX_UNUSED(flags);
                    return *dest_ptr;
                };

                return scan_partitioner_type::call(
                    std::forward<ExPolicy>(policy),
                    make_zip_iterator(first, flags.get()), count, init,
                    // step 1 performs first part of scan algorithm
                    std::move(f1),
                    // step 2 propagates the partition results from left
                    // to right
                    std::move(f2),
                    // step 3 runs final accumulation on each partition
                    std::move(f3),
                    // step 4 use this return value
                    std::move(f4));
            }
        };
        /// \endcond
    }    // namespace detail

    /// Removes all elements satisfying specific criteria from the range
    /// [first, last) and returns a past-the-end iterator for the new
    /// end of the range. This version removes all elements for which predicate
    /// \a pred returns true.
    ///
    /// \note   Complexity: Performs not more than \a last - \a first
    ///         assignments, exactly \a last - \a first applications of
    ///         the predicate \a pred and the projection \a proj.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter     The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Pred        The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a remove_if requires \a Pred to meet the
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
    /// \param pred         Specifies the function (or function object) which
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
    ///                     type \a FwdIter can be dereferenced and then
    ///                     implicitly converted to Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a remove_if algorithm invoked with
    /// an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a remove_if algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a remove_if algorithm returns a \a hpx::future<FwdIter>
    ///           if the execution policy is of type
    ///           \a sequenced_task_policy or \a parallel_task_policy and
    ///           returns \a FwdIter otherwise.
    ///           The \a remove_if algorithm returns the iterator to the new end
    ///           of the range.
    ///
    template <typename ExPolicy, typename FwdIter, typename Pred,
        typename Proj = util::projection_identity,
        HPX_CONCEPT_REQUIRES_(hpx::is_execution_policy<ExPolicy>::value&& hpx::
                traits::is_iterator<FwdIter>::value&& traits::is_projected<Proj,
                    FwdIter>::value&& traits::is_indirect_callable<ExPolicy,
                    Pred, traits::projected<Proj, FwdIter>>::value)>
    typename util::detail::algorithm_result<ExPolicy, FwdIter>::type remove_if(
        ExPolicy&& policy, FwdIter first, FwdIter last, Pred&& pred,
        Proj&& proj = Proj())
    {
        static_assert((hpx::traits::is_forward_iterator<FwdIter>::value),
            "Required at least forward iterator.");

        typedef execution::is_sequenced_execution_policy<ExPolicy> is_seq;

        return detail::remove_if<FwdIter>().call(std::forward<ExPolicy>(policy),
            is_seq(), first, last, std::forward<Pred>(pred),
            std::forward<Proj>(proj));
    }

    /////////////////////////////////////////////////////////////////////////////
    // remove
    /// Removes all elements satisfying specific criteria from the range
    /// [first, last) and returns a past-the-end iterator for the new
    /// end of the range. This version removes all elements that are
    /// equal to \a value.
    ///
    /// \note   Complexity: Performs not more than \a last - \a first
    ///         assignments, exactly \a last - \a first applications of
    ///         the operator==() and the projection \a proj.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter     The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam T           The type of the value to remove (deduced).
    ///                     This value type must meet the requirements of
    ///                     \a CopyConstructible.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param value        Specifies the value of elements to remove.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a remove algorithm invoked with
    /// an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a remove algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a remove algorithm returns a \a hpx::future<FwdIter>
    ///           if the execution policy is of type
    ///           \a sequenced_task_policy or \a parallel_task_policy and
    ///           returns \a FwdIter otherwise.
    ///           The \a remove algorithm returns the iterator to the new end
    ///           of the range.
    ///
    template <typename ExPolicy, typename FwdIter, typename T,
        typename Proj = util::projection_identity,
        HPX_CONCEPT_REQUIRES_(hpx::is_execution_policy<ExPolicy>::value&&
                hpx::traits::is_iterator<FwdIter>::value&&
                    traits::is_projected<Proj, FwdIter>::value)>
    typename util::detail::algorithm_result<ExPolicy, FwdIter>::type remove(
        ExPolicy&& policy, FwdIter first, FwdIter last, T const& value,
        Proj&& proj = Proj())
    {
        typedef typename std::iterator_traits<FwdIter>::value_type Type;

        // Just utilize existing parallel remove_if.
        return remove_if(
            std::forward<ExPolicy>(policy), first, last,
            [value](Type const& a) -> bool { return value == a; },
            std::forward<Proj>(proj));
    }
}}}    // namespace hpx::parallel::v1
