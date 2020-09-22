//  Copyright (c) 2015-2017 Francisco Jose Tapia
//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/async_local/dataflow.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/functional/invoke.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/type_support/decay.hpp>

#include <hpx/algorithms/traits/projected.hpp>
#include <hpx/execution/algorithms/detail/predicates.hpp>
#include <hpx/execution/executors/execution.hpp>
#include <hpx/execution/executors/execution_information.hpp>
#include <hpx/execution/executors/execution_parameters.hpp>
#include <hpx/executors/exception_list.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/detail/parallel_stable_sort.hpp>
#include <hpx/parallel/algorithms/detail/spin_sort.hpp>
#include <hpx/parallel/util/compare_projected.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/detail/chunk_size.hpp>
#include <hpx/parallel/util/projection_identity.hpp>

#include <algorithm>
#include <cstddef>
#include <exception>
#include <functional>
#include <iterator>
#include <list>
#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { inline namespace v1 {
    ///////////////////////////////////////////////////////////////////////////
    // stable_sort
    namespace detail {
        /// \cond NOINTERNAL

        ///////////////////////////////////////////////////////////////////////
        // stable_sort
        template <typename RandomIt>
        struct stable_sort
          : public detail::algorithm<stable_sort<RandomIt>, RandomIt>
        {
            stable_sort()
              : stable_sort::algorithm("stable_sort")
            {
            }

            template <typename ExPolicy, typename Sentinel, typename Compare,
                typename Proj>
            static RandomIt sequential(ExPolicy, RandomIt first, Sentinel last,
                Compare&& comp, Proj&& proj)
            {
                using compare_type =
                    util::compare_projected<typename std::decay<Compare>::type,
                        typename std::decay<Proj>::type>;

                spin_sort(first, last,
                    compare_type(
                        std::forward<Compare>(comp), std::forward<Proj>(proj)));
                return last;
            }

            template <typename ExPolicy, typename Sentinel, typename Compare,
                typename Proj>
            static typename util::detail::algorithm_result<ExPolicy,
                RandomIt>::type
            parallel(ExPolicy&& policy, RandomIt first, Sentinel last,
                Compare&& compare, Proj&& proj)
            {
                using algorithm_result =
                    util::detail::algorithm_result<ExPolicy, RandomIt>;
                using compare_type =
                    util::compare_projected<typename std::decay<Compare>::type,
                        typename std::decay<Proj>::type>;

                // number of elements to sort
                std::size_t count = last - first;

                // figure out the chunk size to use
                std::size_t cores = execution::processing_units_count(
                    policy.parameters(), policy.executor());

                std::size_t max_chunks = execution::maximal_number_of_chunks(
                    policy.parameters(), policy.executor(), cores, count);

                std::size_t chunk_size = execution::get_chunk_size(
                    policy.parameters(), policy.executor(),
                    [](std::size_t) { return 0; }, cores, count);

                util::detail::adjust_chunk_size_and_max_chunks(
                    cores, count, max_chunks, chunk_size);

                // we should not get smaller than our sort_limit_per_task
                chunk_size = (std::max)(chunk_size, stable_sort_limit_per_task);

                try
                {
                    // call the sort routine and return the right type,
                    // depending on execution policy
                    compare_type comp(std::forward<Compare>(compare),
                        std::forward<Proj>(proj));

                    return algorithm_result::get(
                        parallel_stable_sort(policy.executor(), first, last,
                            cores, chunk_size, std::move(comp)));
                }
                catch (...)
                {
                    return algorithm_result::get(
                        detail::handle_exception<ExPolicy, RandomIt>::call(
                            std::current_exception()));
                }
            }
        };
        /// \endcond
    }    // namespace detail

    //-----------------------------------------------------------------------------
    /// Sorts the elements in the range [first, last) in ascending order. The
    /// relative order of equal elements is preserved. The function
    /// uses the given comparison function object comp (defaults to using
    /// operator<()).
    ///
    /// \note   Complexity: O(Nlog(N)), where N = std::distance(first, last)
    ///                     comparisons.
    ///
    /// A sequence is sorted with respect to a comparator \a comp and a
    /// projection \a proj if for every iterator i pointing to the sequence and
    /// every non-negative integer n such that i + n is a valid iterator
    /// pointing to an element of the sequence, and
    /// INVOKE(comp, INVOKE(proj, *(i + n)), INVOKE(proj, *i)) == false.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it applies user-provided function objects.
    /// \tparam RandomIt    The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     random access iterator.
    /// \tparam Sentinel    The type of the end iterators used (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     random access iterator and must be a valid sentinel
    ///                     type for RandomIt.
    /// \tparam Comp        The type of the function/function object to use
    ///                     (deduced).
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param comp         comp is a callable object. The return value of the
    ///                     INVOKE operation applied to an object of type Comp,
    ///                     when contextually converted to bool, yields true if
    ///                     the first argument of the call is less than the
    ///                     second, and false otherwise. It is assumed that comp
    ///                     will not apply any non-constant function through the
    ///                     dereferenced iterator.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each pair of elements as a
    ///                     projection operation before the actual predicate
    ///                     \a comp is invoked.
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
    /// \returns  The \a stable_sort algorithm returns a
    ///           \a hpx::future<RandomIt> if the execution policy is of
    ///           type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and returns \a RandomIt
    ///           otherwise.
    ///           The algorithm returns an iterator pointing to the first
    ///           element after the last element in the input sequence.
    //-----------------------------------------------------------------------------
    // clang-format off
    template <typename ExPolicy, typename RandomIt, typename Sentinel,
        typename Proj = util::projection_identity,
        typename Compare = detail::less,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy<ExPolicy>::value &&
            hpx::traits::is_iterator<RandomIt>::value &&
            hpx::traits::is_sentinel_for<Sentinel, RandomIt>::value &&
            traits::is_projected<Proj, RandomIt>::value &&
            traits::is_indirect_callable<ExPolicy, Compare,
                traits::projected<Proj, RandomIt>,
                traits::projected<Proj, RandomIt>
            >::value
        )>
    // clang-format on
    typename util::detail::algorithm_result<ExPolicy, RandomIt>::type
    stable_sort(ExPolicy&& policy, RandomIt first, Sentinel last,
        Compare&& comp = Compare(), Proj&& proj = Proj())
    {
        static_assert((hpx::traits::is_random_access_iterator<RandomIt>::value),
            "Requires a random access iterator.");

        typedef hpx::is_sequenced_execution_policy<ExPolicy> is_seq;

        return detail::stable_sort<RandomIt>().call(
            std::forward<ExPolicy>(policy), is_seq(), first, last,
            std::forward<Compare>(comp), std::forward<Proj>(proj));
    }
}}}    // namespace hpx::parallel::v1
