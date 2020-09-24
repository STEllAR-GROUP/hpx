//  Copyright (c) 2015 John Biddiscombe
//  Copyright (c) 2015-2020 Hartmut Kaiser
//  Copyright (c) 2015-2019 Francisco Jose Tapia
//  Copyright (c) 2018 Taeguk Kwon
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
#include <hpx/parallel/algorithms/detail/is_sorted.hpp>
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
    // sort
    namespace detail {
        /// \cond NOINTERNAL
        static const std::size_t sort_limit_per_task = 65536ul;

        ///////////////////////////////////////////////////////////////////////
        // std::is_sorted is not available on all supported platforms yet
        template <typename Iter, typename Compare>
        inline bool is_sorted_sequential(
            Iter first, Iter last, Compare const& comp)
        {
            bool sorted = true;
            if (first != last)
            {
                for (Iter it1 = first, it2 = first + 1;
                     it2 != last && (sorted = !comp(*it2, *it1)); it1 = it2++)
                {
                    /**/
                }
            }
            return sorted;
        }

        /// Return the iterator to the mid value of the three values
        /// passed as parameters
        ///
        /// \param iter_1 : iterator to the first value
        /// \param iter_2 : iterator to the second value
        /// \param iter_3 : iterator to the third value
        /// \param comp : object for to compare two values
        /// \return iterator to mid value
        template <typename Iter, typename Compare>
        inline Iter mid3(Iter iter_1, Iter iter_2, Iter iter_3, Compare comp)
        {
            return comp(*iter_1, *iter_2) ?
                (comp(*iter_2, *iter_3) ?
                        iter_2 :
                        (comp(*iter_1, *iter_3) ? iter_3 : iter_1)) :
                (comp(*iter_3, *iter_2) ?
                        iter_2 :
                        (comp(*iter_3, *iter_1) ? iter_3 : iter_1));
        }

        /// Return the iterator to the mid value of the nine values
        /// passed as parameters
        //
        /// \param iter_1   iterator to the first value
        /// \param iter_2   iterator to the second value
        /// \param iter_3   iterator to the third value
        /// \param iter_4   iterator to the fourth value
        /// \param iter_5   iterator to the fifth value
        /// \param iter_6   iterator to the sixth value
        /// \param iter_7   iterator to the seventh value
        /// \param iter_8   iterator to the eighth value
        /// \param iter_9   iterator to the ninth value
        /// \return iterator to the mid value
        template <typename Iter, typename Compare>
        inline Iter mid9(Iter iter_1, Iter iter_2, Iter iter_3, Iter iter_4,
            Iter iter_5, Iter iter_6, Iter iter_7, Iter iter_8, Iter iter_9,
            Compare const& comp)
        {
            return mid3(mid3(iter_1, iter_2, iter_3, comp),
                mid3(iter_4, iter_5, iter_6, comp),
                mid3(iter_7, iter_8, iter_9, comp), comp);
        }

        /// Receive a range between first and last, obtain 9 values
        /// between the elements  including the first and the previous
        /// to the last. Obtain the iterator to the mid value and swap
        /// with the first position
        //
        /// \param first    iterator to the first element
        /// \param last     iterator to the last element
        /// \param comp     object for to compare two elements
        template <typename Iter, typename Compare>
        inline void pivot9(Iter first, Iter last, Compare const& comp)
        {
            std::size_t chunk = (last - first) >> 3;
            Iter itaux = mid9(first + 1, first + chunk, first + 2 * chunk,
                first + 3 * chunk, first + 4 * chunk, first + 5 * chunk,
                first + 6 * chunk, first + 7 * chunk, last - 1, comp);
            std::iter_swap(first, itaux);
        }

        /// \brief this function is the work assigned to each thread in the
        ///        parallel process
        /// \exception
        /// \return
        /// \remarks
        template <typename ExPolicy, typename RandomIt, typename Compare>
        hpx::future<RandomIt> sort_thread(ExPolicy policy, RandomIt first,
            RandomIt last, Compare comp, std::size_t chunk_size)
        {
            std::ptrdiff_t N = last - first;
            if (std::size_t(N) <= chunk_size)
            {
                return execution::async_execute(policy.executor(),
                    [first, last, comp = std::move(comp)]() -> RandomIt {
                        std::sort(first, last, comp);
                        return last;
                    });
            }

            // check if sorted
            if (detail::is_sorted_sequential(first, last, comp))
                return hpx::make_ready_future(last);

            // pivot select
            pivot9(first, last, comp);

            using reference =
                typename std::iterator_traits<RandomIt>::reference;

            reference val = *first;
            RandomIt c_first = first + 1, c_last = last - 1;

            while (comp(*c_first, val))
                ++c_first;
            while (comp(val, *c_last))
                --c_last;
            while (c_first < c_last)
            {
                std::iter_swap(c_first++, c_last--);
                while (comp(*c_first, val))
                    ++c_first;
                while (comp(val, *c_last))
                    --c_last;
            }

            std::iter_swap(first, c_last);

            // spawn tasks for each sub section
            hpx::future<RandomIt> left = execution::async_execute(
                policy.executor(), &sort_thread<ExPolicy, RandomIt, Compare>,
                policy, first, c_last, comp, chunk_size);

            hpx::future<RandomIt> right = execution::async_execute(
                policy.executor(), &sort_thread<ExPolicy, RandomIt, Compare>,
                policy, c_first, last, comp, chunk_size);

            return hpx::dataflow(
                [last](hpx::future<RandomIt>&& left,
                    hpx::future<RandomIt>&& right) -> RandomIt {
                    if (left.has_exception() || right.has_exception())
                    {
                        std::list<std::exception_ptr> errors;
                        if (left.has_exception())
                            errors.push_back(left.get_exception_ptr());
                        if (right.has_exception())
                            errors.push_back(right.get_exception_ptr());

                        throw exception_list(std::move(errors));
                    }
                    return last;
                },
                std::move(left), std::move(right));
        }

        /// \param [in] first   iterator to the first element to sort
        /// \param [in] last    iterator to the next element after the last
        /// \param [in] comp    object for to compare
        /// \exception
        /// \return
        /// \remarks
        template <typename ExPolicy, typename RandomIt, typename Compare>
        hpx::future<RandomIt> parallel_sort_async(
            ExPolicy&& policy, RandomIt first, RandomIt last, Compare comp)
        {
            // number of elements to sort
            std::size_t count = last - first;

            // figure out the chunk size to use
            std::size_t const cores = execution::processing_units_count(
                policy.parameters(), policy.executor());

            std::size_t max_chunks = execution::maximal_number_of_chunks(
                policy.parameters(), policy.executor(), cores, count);

            std::size_t chunk_size = execution::get_chunk_size(
                policy.parameters(), policy.executor(),
                [](std::size_t) { return 0; }, cores, count);

            util::detail::adjust_chunk_size_and_max_chunks(
                cores, count, max_chunks, chunk_size);

            // we should not get smaller than our sort_limit_per_task
            chunk_size = (std::max)(chunk_size, sort_limit_per_task);

            std::ptrdiff_t N = last - first;
            HPX_ASSERT(N >= 0);

            if (std::size_t(N) < chunk_size)
            {
                std::sort(first, last, comp);
                return hpx::make_ready_future(last);
            }

            // check if already sorted
            if (detail::is_sorted_sequential(first, last, comp))
                return hpx::make_ready_future(last);

            return execution::async_execute(policy.executor(),
                &sort_thread<typename std::decay<ExPolicy>::type, RandomIt,
                    Compare>,
                std::forward<ExPolicy>(policy), first, last, comp, chunk_size);
        }

        ///////////////////////////////////////////////////////////////////////
        // sort
        template <typename RandomIt>
        struct sort : public detail::algorithm<sort<RandomIt>, RandomIt>
        {
            sort()
              : sort::algorithm("sort")
            {
            }

            template <typename ExPolicy, typename Compare, typename Proj>
            static RandomIt sequential(ExPolicy, RandomIt first, RandomIt last,
                Compare&& comp, Proj&& proj)
            {
                std::sort(first, last,
                    util::compare_projected<Compare, Proj>(
                        std::forward<Compare>(comp), std::forward<Proj>(proj)));
                return last;
            }

            template <typename ExPolicy, typename Compare, typename Proj>
            static typename util::detail::algorithm_result<ExPolicy,
                RandomIt>::type
            parallel(ExPolicy&& policy, RandomIt first, RandomIt last,
                Compare&& comp, Proj&& proj)
            {
                typedef util::detail::algorithm_result<ExPolicy, RandomIt>
                    algorithm_result;

                try
                {
                    // call the sort routine and return the right type,
                    // depending on execution policy
                    return algorithm_result::get(parallel_sort_async(
                        std::forward<ExPolicy>(policy), first, last,
                        util::compare_projected<Compare, Proj>(
                            std::forward<Compare>(comp),
                            std::forward<Proj>(proj))));
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
    /// order of equal elements is not guaranteed to be preserved. The function
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
    /// \tparam Iter        The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     random access iterator.
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
    /// \returns  The \a sort algorithm returns a
    ///           \a hpx::future<RandomIt> if the execution policy is of
    ///           type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and returns \a RandomIt
    ///           otherwise.
    ///           The algorithm returns an iterator pointing to the first
    ///           element after the last element in the input sequence.
    // clang-format off
    template <typename ExPolicy, typename RandomIt,
        typename Compare = detail::less,
        typename Proj = util::projection_identity,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy<ExPolicy>::value &&
            hpx::traits::is_iterator<RandomIt>::value &&
            traits::is_projected<Proj, RandomIt>::value &&
            traits::is_indirect_callable<ExPolicy, Compare,
                traits::projected<Proj, RandomIt>,
                traits::projected<Proj, RandomIt>
            >::value
        )>
    // clang-format on
    typename util::detail::algorithm_result<ExPolicy, RandomIt>::type sort(
        ExPolicy&& policy, RandomIt first, RandomIt last,
        Compare&& comp = Compare(), Proj&& proj = Proj())
    {
        static_assert((hpx::traits::is_random_access_iterator<RandomIt>::value),
            "Requires a random access iterator.");

        typedef execution::is_sequenced_execution_policy<ExPolicy> is_seq;

        return detail::sort<RandomIt>().call(std::forward<ExPolicy>(policy),
            is_seq(), first, last, std::forward<Compare>(comp),
            std::forward<Proj>(proj));
    }
}}}    // namespace hpx::parallel::v1
