//  Copyright (c) 2020 Francisco Jose Tapia
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/partial_sort.hpp

#pragma once

#if defined(DOXYGEN)
namespace hpx {

    /// Places the first middle - first elements from the range [first, last)
    /// as sorted with respect to comp into the range [first, middle). The rest
    /// of the elements in the range [middle, last) are placed in an unspecified
    /// order.
    ///
    /// \note   Complexity: Approximately (last - first) * log(middle - first)
    ///         comparisons.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it applies user-provided function objects.
    /// \tparam RandIter    The type of the source begin, middle, and end
    ///                     iterators used (deduced). This iterator type must
    ///                     meet the requirements of a random access iterator.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param middle       Refers to the middle of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    ///
    /// \returns  The \a partial_sort algorithm returns a
    ///           \a hpx::future<void> if the execution policy is of
    ///           type \a sequenced_task_policy or
    ///           \a parallel_task_policy and returns void otherwise.
    ///
    template <typename RandIter>
    typename util::detail::algorithm_result<ExPolicy>::type partial_sort(
        ExPolicy&& policy, RandIter first, RandIter middle, RandIter last);

}    // namespace hpx

#else

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/async_local/dataflow.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/functional/invoke.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/parallel/util/detail/sender_util.hpp>
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
#include <hpx/parallel/algorithms/sort.hpp>
#include <hpx/parallel/util/compare_projected.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/detail/chunk_size.hpp>
#include <hpx/parallel/util/projection_identity.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
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

        //////////////////////////////////////////////////////////////////////
        ///
        /// \brief Obtain the position of the most significant bit set in N
        ///
        /// \param num : Number to examine
        ///
        /// \return Position of the first bit set
        ///
        inline constexpr unsigned nbits64(std::uint64_t num) noexcept
        {
            unsigned nb = 0;
            if (num >= (1ull << 32))
            {
                nb += 32;
                num = (num >> 32);
            }
            if (num >= (1ull << 16))
            {
                nb += 16;
                num = (num >> 16);
            }
            if (num >= (1ull << 8))
            {
                nb += 8;
                num = (num >> 8);
            }
            if (num >= (1ull << 4))
            {
                nb += 4;
                num = (num >> 4);
            }
            if (num >= (1ull << 2))
            {
                nb += 2;
                num = (num >> 2);
            }
            if (num >= (1ull << 1))
            {
                nb += 1;
                num = (num >> 1);
            }
            return nb;
        }

        ///////////////////////////////////////////////////////////////////////
        ///
        /// Receive a range between first and last, obtain 9 values
        /// between the elements  including the first and the previous
        /// to the last. Obtain the iterator to the mid value and swap
        /// with the first position
        //
        /// \param first    iterator to the first element
        /// \param last     iterator to the last element
        /// \param comp     object to Comp two elements
        ///
        template <typename Iter, typename Comp>
        inline constexpr void pivot3(
            Iter first, Iter last, Comp&& comp) noexcept
        {
            auto N2 = (last - first) >> 1;
            Iter it_val =
                mid3(first + 1, first + N2, last - 1, std::forward<Comp>(comp));
#if defined(HPX_HAVE_CXX20_STD_RANGES_ITER_SWAP)
            std::ranges::iter_swap(first, it_val);
#else
            std::iter_swap(first, it_val);
#endif
        }

        ///////////////////////////////////////////////////////////////////////
        ///
        /// This function obtain a pivot in the range and filter the elements
        /// according the value of that pivot
        ///
        /// \param first : iterator to the first element
        /// \param end : iterator to the element after the last
        /// \param comp : object to Comp two elements
        ///
        /// \return iterator where is the pivot used in the filtering
        ///
        template <typename Iter, typename Comp>
        constexpr inline Iter filter(Iter first, Iter end, Comp&& comp)
        {
            std::int64_t nelem = (end - first);
            if (nelem > 4096)
            {
                pivot9(first, end, comp);
            }
            else
            {
                pivot3(first, end, comp);
            }

            typename std::iterator_traits<Iter>::value_type const& pivot =
                *first;

            Iter c_first = first + 1, c_last = end - 1;
            while (HPX_INVOKE(comp, *c_first, pivot))
            {
                ++c_first;
            }
            while (HPX_INVOKE(comp, pivot, *c_last))
            {
                --c_last;
            }

            while (c_first < c_last)
            {
#if defined(HPX_HAVE_CXX20_STD_RANGES_ITER_SWAP)
                std::ranges::iter_swap(c_first++, c_last--);
#else
                std::iter_swap(c_first++, c_last--);
#endif
                while (HPX_INVOKE(comp, *c_first, pivot))
                {
                    ++c_first;
                }
                while (HPX_INVOKE(comp, pivot, *c_last))
                {
                    --c_last;
                }
            }

#if defined(HPX_HAVE_CXX20_STD_RANGES_ITER_SWAP)
            std::ranges::iter_swap(first, c_last);
#else
            std::iter_swap(first, c_last);
#endif
            return c_last;
        }

        ///////////////////////////////////////////////////////////////////////
        ///
        /// Internal function to divide and sort the ranges
        //
        /// \param first : iterator to the first element to be sorted
        /// \param middle: iterator defining the last element to be sorted
        /// \param end : iterator to the element after the end in the range
        /// \param level : level of depth from the top level call
        /// \param comp : object for to Comp elements
        ///
        template <typename Iter, typename Comp>
        inline void recursive_partial_sort(
            Iter first, Iter middle, Iter end, std::uint32_t level, Comp&& comp)
        {
            constexpr std::uint32_t nmin = 24;
            std::int64_t nelem = end - first;
            std::int64_t nmid = middle - first;
            if (nelem == 0 || nmid == 0)
            {
                return;
            }

            if (nelem < nmin)
            {
                std::sort(first, end, comp);
                return;
            }

            if (nmid == 1)
            {
                for (Iter it = first + 1; it != end; ++it)
                {
                    if (HPX_INVOKE(comp, *it, *first))
                    {
#if defined(HPX_HAVE_CXX20_STD_RANGES_ITER_SWAP)
                        std::ranges::iter_swap(it, first);
#else
                        std::iter_swap(it, first);
#endif
                    }
                }
                return;
            }

            if (level == 0)
            {
                std::make_heap(first, end, comp);
                std::sort_heap(first, middle, comp);
                return;
            }

            Iter c_last = filter(first, end, comp);
            if (middle >= c_last)
            {
                std::sort(first, c_last, comp);
                if (middle != c_last)
                {
                    recursive_partial_sort(c_last + 1, middle, end, level - 1,
                        std::forward<Comp>(comp));
                }
                return;
            }

            recursive_partial_sort(
                first, middle, c_last, level - 1, std::forward<Comp>(comp));
        }

        ///////////////////////////////////////////////////////////////////////
        ///
        /// Internal function to divide and sort the ranges
        ///
        /// \param first : iterator to the first element
        /// \param middle: iterator defining the last element to be sorted
        /// \param end : iterator to the element after the end in the range
        /// \param level : level of depth from the top level call
        /// \param comp : object for to Comp elements
        ///
        template <typename ExPolicy, typename Iter, typename Comp>
        hpx::future<Iter> parallel_partial_sort(ExPolicy&& policy, Iter first,
            Iter middle, Iter last, std::uint32_t level, Comp&& comp = Comp())
        {
            std::int64_t nelem = last - first;
            std::int64_t nmid = middle - first;
            HPX_ASSERT(nelem >= nmid);

            if (nelem == 0 || nmid == 0)
            {
                return hpx::make_ready_future(last);
            }

            if (nmid < 4096 || level < 12)
            {
                recursive_partial_sort(first, middle, last, level, comp);
                return hpx::make_ready_future(last);
            }

            Iter c_last = filter(first, last, comp);
            if (middle >= c_last)
            {
                // figure out the chunk size to use
                std::size_t const cores = execution::processing_units_count(
                    policy.parameters(), policy.executor());

                // number of elements to sort
                std::size_t chunk_size = execution::get_chunk_size(
                    policy.parameters(), policy.executor(), 0, cores, nelem);

                hpx::future<Iter> left = execution::async_execute(
                    policy.executor(), &sort_thread<ExPolicy, Iter, Comp>,
                    policy, first, c_last, comp, chunk_size);

                hpx::future<Iter> right;
                if (middle != c_last)
                {
                    right = execution::async_execute(policy.executor(),
                        &parallel_partial_sort<ExPolicy, Iter, Comp>, policy,
                        c_last + 1, middle, last, level - 1, comp);
                }
                else
                {
                    right = hpx::make_ready_future(last);
                }

                return hpx::dataflow(
                    [last](hpx::future<Iter>&& left,
                        hpx::future<Iter>&& right) -> Iter {
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

            return parallel_partial_sort(std::forward<ExPolicy>(policy), first,
                middle, c_last, level - 1, std::forward<Comp>(comp));
        }
        /// \endcond NOINTERNAL
    }    // end namespace detail

    ///////////////////////////////////////////////////////////////////////////
    ///
    /// \brief : Rearranges elements such that the range [first, middle)
    ///          contains the sorted middle - first smallest elements in the
    ///          range [first, end).
    ///
    /// \param first : iterator to the first element
    /// \param middle: iterator defining the last element to be sorted
    /// \param end : iterator to the element after the end in the range
    /// \param comp : object for to Comp elements
    ///
    template <typename Iter, typename Sent, typename Comp>
    Iter sequential_partial_sort(Iter first, Iter middle, Sent end, Comp&& comp)
    {
        std::int64_t nelem = end - first;
        HPX_ASSERT(nelem >= 0);

        std::int64_t nmid = middle - first;
        HPX_ASSERT(nmid >= 0 && nmid <= nelem);

        if (nmid > 1024)
        {
            if (detail::is_sorted_sequential(first, middle, comp))
            {
                return first + nelem;
            }
        }

        std::uint32_t level = detail::nbits64(nelem) * 2;
        detail::recursive_partial_sort(
            first, middle, first + nelem, level, std::forward<Comp>(comp));
        return first + nelem;
    }

    ///////////////////////////////////////////////////////////////////////////
    ///
    /// \brief : Rearranges elements such that the range [first, middle)
    ///          contains the sorted middle - first smallest elements in the
    ///          range [first, end).
    ///
    /// \param first : iterator to the first element
    /// \param middle: iterator defining the last element to be sorted
    /// \param end : iterator to the element after the end in the range
    /// \param comp : object for to Comp elements
    ///
    template <typename ExPolicy, typename Iter, typename Sent, typename Comp>
    hpx::future<Iter> parallel_partial_sort(
        ExPolicy&& policy, Iter first, Iter middle, Sent end, Comp&& comp)
    {
        std::int64_t nelem = end - first;
        HPX_ASSERT(nelem >= 0);

        std::int64_t nmid = middle - first;
        HPX_ASSERT(nmid >= 0 && nmid <= nelem);

        if (nmid > 1024)
        {
            if (detail::is_sorted_sequential(first, middle, comp))
            {
                return hpx::make_ready_future(first + nelem);
            }
        }

        std::uint32_t level = parallel::v1::detail::nbits64(nelem) * 2;
        return detail::parallel_partial_sort(std::forward<ExPolicy>(policy),
            first, middle, first + nelem, level, std::forward<Comp>(comp));
    }

    ///////////////////////////////////////////////////////////////////////
    // partial_sort
    template <typename RandIter>
    struct partial_sort
      : public detail::algorithm<partial_sort<RandIter>, RandIter>
    {
        partial_sort()
          : partial_sort::algorithm("partial_sort")
        {
        }

        template <typename ExPolicy, typename Iter, typename Sent,
            typename Comp, typename Proj>
        static Iter sequential(ExPolicy, Iter first, Iter middle, Sent last,
            Comp&& comp, Proj&& proj)
        {
            return sequential_partial_sort(first, middle, last,
                util::compare_projected<Comp, Proj>(
                    std::forward<Comp>(comp), std::forward<Proj>(proj)));
        }

        template <typename ExPolicy, typename Iter, typename Sent,
            typename Comp, typename Proj>
        static typename util::detail::algorithm_result<ExPolicy, Iter>::type
        parallel(ExPolicy&& policy, Iter first, Iter middle, Sent last,
            Comp&& comp, Proj&& proj)
        {
            typedef util::detail::algorithm_result<ExPolicy, Iter>
                algorithm_result;

            try
            {
                // call the sort routine and return the right type,
                // depending on execution policy
                return algorithm_result::get(parallel_partial_sort(
                    std::forward<ExPolicy>(policy), first, middle, last,
                    util::compare_projected<Comp, Proj>(
                        std::forward<Comp>(comp), std::forward<Proj>(proj))));
            }
            catch (...)
            {
                return algorithm_result::get(
                    detail::handle_exception<ExPolicy, Iter>::call(
                        std::current_exception()));
            }
        }
    };
}}}    // namespace hpx::parallel::v1

namespace hpx {

    HPX_INLINE_CONSTEXPR_VARIABLE struct partial_sort_t final
      : hpx::detail::tag_parallel_algorithm<partial_sort_t>
    {
    private:
        // clang-format off
        template <typename RandIter,
            typename Comp = hpx::parallel::v1::detail::less,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator<RandIter>::value &&
                hpx::is_invocable_v<Comp,
                    typename std::iterator_traits<RandIter>::value_type,
                    typename std::iterator_traits<RandIter>::value_type
                >
            )>
        // clang-format on
        friend RandIter tag_fallback_dispatch(hpx::partial_sort_t,
            RandIter first, RandIter middle, RandIter last,
            Comp&& comp = Comp())
        {
            static_assert(
                hpx::traits::is_random_access_iterator<RandIter>::value,
                "Requires at least random access iterator.");

            return parallel::v1::partial_sort<RandIter>().call(
                hpx::execution::seq, first, middle, last,
                std::forward<Comp>(comp),
                parallel::util::projection_identity());
        }

        // clang-format off
        template <typename ExPolicy, typename RandIter,
            typename Comp = hpx::parallel::v1::detail::less,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_iterator<RandIter>::value &&
                hpx::is_invocable_v<Comp,
                    typename std::iterator_traits<RandIter>::value_type,
                    typename std::iterator_traits<RandIter>::value_type
                >
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            RandIter>::type
        tag_fallback_dispatch(hpx::partial_sort_t, ExPolicy&& policy,
            RandIter first, RandIter middle, RandIter last,
            Comp&& comp = Comp())
        {
            static_assert(
                hpx::traits::is_random_access_iterator<RandIter>::value,
                "Requires at least random access iterator.");

            using algorithm_result =
                parallel::util::detail::algorithm_result<ExPolicy, RandIter>;

            return algorithm_result::get(
                parallel::v1::partial_sort<RandIter>().call(
                    std::forward<ExPolicy>(policy), first, middle, last,
                    std::forward<Comp>(comp),
                    parallel::util::projection_identity()));
        }
    } partial_sort{};
}    // namespace hpx

#endif    // DOXYGEN
