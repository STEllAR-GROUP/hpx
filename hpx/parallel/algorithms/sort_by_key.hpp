//  Copyright (c) 2016 Hartmut Kaiser
//  Copyright (c) 2016 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_ALGORITHM_SORT_BY_KEY_DEC_2015)
#define HPX_PARALLEL_ALGORITHM_SORT_BY_KEY_DEC_2015

#include <hpx/config.hpp>
#include <hpx/util/tagged_pair.hpp>
#include <hpx/util/tuple.hpp>

#include <hpx/parallel/algorithms/sort.hpp>
#include <hpx/parallel/tagspec.hpp>
#include <hpx/parallel/util/zip_iterator.hpp>

#include <algorithm>
#include <iterator>
#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
    ///////////////////////////////////////////////////////////////////////////
    // sort
    namespace detail
    {
        /// \cond NOINTERNAL
        struct extract_key
        {
            template <typename Tuple>
            auto operator() (Tuple && t) const
            ->  decltype(hpx::util::get<0>(std::forward<Tuple>(t)))
            {
                return hpx::util::get<0>(std::forward<Tuple>(t));
            }
        };
        /// \endcond
    }

    //-----------------------------------------------------------------------------
    /// Sorts one range of data using keys supplied in another range.
    /// The key elements in the range [key_first, key_last) are sorted in
    /// ascending order with the corresponding elements in the value range
    /// moved to follow the sorted order.
    /// The algorithm is not stable, the order of equal elements is not guaranteed
    /// to be preserved.
    /// The function uses the given comparison function object comp (defaults
    /// to using operator<()).
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
    /// \tparam KeyIter     The type of the key iterators used (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     random access iterator.
    /// \tparam ValueIter   The type of the value iterators used (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     random access iterator.
    /// \tparam Comp        The type of the function/function object to use
    ///                     (deduced).
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param key_first    Refers to the beginning of the sequence of key
    ///                     elements the algorithm will be applied to.
    /// \param key_last     Refers to the end of the sequence of key elements the
    ///                     algorithm will be applied to.
    /// \param value_first  Refers to the beginning of the sequence of value
    ///                     elements the algorithm will be applied to, the range
    ///                     of elements must match [key_first, key_last)
    /// \param comp         comp is a callable object. The return value of the
    ///                     INVOKE operation applied to an object of type Comp,
    ///                     when contextually converted to bool, yields true if
    ///                     the first argument of the call is less than the
    ///                     second, and false otherwise. It is assumed that comp
    ///                     will not apply any non-constant function through the
    ///                     dereferenced iterator.
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
    /// \returns  The \a sort_by-key algorithm returns a
    /// \a hpx::future<tagged_pair<tag::in1(KeyIter>, tag::in2(ValueIter)> >
    ///           if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and returns \a
    ///           \a tagged_pair<tag::in1(KeyIter), tag::in2(ValueIter)>
    ///           otherwise.
    ///           The algorithm returns a pair holding an iterator pointing to
    ///           the first element after the last element in the input key
    ///           sequence and an iterator pointing to the first element after
    ///           the last element in the input value sequence.
    //-----------------------------------------------------------------------------

    template <
        typename ExPolicy, typename KeyIter, typename ValueIter,
        typename Compare = detail::less
    >
    typename util::detail::algorithm_result<
        ExPolicy,
        hpx::util::tagged_pair<tag::in1(KeyIter), tag::in2(ValueIter)>
    >::type
    sort_by_key(ExPolicy && policy,
                KeyIter key_first,
                KeyIter key_last,
                ValueIter value_first,
                Compare && comp = Compare())
    {
#if !defined(HPX_HAVE_TUPLE_RVALUE_SWAP)
        static_assert(sizeof(KeyIter) == 0, // always false
            "sort_by_key is not supported unless HPX_HAVE_TUPLE_RVALUE_SWAP "
            "is defined");
#else
        static_assert(
            (hpx::traits::is_random_access_iterator<KeyIter>::value),
            "Requires a random access iterator.");
        static_assert(
            (hpx::traits::is_random_access_iterator<ValueIter>::value),
            "Requires a random access iterator.");

        ValueIter value_last = value_first;
        std::advance(value_last, std::distance(key_first, key_last));

        return detail::get_iter_tagged_pair<tag::in1, tag::in2>(
            hpx::parallel::sort(
                std::forward<ExPolicy>(policy),
                hpx::util::make_zip_iterator(key_first, value_first),
                hpx::util::make_zip_iterator(key_last, value_last),
                std::forward<Compare>(comp),
                detail::extract_key()));
#endif
    }
}}}

#endif
