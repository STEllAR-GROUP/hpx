//  Copyright (c) 2014 Grant Mercer
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/detail/swap_ranges.hpp

#if !defined(HPX_PARALLEL_DETAIL_SWAP_RANGES_JUNE_20_2014_1006AM)
#define HPX_PARALLEL_DETAIL_SWAP_RANGES_JUNE_20_2014_1006AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/move.hpp>

#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/detail/algorithm_result.hpp>
#include <hpx/parallel/detail/dispatch.hpp>
#include <hpx/parallel/detail/for_each.hpp>
#include <hpx/parallel/util/zip_iterator.hpp>

#include <algorithm>
#include <iterator>

#include <boost/static_assert.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_base_of.hpp>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
    ///////////////////////////////////////////////////////////////////////////
    // swap ranges
    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename ForwardIter2>
        struct swap_ranges
          : public detail::algorithm<swap_ranges<ForwardIter2>, ForwardIter2>
        {
            swap_ranges()
              : swap_ranges::algorithm("swap_ranges")
            {}

            template <typename ExPolicy, typename ForwardIter1>
            static ForwardIter2
            sequential(ExPolicy const&, ForwardIter1 first1, ForwardIter1 last1,
                ForwardIter2 first2)
            {
                return std::swap_ranges(first1, last1, first2);
            }

            template <typename ExPolicy, typename ForwardIter1>
            static typename detail::algorithm_result<ExPolicy, ForwardIter2>::type
            parallel(ExPolicy const& policy, ForwardIter1 first1,
                ForwardIter1 last1, ForwardIter2 first2)
            {
                typedef hpx::util::zip_iterator<ForwardIter1, ForwardIter2>
                    zip_iterator;
                typedef typename zip_iterator::reference reference;
                typedef
                    typename detail::algorithm_result<ExPolicy, ForwardIter2>::type
                result_type;

                return get_iter<1, result_type>(
                    for_each_n<zip_iterator>().call(policy,
                        hpx::util::make_zip_iterator(first1, first2),
                        std::distance(first1, last1),
                        [](reference t) {
                            std::swap(hpx::util::get<0>(t), hpx::util::get<1>(t));
                        },
                        boost::mpl::false_()));
            }
        };
        /// \endcond
    }

    /// Exchanges elements between range [first1, last1) and another range
    /// starting at \a first2.
    ///
    /// \note   Complexity: Linear in the distance between \a first1 and \a last1
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the swap operations.
    /// \tparam ForwardIter1 The type of the first range of iterators to swap
    ///                     (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam ForwardIter2 The type of the second range of iterators to swap
    ///                     (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first1       Refers to the beginning of the first sequence of
    ///                     elements the algorithm will be applied to.
    /// \param last1        Refers to the end of the first sequence of elements
    ///                     the algorithm will be applied to.
    /// \param first2       Refers to the beginning of the second  sequence of
    ///                     elements the algorithm will be applied to.
    ///
    /// The swap operations in the parallel \a swap_ranges algorithm
    /// invoked with an execution policy object of type
    /// \a sequential_execution_policy execute in sequential order in
    /// the calling thread.
    ///
    /// The swap operations in the parallel \a swap_ranges algorithm
    /// invoked with an execution policy object of type
    /// \a parallel_execution_policy or \a task_execution_policy are
    /// permitted to execute in an unordered fashion in unspecified
    /// threads, and indeterminately sequenced within each thread.
    ///
    /// \returns  The \a swap_ranges algorithm returns a
    ///           \a hpx::future<ForwardIter2>  if the execution policy is of
    ///           type \a task_execution_policy and returns \a ForwardIter2
    ///           otherwise.
    ///           The \a swap_ranges algorithm returns iterator to the element
    ///           past the last element exchanged in the range beginning with
    ///           \a first2.
    ///
    template <typename ExPolicy, typename ForwardIter1, typename ForwardIter2>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename detail::algorithm_result<ExPolicy, ForwardIter2>::type
    >::type
    swap_ranges(ExPolicy && policy, ForwardIter1 first1, ForwardIter1 last1,
        ForwardIter2 first2)
    {
        typedef typename std::iterator_traits<ForwardIter1>::iterator_category
            iter1_category;
        typedef typename std::iterator_traits<ForwardIter2>::iterator_category
            iter2_category;

        BOOST_STATIC_ASSERT_MSG(
            (boost::is_base_of<
                std::forward_iterator_tag, iter1_category>::value),
            "Required at least forward iterator tag.");
        BOOST_STATIC_ASSERT_MSG(
            (boost::is_base_of<
                std::forward_iterator_tag, iter2_category>::value),
            "Required at least forward iterator tag.");

        typedef typename is_sequential_execution_policy<ExPolicy>::type is_seq;
        return detail::swap_ranges<ForwardIter2>().call(
            std::forward<ExPolicy>(policy),
            first1, last1, first2, is_seq());
    }
}}}

#endif
