// Copyright (c) 2007-2015 Grant Mercer
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/is_heap.hpp

#if !defined(HPX_PARALLEL_DETAIL_IS_HEAP_NOV_28_2015_1000PM)
#define HPX_PARALLEL_DETAIL_IS_HEAP_NOV_28_2015_1000PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/partitioner.hpp>

#include <boost/range/functions.hpp>
#include <boost/algorithm/cxx11/all_of.hpp>
#include <boost/type_traits/is_base_of.hpp>
#include <boost/utility/enable_if.hpp>

#include <algorithm>
#include <numeric>
#include <iterator>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
    //////////////////////////////////////////////////////////////////////
    // is_heap
    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename RndIter>
        struct is_heap: public detail::algorithm<is_heap<RndIter>, bool>
        {
            is_heap()
                : is_heap::algorithm("is_heap")
            {}

            template<typename ExPolicy, typename Pred>
            static bool
            sequential(ExPolicy, RndIter first, RndIter last,
                    Pred && pred)
            {
                return std::is_heap(first, last, std::forward<Pred>(pred));
            }

            template<typename ExPolicy, typename Pred>
            static typename util::detail::algorithm_result<ExPolicy, bool>::type
            parallel(ExPolicy policy, RndIter first, RndIter last,
                    Pred && pred)
            {
                typedef typename std::iterator_traits<RndIter>::reference
                    reference;
                typedef typename std::iterator_traits<RndIter>::difference_type
                    difference_type;
                typedef typename util::detail::algorithm_result<ExPolicy, bool>
                    result;

                difference_type count = std::distance(first, last);
                if(count <= 1)
                    return result::get(true);

                return util::partitioner<ExPolicy, bool>::call(
                    policy, first, count,
                    [pred, last](RndIter part_begin,
                    std::size_t part_size) mutable -> bool
                    {
                        std::size_t p = 0;
                        std::size_t c = 1;

                        RndIter pp = part_begin;
                        while(c < part_size) {
                            RndIter cp = part_begin + c;
                            if(pred(*pp, *cp))
                                return false;
                            ++c;
                            ++cp;

                            if(c == part_size)
                                return true;
                            if(pred(*pp, *cp))
                                return false;
                            ++p;
                            ++pp;
                            c = 2 * p + 1;
                        }
                        return true;
                    },
                    [](std::vector<hpx::future<bool> > && results)
                    {
                        return boost::algorithm::all_of(
                                boost::begin(results), boost::end(results),
                                [](hpx::future<bool>& val)
                                {
                                    return val.get();
                                });
                    });
            }
        };
        /// \endcond
    }

    /// Determintes if the range [first, last) is a max heap. 
    /// Uses operator < to compare elements.
    ///
    /// \note   Complexity: at most(N+S-1) comparisons where
    ///         \a N = distance(first, last).
    ///         \a S = number of partitions
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam RndIter     The type of the source iterators used. The iterator
    ///                     type must meet the requirements for a Random Access
    ///                     Iterator
    /// \param policy       The execution policy to use for the scheduling of 
    ///                     iterations.
    /// \param first        Refers to the beginning of the sequence of elements 
    ///                     of that the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements of
    ///                     that the algorithm will be applied to.
    ///
    /// The comparison operations in the parallel \a is_heap algorithm invoked
    /// with an execution policy object of type \a sequential_execution_policy 
    /// executes in sequential order in the calling thread.
    ///
    /// The comparison operations in the parallel \a is_heap algorithm invoked
    /// with an execution_policy object of type \a parallel_execution_policy 
    /// or \a parallel_task_execution_policy are permitted to execute in an
    /// unordered fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns The \a is_heap algorithm returns a \a hpx::future<bool> if
    ///          the execution policy is of type \a task_execution_policy 
    ///          and returns \a bool otherwise.
    ///          The \a is_heap algorithm returns a bool if each element in
    ///          the sequence [first, last) satisfies the predicate. If the
    ///          range [first, last) contains less than two elements, the 
    ///          function always returns true.
    template <typename ExPolicy, typename RndIter>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename util::detail::algorithm_result<ExPolicy, bool>::type
    >::type
    is_heap(ExPolicy && policy, RndIter first, RndIter last)
    {
        typedef typename std::iterator_traits<RndIter>::iterator_category
            iterator_category;
        static_assert(
                (boost::is_base_of<
                 std::random_access_iterator_tag, iterator_category
                    >::value),
                "Requires at least random access iterator.");

        typedef typename is_sequential_execution_policy<ExPolicy>::type is_seq;
        typedef typename std::iterator_traits<RndIter>::value_type value_type;

        return detail::is_heap<RndIter>().call(
                std::forward<ExPolicy>(policy), is_seq(), first, last,
                std::less<value_type>());
    }
    
    /// Determintes if the range [first, last) is a heap. Uses pred to 
    /// compare elements.
    ///
    /// \note   Complexity: at most(N+S-1) comparisons where
    ///         \a N = distance(first, last).
    ///         \a S = number of partitions
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam RndIter     The type of the source iterators used. The iterator
    ///                     type must meet the requirements for a Random Access
    ///                     Iterator
    /// \param policy       The execution policy to use for the scheduling of 
    ///                     iterations.
    /// \param first        Refers to the beginning of the sequence of elements 
    ///                     of that the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements of
    ///                     that the algorithm will be applied to.
    /// \param pred         Refers to the binary predicate which returns true if
    ///                     the first argument should be treated as less than the
    ///                     second argument. The signature of the function should
    ///                     be equivalent to
    ///                     \code
    ///                     bool pred(const Type &a, const Type &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but the 
    ///                     function must not modify the objects passed to it. 
    ///                     The type \a Type must be such that objects of type 
    ///                     \a RndIter can be dereferenced and then implicity 
    ///                     converted to \a Type.
    ///
    /// The comparison operations in the parallel \a is_heap algorithm invoked
    /// with an execution policy object of type \a sequential_execution_policy 
    /// executes in sequential order in the calling thread.
    ///
    /// The comparison operations in the parallel \a is_heap algorithm invoked
    /// with an execution_policy object of type \a parallel_execution_policy 
    /// or \a parallel_task_execution_policy are permitted to execute in an
    /// unordered fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns The \a is_heap algorithm returns a \a hpx::future<bool> if
    ///          the execution policy is of type \a task_execution_policy 
    ///          and returns \a bool otherwise.
    ///          The \a is_heap algorithm returns a bool if each element in
    ///          the sequence [first, last) satisfies the predicate. If the
    ///          range [first, last) contains less than two elements, the 
    ///          function always returns true.
    template <typename ExPolicy, typename RndIter, typename Pred>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename util::detail::algorithm_result<ExPolicy, bool>::type
    >::type
    is_heap(ExPolicy && policy, RndIter first, RndIter last, Pred && pred)
    {
        typedef typename std::iterator_traits<RndIter>::iterator_category
            iterator_category;
        static_assert(
                (boost::is_base_of<
                 std::random_access_iterator_tag, iterator_category
                    >::value),
                "Requires at least random access iterator.");

        typedef typename is_sequential_execution_policy<ExPolicy>::type is_seq;

        return detail::is_heap<RndIter>().call(
                std::forward<ExPolicy>(policy), is_seq(), first, last,
                std::forward<Pred>(pred));
    }
}}}

#endif
