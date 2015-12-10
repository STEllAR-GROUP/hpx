//  Copyright (c) 2015 Grant Mercer
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/make_heap.hpp

#if !defined(HPX_PARALLEL_ALGORITHMS_MAKE_HEAP_DEC_10_2015_0331PM)
#define HPX_PARALLEL_ALGORITHMS_MAKE_HEAP_DEC_10_2015_0331PM

#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/cancellation_token.hpp>
#include <hpx/parallel/util/partitioner.hpp>
#include <hpx/parallel/util/loop.hpp>

#include <algorithm>
#include <iterator>
#include <functional>

#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_base_of.hpp>
#include <boost/type_traits/is_same.hpp>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
    //////////////////////////////////////////////////////////////////////
    // make_heap
    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename RndIter>
        struct make_heap: public detail::algorithm<make_heap<RndIter>, void>
        {
            make_heap()
                : make_heap::algorithm("make_heap")
            {}

            template<typename ExPolicy, typename Pred>
            static hpx::util::unused_type
            sequential(ExPolicy, RndIter first, RndIter last,
                    Pred && pred)
            {
                std::make_heap(first, last, std::forward<Pred>(pred));
                return hpx::util::unused;
            }

            template<typename ExPolicy, typename Pred>
            static typename util::detail::algorithm_result<ExPolicy, void>::type
            parallel(ExPolicy policy, RndIter first, RndIter last,
                    Pred && pred)
            {

            }
        };
    }

    template <typename ExPolicy, typename RndIter>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename util::detail::algorithm_result<ExPolicy>::type
    >::type
    make_heap(ExPolicy && policy, RndIter first, RndIter last)
    {
        typedef typename std::iterator_traits<RndIter>::iterator_category
            iterator_category;
        static_assert(
                (boost::is_base_of<
                 std::random_access_iterator_tag, iterator_category
                    >::value),
                "Requires random access iterator.");

        typedef typename is_sequential_execution_policy<ExPolicy>::type is_seq;
        typedef typename std::iterator_traits<RndIter>::value_type value_type;

        return detail::make_heap<RndIter>().call(
                std::forward<ExPolicy>(policy), is_seq(), first, last,
                std::less<value_type>());
    }
}}}
#endif
