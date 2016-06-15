//  Copyright (c) 2016 Minh-Khanh Do
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_SEGMENTED_ALGORITHM_FILL_MAY_30_2016)
#define HPX_PARALLEL_SEGMENTED_ALGORITHM_FILL_MAY_30_2016

#include <hpx/config.hpp>

#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/segmented_algorithms/for_each.hpp>

#include <boost/exception_ptr.hpp>

#include <algorithm>
#include <iterator>
#include <list>
#include <type_traits>
#include <vector>


namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
    ///////////////////////////////////////////////////////////////////////////
    // segmented_fill
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        /// \cond NOINTERNAL

        template<typename T>
        struct fill_function
        {
            fill_function(T val = T()) : value_(val) {}

            T value_;

            void operator()(T& val) const
            {
                val = value_;
            }

            template <typename Archive>
            void serialize(Archive& ar, unsigned version)
            {
                ar & value_;
            }
        };

        ///////////////////////////////////////////////////////////////////////
        // segmented implementation
        template <typename ExPolicy, typename InIter, typename T>
        static typename util::detail::algorithm_result<
            ExPolicy, void
        >::type
        fill_(ExPolicy && policy, InIter first, InIter last, T const& value,
            std::true_type)
        {
            typedef typename util::detail::algorithm_result<ExPolicy, void>::type result_type;
            return hpx::util::void_guard<result_type>(),
                hpx::parallel::for_each(std::forward<ExPolicy>(policy),
                    first, last, fill_function<T>(value));
        }

        // forward declare the non-segmented version of this algorithm
        template <typename ExPolicy, typename InIter, typename T>
        static typename util::detail::algorithm_result<
            ExPolicy, void>::type
        fill_(ExPolicy && policy, InIter first, InIter last, T const& value,
            std::false_type);
    }
        /// \endcond
}}}

#endif
