//  Copyright (c) 2016 Minh-Khanh Do
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_SEGMENTED_ALGORITHM_FILL_MAY_30_2016)
#define HPX_PARALLEL_SEGMENTED_ALGORITHM_FILL_MAY_30_2016

#include <hpx/config.hpp>
#include <hpx/traits/segmented_iterator_traits.hpp>

#include <hpx/include/parallel_for_each.hpp>
#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/fill.hpp>
#include <hpx/parallel/segmented_algorithms/detail/dispatch.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/detail/handle_remote_exceptions.hpp>

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

        // sequential remote implementation
        template <typename Algo, typename ExPolicy, typename SegIter, typename T>
        static typename util::detail::algorithm_result<ExPolicy, void>::type
        segmented_fill(Algo && algo, ExPolicy && policy,
            SegIter first, SegIter last, T const& value, std::true_type)
        {
            auto a = hpx::parallel::for_each(
                std::forward<ExPolicy>(policy), first, last, fill_function<T>(value));
            return util::detail::algorithm_result<ExPolicy, void>::get();
        }

        // parallel remote implementation
        template <typename Algo, typename ExPolicy, typename SegIter, typename T>
        static typename util::detail::algorithm_result<ExPolicy, void>::type
        segmented_fill(Algo && algo, ExPolicy && policy,
            SegIter first, SegIter last, T const& value, std::false_type)
        {
            typedef util::detail::algorithm_result<ExPolicy, void> result;

            auto r = hpx::parallel::for_each(
                std::forward<ExPolicy>(policy), first, last, fill_function<T>(value));

            return result::get(
                dataflow(
                    [=](decltype(r) && res)
                    {
                        hpx::util::unwrapped(res);
                        return;
                    },
                    std::move(r)));
        }

        ///////////////////////////////////////////////////////////////////////
        // segmented implementation
        template <typename ExPolicy, typename InIter, typename T>
        inline typename util::detail::algorithm_result<
            ExPolicy, void
        >::type
        fill_(ExPolicy && policy, InIter first, InIter last, T const& value,
            std::true_type)
        {
            typedef parallel::is_sequential_execution_policy<ExPolicy> is_seq;

            if (first == last)
            {
                return util::detail::algorithm_result<ExPolicy, void>::get();
            }

            return segmented_fill(
                fill(), std::forward<ExPolicy>(policy), first, last, value, is_seq()
            );

        }

        // forward declare the non-segmented version of this algorithm
        template <typename ExPolicy, typename InIter, typename T>
        inline typename util::detail::algorithm_result<
            ExPolicy, void>::type
        fill_(ExPolicy && policy, InIter first, InIter last, T const& value,
            std::false_type);
    }
        /// \endcond
}}}

#endif
