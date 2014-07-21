//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_DETAIL_ALGORITHM_RESULT_MAY_28_2014_1020PM)
#define HPX_PARALLEL_DETAIL_ALGORITHM_RESULT_MAY_28_2014_1020PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/util/unused.hpp>

#include <boost/static_assert.hpp>
#include <boost/type_traits/is_lvalue_reference.hpp>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1) { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename ExPolicy, typename T>
    struct algorithm_result_impl
    {
        // The return type of the initiating function.
        typedef T type;

        // Obtain initiating function's return type.
        static type get(T && t)
        {
            return std::move(t);
        }
    };

    template <typename ExPolicy>
    struct algorithm_result_impl<ExPolicy, void>
    {
        // The return type of the initiating function.
        typedef void type;

        // Obtain initiating function's return type.
        static void get(hpx::util::unused_type) {}
    };

    template <typename T>
    struct algorithm_result_impl<task_execution_policy, T>
    {
        // The return type of the initiating function.
        typedef hpx::future<T> type;

        // Obtain initiating function's return type.
        static type get(T && t)
        {
            return hpx::make_ready_future(std::move(t));
        }
    };

    template <>
    struct algorithm_result_impl<task_execution_policy, void>
    {
        // The return type of the initiating function.
        typedef hpx::future<void> type;

        // Obtain initiating function's return type.
        static type get(hpx::util::unused_type)
        {
            return hpx::make_ready_future();
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename ExPolicy, typename T = void>
    struct algorithm_result
      : algorithm_result_impl<typename hpx::util::decay<ExPolicy>::type, T>
    {
        BOOST_STATIC_ASSERT_MSG(!boost::is_lvalue_reference<T>::value,
            "T shouldn't be a lvalue reference");
    };
}}}}

#endif
