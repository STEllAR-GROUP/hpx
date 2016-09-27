//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_DETAIL_ALGORITHM_RESULT_MAY_28_2014_1020PM)
#define HPX_PARALLEL_DETAIL_ALGORITHM_RESULT_MAY_28_2014_1020PM

#include <hpx/config.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/parallel/execution_policy_fwd.hpp>
#include <hpx/traits/concepts.hpp>
#include <hpx/util/invoke.hpp>
#include <hpx/util/unused.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { namespace util { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename ExPolicy, typename T>
    struct algorithm_result_impl
    {
        // The return type of the initiating function.
        typedef T type;

        // Obtain initiating function's return type.
        static type get()
        {
            return T();
        }

        static type get(T && t)
        {
            return t;
        }

        static type get(hpx::future<T> && t)
        {
            return t.get();
        }
    };

    template <typename ExPolicy>
    struct algorithm_result_impl<ExPolicy, void>
    {
        // The return type of the initiating function.
        typedef void type;

        // Obtain initiating function's return type.
        static void get() {}

        static void get(hpx::util::unused_type) {}

        static type get(hpx::future<void> && t)
        {
            t.get();
        }

        template <typename T>
        static type get(hpx::future<T> && t)
        {
            t.get();
        }
    };

    template <typename T>
    struct algorithm_result_impl<sequential_task_execution_policy, T>
    {
        // The return type of the initiating function.
        typedef hpx::future<T> type;

        // Obtain initiating function's return type.
        static type get(T && t)
        {
            return hpx::make_ready_future(std::move(t));
        }

        static type get(hpx::future<T> && t)
        {
            return std::move(t);
        }
    };

    template <typename T>
    struct algorithm_result_impl<parallel_task_execution_policy, T>
    {
        // The return type of the initiating function.
        typedef hpx::future<T> type;

        // Obtain initiating function's return type.
        static type get(T && t)
        {
            return hpx::make_ready_future(std::move(t));
        }

        static type get(hpx::future<T> && t)
        {
            return std::move(t);
        }
    };

    template <>
    struct algorithm_result_impl<sequential_task_execution_policy, void>
    {
        // The return type of the initiating function.
        typedef hpx::future<void> type;

        // Obtain initiating function's return type.
        static type get()
        {
            return hpx::make_ready_future();
        }

        static type get(hpx::util::unused_type)
        {
            return hpx::make_ready_future();
        }

        static type get(hpx::future<void> && t)
        {
            return std::move(t);
        }

        template <typename T>
        static type get(hpx::future<T> && t)
        {
            return hpx::future<void>(std::move(t));
        }
    };

    template <>
    struct algorithm_result_impl<parallel_task_execution_policy, void>
    {
        // The return type of the initiating function.
        typedef hpx::future<void> type;

        // Obtain initiating function's return type.
        static type get()
        {
            return hpx::make_ready_future();
        }

        static type get(hpx::util::unused_type)
        {
            return hpx::make_ready_future();
        }

        static type get(hpx::future<void> && t)
        {
            return std::move(t);
        }

        template <typename T>
        static type get(hpx::future<T> && t)
        {
            return hpx::future<void>(std::move(t));
        }
    };

#if defined(HPX_HAVE_VC_DATAPAR)
    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    struct algorithm_result_impl<datapar_task_execution_policy, T>
    {
        // The return type of the initiating function.
        typedef hpx::future<T> type;

        // Obtain initiating function's return type.
        static type get(T && t)
        {
            return hpx::make_ready_future(std::move(t));
        }

        static type get(hpx::future<T> && t)
        {
            return std::move(t);
        }
    };

    template <>
    struct algorithm_result_impl<datapar_task_execution_policy, void>
    {
        // The return type of the initiating function.
        typedef hpx::future<void> type;

        // Obtain initiating function's return type.
        static type get()
        {
            return hpx::make_ready_future();
        }

        static type get(hpx::util::unused_type)
        {
            return hpx::make_ready_future();
        }

        static type get(hpx::future<void> && t)
        {
            return std::move(t);
        }

        template <typename T>
        static type get(hpx::future<T> && t)
        {
            return hpx::future<void>(std::move(t));
        }
    };
#endif

    ///////////////////////////////////////////////////////////////////////////
    template <typename Executor, typename Parameters, typename T>
    struct algorithm_result_impl<
            sequential_task_execution_policy_shim<Executor, Parameters>, T>
      : algorithm_result_impl<sequential_task_execution_policy, T>
    {};

    template <typename Executor, typename Parameters>
    struct algorithm_result_impl<
            sequential_task_execution_policy_shim<Executor, Parameters>, void>
      : algorithm_result_impl<sequential_task_execution_policy, void>
    {};

    template <typename Executor, typename Parameters, typename T>
    struct algorithm_result_impl<
            parallel_task_execution_policy_shim<Executor, Parameters>, T>
      : algorithm_result_impl<parallel_task_execution_policy, T>
    {};

    template <typename Executor, typename Parameters>
    struct algorithm_result_impl<
            parallel_task_execution_policy_shim<Executor, Parameters>, void>
      : algorithm_result_impl<parallel_task_execution_policy, void>
    {};

    ///////////////////////////////////////////////////////////////////////////
    template <typename ExPolicy, typename T = void>
    struct algorithm_result
      : algorithm_result_impl<typename hpx::util::decay<ExPolicy>::type, T>
    {
        static_assert(!std::is_lvalue_reference<T>::value,
            "T shouldn't be a lvalue reference");
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename U, typename Conv,
    HPX_CONCEPT_REQUIRES_(
        hpx::traits::is_callable<Conv(U)>::value)>
    typename hpx::util::result_of<Conv(U)>::type
    convert_to_result(U && val, Conv && conv)
    {
        return hpx::util::invoke(conv, val);
    }

    template <typename U, typename Conv,
    HPX_CONCEPT_REQUIRES_(
        hpx::traits::is_callable<Conv(U)>::value)>
    hpx::future<typename hpx::util::result_of<Conv(U)>::type>
    convert_to_result(hpx::future<U> && f, Conv && conv)
    {
        typedef typename hpx::util::result_of<Conv(U)>::type result_type;

        return lcos::make_future<result_type>(
                std::move(f), std::forward<Conv>(conv)
            );
    }
}}}}

#endif
