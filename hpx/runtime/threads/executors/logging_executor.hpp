//  Copyright (c)      2017 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_RUNTIME_THREADS_LOGGING_EXECUTOR
#define HPX_RUNTIME_THREADS_LOGGING_EXECUTOR

#include <hpx/async.hpp>
#include <hpx/runtime/threads/thread_executor.hpp>
#include <hpx/runtime/threads/executors/pool_executor.hpp>
#include <hpx/runtime/threads/thread_pool_base.hpp>
#include <hpx/util/thread_description.hpp>
#include <hpx/util/thread_specific_ptr.hpp>
#include <hpx/lcos/dataflow.hpp>
#include <hpx/util/invoke.hpp>
#include <hpx/util/pack_traversal.hpp>
#include <hpx/util/demangle_helper.hpp>
#include <hpx/traits/future_traits.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <iostream>
#include <type_traits>
#include <utility>
#include <atomic>

#include <hpx/config/warnings_prefix.hpp>

//#define LOGGING_EXECUTOR_DEBUG 1
//#define GUIDED_POOL_EXECUTOR_FAKE_NOOP

//hpx::lcos::shared_future<hpx_linalg::MatrixGen<hpx_linalg::HostMem<double, std::allocator<double> > > >
//hpx::lcos::       future<hpx::threads::executors::async_log_type<
//                         hpx_linalg::MatrixGen<hpx_linalg::HostMem<double, std::allocator<double> > > > >

// --------------------------------------------------------------------
// pool_numa_hint
// --------------------------------------------------------------------
namespace hpx { namespace threads { namespace executors
{
    // --------------------------------------------------------------------
    namespace detail {
        static std::atomic<int> log_counter_;
    }

    // --------------------------------------------------------------------
    template <typename Future>
    struct async_log_type
    {
        using T = typename hpx::traits::future_traits<Future>::type;
        std::string name_;
        Future      data_;
        //
        operator Future &&() { return std::move(data_); }
        operator hpx::shared_future<T> &&() { return data_.share(); }
    };


    // --------------------------------------------------------------------
    // logging_executor
    // an executor compatible with scheduled executor API
    // --------------------------------------------------------------------
    template <typename Executor>
    struct HPX_EXPORT logging_executor {

//        template <typename F, typename ... Ts>
//        using deferred_log_type = async_log_type<hpx::future<typename util::detail::invoke_deferred_result<F, Ts...>::type>>;

    public:
        logging_executor(const Executor& ex)
            : internal_exec_(ex)
        {}

        // --------------------------------------------------------------------
        // async
        // --------------------------------------------------------------------
        template <typename F, typename ... Ts>
        future<typename util::detail::invoke_deferred_result<F, Ts...>::type>
        async_execute(F && f, Ts &&... ts)
        {
            typedef typename util::detail::invoke_deferred_result<F, Ts...>::type
                result_type;

            std::string node_name ="Node_" + std::to_string(detail::log_counter_++);
            std::cout << "\n-------------------------------\n";
            std::cout << "Node " << node_name << "\n";
            std::cout << "LOG: async_execute : Function    : "
                      << debug::print_type<F>() << "\n";
            std::cout << "LOG: async_execute : Arguments   : "
                      << debug::print_type<Ts...>(" | ") << "\n";
            std::cout << "LOG: async_execute : Result      : "
                      << debug::print_type<result_type>() << "\n";

            return
                internal_exec_.async_execute(
                    std::forward<F>(f),
                    std::forward<Ts>(ts)...);

//            return future<result_type> {
//                node_name,
//                internal_exec_.async_execute(
//                    std::forward<F>(f),
//                    std::forward<Ts>(ts)...)};
        }

        // --------------------------------------------------------------------
        // Continuation
        // --------------------------------------------------------------------
        template <typename F,
                  typename Future,
                  typename ... Ts,
                  typename = typename std::enable_if_t<traits::is_future<Future>::value>
                  >
        auto
        then_execute(F && f, Future && predecessor, Ts &&... ts)
        ->  future<typename util::detail::invoke_deferred_result<
            F, Future, Ts...>::type>
        {
            typedef typename hpx::util::detail::invoke_deferred_result<
                    F, Future, Ts...
                >::type result_type;

            std::string node_name ="Node_" + std::to_string(detail::log_counter_++);
            std::cout << "\n-------------------------------\n";
            std::cout << "Node " << node_name << "\n";
            std::cout << "LOG: then_execute : Function     : "
                      << debug::print_type<F>() << "\n";
            std::cout << "LOG: then_execute : Predecessor  : "
                      << debug::print_type<Future>() << "\n";
            std::cout << "LOG: then_execute : Future       : "
                      << debug::print_type<typename
                         traits::future_traits<Future>::result_type>() << "\n";
            std::cout << "LOG: then_execute : Arguments    : "
                      << debug::print_type<Ts...>(" | ") << "\n";
            std::cout << "LOG: then_execute : Result       : "
                      << debug::print_type<result_type>() << "\n";

            return
                internal_exec_.then_execute(
                        std::forward<F>(f),
                        std::forward<Future>(predecessor),
                        std::forward<Ts>(ts)...);

//            return async_log_type<result_type> {
//                node_name,
//                internal_exec_.then_execute(
//                        std::forward<F>(f),
//                        std::forward<Future>(predecessor),
//                        std::forward<Ts>(ts)...) };
        }

        // --------------------------------------------------------------------
        Executor internal_exec_;
    };
}}}

namespace hpx { namespace parallel { namespace execution
{
    template <typename Executor>
    struct executor_execution_category<
        threads::executors::logging_executor<Executor> >
    {
        typedef parallel::execution::parallel_execution_tag type;
    };

    template <typename Executor>
    struct is_two_way_executor<
            threads::executors::logging_executor<Executor> >
      : std::true_type
    {};

}}}

#include <hpx/config/warnings_suffix.hpp>

#endif /*HPX_RUNTIME_THREADS_GUIDED_POOL_EXECUTOR*/
