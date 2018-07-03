//  Copyright (c)      2017 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_RUNTIME_THREADS_CUDA_GRAPHS_EXECUTOR
#define HPX_RUNTIME_THREADS_CUDA_GRAPHS_EXECUTOR

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

#include <cstddef>
#include <cstdint>
#include <string>
#include <iostream>
#include <type_traits>
#include <utility>

#include <hpx/config/warnings_prefix.hpp>

#define CUDA_GRAPHS_EXECUTOR_DEBUG 1
//#define GUIDED_POOL_EXECUTOR_FAKE_NOOP

// --------------------------------------------------------------------
// pool_numa_hint
// --------------------------------------------------------------------
namespace hpx { namespace threads { namespace executors
{
    static std::atomic<int> node_count;

    struct cuda_graph_node {
        std::string name;
        //
        cuda_graph_node(const std::string &name_prefix) {
            name = name_prefix + std::to_string(node_count);
            node_count++;
        }
    };

    namespace detail {
        template <typename F, typename ... Ts>
        struct invoke_graph_result {
            using type = hpx::future<cuda_graph_node>;
        };
    }

    // --------------------------------------------------------------------
    // cuda_graphs_executor
    // an executor compatible with scheduled executor API
    // --------------------------------------------------------------------
    struct HPX_EXPORT cuda_graphs_executor {
    public:
        cuda_graphs_executor(const std::string& pool_name)
            : pool_exec_(pool_name)
        {}

        cuda_graphs_executor(const std::string& pool_name,
                         thread_stacksize stacksize)
            : pool_exec_(pool_name, stacksize)
        {}

        cuda_graphs_executor(const std::string& pool_name,
                         thread_priority priority,
                         thread_stacksize stacksize = thread_stacksize_default)
            : pool_exec_(pool_name, priority, stacksize)
        {}

        // --------------------------------------------------------------------
        // async
        // --------------------------------------------------------------------
        template <typename F, typename ... Ts>
        future<typename detail::invoke_graph_result<F, Ts...>::type>
        async_execute(F && f, Ts &&... ts)
        {
            typedef typename detail::invoke_graph_result<F, Ts...>::type
                result_type;

            std::cout << "LOG: async_execute : Function    : "
                      << debug::print_type<F>() << "\n";
            std::cout << "LOG: async_execute : Arguments   : "
                      << debug::print_type<Ts...>(" | ") << "\n";
            std::cout << "LOG: async_execute : Result      : "
                      << debug::print_type<result_type>() << "\n";

            lcos::local::futures_factory<result_type()> p(
                pool_exec_,
                util::deferred_call(std::forward<F>(f), std::forward<Ts>(ts)...)
            );
            auto id = p.apply(
                hpx::launch::async,
                pool_exec_.get_priority(),
                pool_exec_.get_stacksize(),
                threads::thread_schedule_hint_none
            );

            cuda_graph_node gn("graph_node");
            return hpx::make_ready_future(gn);
            //return p.get_future();
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
//        ->  future<typename detail::invoke_graph_result<
//            F, Future, Ts...>::type>
        -> future<typename detail::invoke_graph_result<F, Future, Ts...>::type>
        {
            typedef typename detail::invoke_graph_result<
                    F, Future, Ts...
                >::type result_type;

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

            auto func = hpx::util::bind(
                hpx::util::one_shot(std::forward<F>(f)),
                hpx::util::placeholders::_1, std::forward<Ts>(ts)...);

            typename hpx::traits::detail::shared_state_ptr<result_type>::type p =
                hpx::lcos::detail::make_continuation_exec<result_type>(
                    std::forward<Future>(predecessor), pool_exec_,
                    std::move(func));

            cuda_graph_node gn("graph_node");
            return hpx::make_ready_future(gn);

//            return hpx::traits::future_access<hpx::lcos::future<result_type>>::
//                create(std::move(p));
        }

        // --------------------------------------------------------------------
        pool_executor           pool_exec_;
    };

}}}

namespace hpx { namespace parallel { namespace execution
{
    template <>
    struct executor_execution_category<
        threads::executors::cuda_graphs_executor >
    {
        typedef parallel::execution::parallel_execution_tag type;
    };

    template <>
    struct is_two_way_executor<
            threads::executors::cuda_graphs_executor >
      : std::true_type
    {};

}}}

#include <hpx/config/warnings_suffix.hpp>

#endif /*HPX_RUNTIME_THREADS_GUIDED_POOL_EXECUTOR*/
