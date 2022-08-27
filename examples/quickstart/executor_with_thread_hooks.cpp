//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

///////////////////////////////////////////////////////////////////////////////
// The purpose of this example is to show how to rite an executor that wraps
// any other executor and adds a hook into thread start and thread exit allowing
// to associate custom thread data with the tasks that are created by the
// underlying executor.

#include <hpx/assert.hpp>
#include <hpx/local/algorithm.hpp>
#include <hpx/local/execution.hpp>
#include <hpx/local/init.hpp>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <iostream>
#include <type_traits>
#include <utility>
#include <vector>

namespace executor_example {

    template <typename BaseExecutor>
    class executor_with_thread_hooks
    {
    private:
        struct on_exit
        {
            explicit on_exit(executor_with_thread_hooks const& exec)
              : exec_(exec)
            {
                exec_.on_start_();
            }

            ~on_exit()
            {
                exec_.on_stop_();
            }

            executor_with_thread_hooks const& exec_;
        };

        template <typename F>
        struct hook_wrapper
        {
            template <typename... Ts>
            decltype(auto) operator()(Ts&&... ts)
            {
                on_exit _{exec_};
                return hpx::util::invoke(f_, std::forward<Ts>(ts)...);
            }

            executor_with_thread_hooks const& exec_;
            F f_;
        };

    public:
        using execution_category = typename BaseExecutor::execution_category;
        using executor_parameters_type =
            typename BaseExecutor::executor_parameters_type;

        template <typename OnStart, typename OnStop>
        executor_with_thread_hooks(
            BaseExecutor& exec, OnStart&& start, OnStop&& stop)
          : exec_(exec)
          , on_start_(std::forward<OnStart>(start))
          , on_stop_(std::forward<OnStop>(stop))
        {
        }

        bool operator==(executor_with_thread_hooks const& rhs) const noexcept
        {
            return exec_ == rhs.exec_;
        }

        bool operator!=(executor_with_thread_hooks const& rhs) const noexcept
        {
            return !(*this == rhs);
        }

        executor_with_thread_hooks const& context() const noexcept
        {
            return *this;
        }

        // OneWayExecutor interface
        template <typename F, typename... Ts>
        decltype(auto) sync_execute(F&& f, Ts&&... ts) const
        {
            return hpx::parallel::execution::sync_execute(exec_,
                hook_wrapper<F>{*this, std::forward<F>(f)},
                std::forward<Ts>(ts)...);
        }

        // TwoWayExecutor interface
        template <typename F, typename... Ts>
        decltype(auto) async_execute(F&& f, Ts&&... ts) const
        {
            return hpx::parallel::execution::async_execute(exec_,
                hook_wrapper<F>{*this, std::forward<F>(f)},
                std::forward<Ts>(ts)...);
        }

        template <typename F, typename Future, typename... Ts>
        decltype(auto) then_execute(
            F&& f, Future&& predecessor, Ts&&... ts) const
        {
            return hpx::parallel::execution::then_execute(exec_,
                hook_wrapper<F>{*this, std::forward<F>(f)},
                std::forward<Future>(predecessor), std::forward<Ts>(ts)...);
        }

        // NonBlockingOneWayExecutor (adapted) interface
        template <typename F, typename... Ts>
        void post(F&& f, Ts&&... ts) const
        {
            hpx::parallel::execution::post(exec_,
                hook_wrapper<F>{*this, std::forward<F>(f)},
                std::forward<Ts>(ts)...);
        }

        // BulkOneWayExecutor interface
        template <typename F, typename S, typename... Ts>
        decltype(auto) bulk_sync_execute(
            F&& f, S const& shape, Ts&&... ts) const
        {
            return hpx::parallel::execution::bulk_sync_execute(exec_,
                hook_wrapper<F>{*this, std::forward<F>(f)}, shape,
                std::forward<Ts>(ts)...);
        }

        // BulkTwoWayExecutor interface
        template <typename F, typename S, typename... Ts>
        decltype(auto) bulk_async_execute(
            F&& f, S const& shape, Ts&&... ts) const
        {
            return hpx::parallel::execution::bulk_async_execute(exec_,
                hook_wrapper<F>{*this, std::forward<F>(f)}, shape,
                std::forward<Ts>(ts)...);
        }

        template <typename F, typename S, typename Future, typename... Ts>
        decltype(auto) bulk_then_execute(
            F&& f, S const& shape, Future&& predecessor, Ts&&... ts) const
        {
            return hpx::parallel::execution::bulk_then_execute(exec_,
                hook_wrapper<F>{*this, std::forward<F>(f)}, shape,
                std::forward<Future>(predecessor), std::forward<Ts>(ts)...);
        }

    private:
        using thread_hook = hpx::function<void()>;

        BaseExecutor& exec_;
        thread_hook on_start_;
        thread_hook on_stop_;
    };

    template <typename BaseExecutor, typename OnStart, typename OnStop>
    executor_with_thread_hooks<BaseExecutor> make_executor_with_thread_hooks(
        BaseExecutor& exec, OnStart&& on_start, OnStop&& on_stop)
    {
        return executor_with_thread_hooks<BaseExecutor>(exec,
            std::forward<OnStart>(on_start), std::forward<OnStop>(on_stop));
    }
}    // namespace executor_example

///////////////////////////////////////////////////////////////////////////////
// simple forwarding implementations of executor traits
namespace hpx { namespace parallel { namespace execution {

    template <typename BaseExecutor>
    struct is_one_way_executor<
        executor_example::executor_with_thread_hooks<BaseExecutor>>
      : is_one_way_executor<typename std::decay<BaseExecutor>::type>
    {
    };

    template <typename BaseExecutor>
    struct is_never_blocking_one_way_executor<
        executor_example::executor_with_thread_hooks<BaseExecutor>>
      : is_never_blocking_one_way_executor<
            typename std::decay<BaseExecutor>::type>
    {
    };

    template <typename BaseExecutor>
    struct is_two_way_executor<
        executor_example::executor_with_thread_hooks<BaseExecutor>>
      : is_two_way_executor<typename std::decay<BaseExecutor>::type>
    {
    };

    template <typename BaseExecutor>
    struct is_bulk_one_way_executor<
        executor_example::executor_with_thread_hooks<BaseExecutor>>
      : is_bulk_one_way_executor<typename std::decay<BaseExecutor>::type>
    {
    };

    template <typename BaseExecutor>
    struct is_bulk_two_way_executor<
        executor_example::executor_with_thread_hooks<BaseExecutor>>
      : is_bulk_two_way_executor<typename std::decay<BaseExecutor>::type>
    {
    };
}}}    // namespace hpx::parallel::execution

int hpx_main()
{
    std::vector<double> v(1000);
    std::iota(v.begin(), v.end(), 0.0);

    std::atomic<std::size_t> starts(0);
    std::atomic<std::size_t> stops(0);

    auto on_start = [&]() { ++starts; };
    auto on_stop = [&]() { ++stops; };

    auto exec = executor_example::make_executor_with_thread_hooks(
        hpx::execution::par.executor(), on_start, on_stop);

    hpx::experimental::for_loop(
        hpx::execution::par.on(exec), 0, v.size(), [](std::size_t) {});

    std::cout << "Executed " << starts.load() << " starts and " << stops.load()
              << " stops\n";

    HPX_ASSERT(starts.load() != 0);
    HPX_ASSERT(stops.load() != 0);

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    return hpx::local::init(hpx_main, argc, argv);
}
