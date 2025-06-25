//  Copyright (c) 2022 Srinivas Yadav
//  Copyright (c) 2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_MODULE_LIKWID)
#include <hpx/execution/execution.hpp>
#include <hpx/functional/experimental/scope_exit.hpp>

#include <hpx/likwid/likwid_tls.hpp>

#include <string>
#include <utility>

namespace hpx::execution::experimental {

    ///////////////////////////////////////////////////////////////////////////
    template <typename BaseExecutor>
    class likwid_executor
    {
    private:
        template <typename F>
        struct hook_wrapper
        {
            template <typename... Ts>
            decltype(auto) operator()(Ts&&... ts)
            {
                exec_.on_start_();

                auto on_exit = hpx::experimental::scoped_exit{
                    [&exec_]() { exec_.on_stop_() }};

                return hpx::util::invoke(f_, HPX_FORWARD(Ts, ts)...);
            }

            likwid_executor const& exec_;
            F f_;
        };

    public:
        using execution_category =
            hpx::parallel::execution::executor_execution_category_t<
                BaseExecutor>;
        using executor_parameters_type =
            hpx::parallel::execution::executor_parameters_type_t<BaseExecutor>;

        likwid_executor(BaseExecutor& exec, std::string name)
          : exec_(exec)
          , name_(HPX_MOVE(name))
          , on_start_([&name_]() { hpx::likwid::start_region(name_.c_str()); })
          , on_stop_([&name_]() { hpx::likwid::stop_region(name_.c_str()); })
        {
        }

        bool operator==(likwid_executor const& rhs) const noexcept
        {
            return exec_ == rhs.exec_;
        }

        bool operator!=(likwid_executor const& rhs) const noexcept
        {
            return !(*this == rhs);
        }

        constexpr likwid_executor const& context() const noexcept
        {
            return *this;
        }

        // OneWayExecutor interface
        template <typename F, typename... Ts>
        friend decltype(auto) tag_invoke(
            hpx::parallel::execution::sync_execute_t,
            likwid_executor const& exec, F&& f, Ts&&... ts)
        {
            return hpx::parallel::execution::sync_execute(exec.exec_,
                hook_wrapper<F>{exec, HPX_FORWARD(F, f)},
                HPX_FORWARD(Ts, ts)...);
        }

        // TwoWayExecutor interface
        template <typename F, typename... Ts>
        friend decltype(auto) tag_invoke(
            hpx::parallel::execution::async_execute_t,
            likwid_executor const& exec, F&& f, Ts&&... ts)
        {
            return hpx::parallel::execution::async_execute(exec.exec_,
                hook_wrapper<F>{exec, HPX_FORWARD(F, f)},
                HPX_FORWARD(Ts, ts)...);
        }

        template <typename F, typename Future, typename... Ts>
        friend decltype(auto) tag_invoke(
            hpx::parallel::execution::then_execute_t,
            likwid_executor const& exec, F&& f, Future&& predecessor,
            Ts&&... ts)
        {
            return hpx::parallel::execution::then_execute(exec.exec_,
                hook_wrapper<F>{exec, HPX_FORWARD(F, f)},
                HPX_FORWARD(Future, predecessor), HPX_FORWARD(Ts, ts)...);
        }

        // NonBlockingOneWayExecutor (adapted) interface
        template <typename F, typename... Ts>
        friend void tag_invoke(hpx::parallel::execution::post_t,
            likwid_executor const& exec, F&& f, Ts&&... ts)
        {
            hpx::parallel::execution::post(exec.exec_,
                hook_wrapper<F>{exec, HPX_FORWARD(F, f)},
                HPX_FORWARD(Ts, ts)...);
        }

        // BulkOneWayExecutor interface
        template <typename F, typename S, typename... Ts>
        friend decltype(auto) tag_invoke(
            hpx::parallel::execution::bulk_sync_execute_t,
            likwid_executor const& exec, F&& f, S const& shape, Ts&&... ts)
        {
            return hpx::parallel::execution::bulk_sync_execute(exec.exec_,
                hook_wrapper<F>{exec, HPX_FORWARD(F, f)}, shape,
                HPX_FORWARD(Ts, ts)...);
        }

        // BulkTwoWayExecutor interface
        template <typename F, typename S, typename... Ts>
        friend decltype(auto) tag_invoke(
            hpx::parallel::execution::bulk_async_execute_t,
            likwid_executor const& exec, F&& f, S const& shape, Ts&&... ts)
        {
            return hpx::parallel::execution::bulk_async_execute(exec.exec_,
                hook_wrapper<F>{exec, HPX_FORWARD(F, f)}, shape,
                HPX_FORWARD(Ts, ts)...);
        }

        template <typename F, typename S, typename Future, typename... Ts>
        friend decltype(auto) tag_invoke(
            hpx::parallel::execution::bulk_then_execute_t,
            likwid_executor const& exec, F&& f, S const& shape,
            Future&& predecessor, Ts&&... ts)
        {
            return hpx::parallel::execution::bulk_then_execute(exec.exec_,
                hook_wrapper<F>{exec, HPX_FORWARD(F, f)}, shape,
                HPX_FORWARD(Future, predecessor), HPX_FORWARD(Ts, ts)...);
        }

    private:
        using thread_hook = hpx::function<void()>;

        BaseExecutor& exec_;
        std::string name_;
        thread_hook on_start_;
        thread_hook on_stop_;
    };
}    // namespace hpx::execution::experimental

///////////////////////////////////////////////////////////////////////////////
// simple forwarding implementations of executor traits
namespace hpx::parallel::execution {

    template <typename BaseExecutor>
    struct is_one_way_executor<
        hpx::execution::experimental::likwid_executor<BaseExecutor>>
      : is_one_way_executor<std::decay_t<BaseExecutor>>
    {
    };

    template <typename BaseExecutor>
    struct is_never_blocking_one_way_executor<
        hpx::execution::experimental::likwid_executor<BaseExecutor>>
      : is_never_blocking_one_way_executor<std::decay_t<BaseExecutor>>
    {
    };

    template <typename BaseExecutor>
    struct is_two_way_executor<
        hpx::execution::experimental::likwid_executor<BaseExecutor>>
      : is_two_way_executor<std::decay_t<BaseExecutor>>
    {
    };

    template <typename BaseExecutor>
    struct is_bulk_one_way_executor<
        hpx::execution::experimental::likwid_executor<BaseExecutor>>
      : is_bulk_one_way_executor<std::decay_t<BaseExecutor>>
    {
    };

    template <typename BaseExecutor>
    struct is_bulk_two_way_executor<
        hpx::execution::experimental::likwid_executor<BaseExecutor>>
      : is_bulk_two_way_executor<std::decay_t<BaseExecutor>>
    {
    };
}    // namespace hpx::parallel::execution

#endif
