//  Copyright (c) 2022 Srinivas Yadav
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <likwid.h>
#include <hpx/execution/execution.hpp>

namespace hpx { namespace execution {

    ///////////////////////////////////////////////////////////////////////////
    template <typename BaseExecutor>
    class likwid_executor
    {
    private:
        struct on_exit
        {
            explicit on_exit(likwid_executor const& exec)
              : exec_(exec)
            {
                exec_.on_start_();
            }

            ~on_exit()
            {
                exec_.on_stop_();
            }

            likwid_executor const& exec_;
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

            likwid_executor const& exec_;
            F f_;
        };

    public:
        using execution_category = typename 
            hpx::parallel::execution::executor_execution_category<BaseExecutor>::type;
        using executor_parameters_type =
            typename hpx::parallel::execution::executor_parameters_type<BaseExecutor>::type;

        likwid_executor(
            BaseExecutor& exec, std::string name)
          : exec_(exec)
        {
            on_start_ = [name]() {likwid_markerStartRegion(name.data());};
            on_stop_ = [name]() {likwid_markerStopRegion(name.data());};
        }

        bool operator==(likwid_executor const& rhs) const noexcept
        {
            return exec_ == rhs.exec_;
        }

        bool operator!=(likwid_executor const& rhs) const noexcept
        {
            return !(*this == rhs);
        }

        likwid_executor const& context() const noexcept
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

    template <typename BaseExecutor>
    likwid_executor<BaseExecutor> make_likwid_executor(
        BaseExecutor& exec, std::string name)
    {
        return likwid_executor<BaseExecutor>(exec, name);
    }
}}

///////////////////////////////////////////////////////////////////////////////
// simple forwarding implementations of executor traits
namespace hpx { namespace parallel { namespace execution {

    template <typename BaseExecutor>
    struct is_one_way_executor<
        hpx::execution::likwid_executor<BaseExecutor>>
      : is_one_way_executor<std::decay_t<BaseExecutor>>
    {
    };

    template <typename BaseExecutor>
    struct is_never_blocking_one_way_executor<
        hpx::execution::likwid_executor<BaseExecutor>>
      : is_never_blocking_one_way_executor<
            std::decay_t<BaseExecutor>>
    {
    };

    template <typename BaseExecutor>
    struct is_two_way_executor<
        hpx::execution::likwid_executor<BaseExecutor>>
      : is_two_way_executor<std::decay_t<BaseExecutor>>
    {
    };

    template <typename BaseExecutor>
    struct is_bulk_one_way_executor<
        hpx::execution::likwid_executor<BaseExecutor>>
      : is_bulk_one_way_executor<std::decay_t<BaseExecutor>>
    {
    };

    template <typename BaseExecutor>
    struct is_bulk_two_way_executor<
        hpx::execution::likwid_executor<BaseExecutor>>
      : is_bulk_two_way_executor<std::decay_t<BaseExecutor>>
    {
    };
}}}    // namespace hpx::parallel::execution