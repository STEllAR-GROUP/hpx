//  Copyright (c) 2007-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/service_executors.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/errors/try_catch_exception_ptr.hpp>
#include <hpx/execution/executors/execution.hpp>
#include <hpx/execution/executors/fused_bulk_execute.hpp>
#include <hpx/execution/executors/static_chunk_size.hpp>
#include <hpx/execution/traits/executor_traits.hpp>
#include <hpx/execution_base/execution.hpp>
#include <hpx/executors/current_executor.hpp>
#include <hpx/functional/bind_front.hpp>
#include <hpx/functional/deferred_call.hpp>
#include <hpx/io_service/io_service_pool.hpp>
#include <hpx/modules/futures.hpp>
#include <hpx/pack_traversal/unwrap.hpp>
#include <hpx/threading_base/thread_helpers.hpp>

#include <algorithm>
#include <cstddef>
#include <exception>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx::parallel::execution::detail {

    class service_executor
    {
    public:
        // Associate the parallel_execution_tag executor tag type as a default
        // with this executor.
        using execution_category = hpx::execution::sequenced_execution_tag;

        // Associate the static_chunk_size executor parameters type as a
        // default with this executor.
        using executor_parameters_type =
            hpx::execution::experimental::static_chunk_size;

        explicit service_executor(
            [[maybe_unused]] hpx::util::io_service_pool* pool) noexcept
#if defined(HPX_COMPUTE_HOST_CODE)
          : pool_(pool)
#endif
        {
            HPX_ASSERT(pool);
        }

        template <typename F, typename... Ts>
        friend decltype(auto) tag_invoke(hpx::parallel::execution::post_t,
            [[maybe_unused]] service_executor const& exec, F&& f, Ts&&... ts)
        {
            using result_type =
                hpx::util::detail::invoke_deferred_result_t<F, Ts...>;

            hpx::move_only_function<result_type()> f_wrapper =
                hpx::util::deferred_call(
                    HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
            auto t = std::make_shared<post_wrapper_helper<decltype(f_wrapper)>>(
                HPX_MOVE(f_wrapper));

#if defined(HPX_COMPUTE_HOST_CODE)
            exec.pool_->get_io_service().post(hpx::bind_front(
                &post_wrapper_helper<decltype(f_wrapper)>::invoke,
                HPX_MOVE(t)));
#else
            HPX_ASSERT_MSG(
                false, "Attempting to use io_service_pool in device code");
#endif
        }

        template <typename F, typename... Ts>
        friend decltype(auto) tag_invoke(
            hpx::parallel::execution::async_execute_t,
            [[maybe_unused]] service_executor const& exec, F&& f, Ts&&... ts)
        {
            using result_type =
                hpx::util::detail::invoke_deferred_result_t<F, Ts...>;

            hpx::move_only_function<result_type()> f_wrapper =
                hpx::util::deferred_call(
                    HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
            auto t = std::make_shared<
                async_execute_wrapper_helper<decltype(f_wrapper), result_type>>(
                HPX_MOVE(f_wrapper));

#if defined(HPX_COMPUTE_HOST_CODE)
            exec.pool_->get_io_service().post(hpx::bind_front(
                &async_execute_wrapper_helper<decltype(f_wrapper),
                    result_type>::invoke,
                t));
#else
            HPX_ASSERT_MSG(
                false, "Attempting to use io_service_pool in device code");
#endif

            return t->p_.get_future();
        }

        template <typename F, typename Shape, typename... Ts>
        friend auto tag_invoke(hpx::parallel::execution::bulk_async_execute_t,
            service_executor const& exec, F&& f, Shape const& shape, Ts&&... ts)
        {
            std::vector<
                hpx::future<detail::bulk_function_result_t<F, Shape, Ts...>>>
                results;
            results.reserve(hpx::util::size(shape));

            for (auto const& elem : shape)
            {
                results.push_back(hpx::parallel::execution::async_execute(
                    exec, HPX_FORWARD(F, f), elem, ts...));
            }

            return results;
        }

        // This has to be specialized for service executors. bulk_then_execute
        // spawns an intermediate continuation which then spawns the bulk
        // continuations. The intermediate task must be allowed to yield to wait
        // for the bulk continuations. Because of this the intermediate task is
        // spawned on the current thread pool, not the service pool.
        template <typename F, typename Shape, typename Future, typename... Ts>
        friend auto tag_invoke(hpx::parallel::execution::bulk_then_execute_t,
            service_executor const& exec, F&& f, Shape const& shape,
            Future&& predecessor, Ts&&... ts)
        {
            auto func = parallel::execution::detail::
                make_fused_bulk_async_execute_helper(exec, HPX_FORWARD(F, f),
                    shape, hpx::make_tuple(HPX_FORWARD(Ts, ts)...));
            using vector_result_type =
                parallel::execution::detail::bulk_then_execute_result_t<F,
                    Shape, Future, Ts...>;
            using result_future_type = hpx::future<vector_result_type>;
            using shared_state_type =
                hpx::traits::detail::shared_state_ptr_t<vector_result_type>;

            auto exec_current = hpx::this_thread::get_executor();
            shared_state_type p =
                lcos::detail::make_continuation_exec<vector_result_type>(
                    HPX_FORWARD(Future, predecessor), exec_current,
                    [func = HPX_MOVE(func)](
                        auto&& predecessor) mutable -> vector_result_type {
                        return hpx::unwrap(func(HPX_MOVE(predecessor)));
                    });

            return hpx::traits::future_access<result_future_type>::create(
                HPX_MOVE(p));
        }

    private:
        template <typename F, typename Result>
        struct async_execute_wrapper_helper
        {
            explicit async_execute_wrapper_helper(F&& f)
              : f_(HPX_MOVE(f))
            {
            }

            void invoke()
            {
                hpx::detail::try_catch_exception_ptr(
                    [&]() {
                        if constexpr (std::is_void_v<Result>)
                        {
                            f_();
                            p_.set_value();
                        }
                        else
                        {
                            p_.set_value(f_());
                        }
                    },
                    [&](std::exception_ptr ep) {
                        p_.set_exception(HPX_MOVE(ep));
                    });
            }

            F f_;
            hpx::promise<Result> p_;
        };

        template <typename F>
        struct post_wrapper_helper
        {
            explicit post_wrapper_helper(F&& f)
              : f_(HPX_MOVE(f))
            {
            }

            void invoke()
            {
                f_();
            }

            F f_;
        };

#if defined(HPX_COMPUTE_HOST_CODE)
    private:
        hpx::util::io_service_pool* pool_;
#endif
    };
}    // namespace hpx::parallel::execution::detail
