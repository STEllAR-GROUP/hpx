//  Copyright (c) 2007-2016 Hartmut Kaiser
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

namespace hpx { namespace parallel { namespace execution { namespace detail {
    class service_executor
    {
    public:
        /// Associate the parallel_execution_tag executor tag type as a default
        /// with this executor.
        using execution_category = hpx::execution::sequenced_execution_tag;

        /// Associate the static_chunk_size executor parameters type as a default
        /// with this executor.
        using executor_parameters_type = hpx::execution::static_chunk_size;

        service_executor(hpx::util::io_service_pool* pool)
#if defined(HPX_COMPUTE_HOST_CODE)
          : pool_(pool)
#endif
        {
            (void) pool;
            HPX_ASSERT(pool);
        }

        template <typename F, typename... Ts>
        void post(F&& f, Ts&&... ts) const
        {
            typedef typename hpx::util::detail::invoke_deferred_result<F,
                Ts...>::type result_type;

            hpx::move_only_function<result_type()> f_wrapper =
                hpx::util::deferred_call(
                    HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
            auto t = std::make_shared<post_wrapper_helper<decltype(f_wrapper)>>(
                HPX_MOVE(f_wrapper));
#if defined(HPX_COMPUTE_HOST_CODE)
            pool_->get_io_service().post(hpx::bind_front(
                &post_wrapper_helper<decltype(f_wrapper)>::invoke,
                HPX_MOVE(t)));
#else
            HPX_ASSERT_MSG(
                false, "Attempting to use io_service_pool in device code");
#endif
        }

        template <typename F, typename... Ts>
        hpx::future<
            typename hpx::util::detail::invoke_deferred_result<F, Ts...>::type>
        async_execute(F&& f, Ts&&... ts) const
        {
            typedef typename hpx::util::detail::invoke_deferred_result<F,
                Ts...>::type result_type;

            hpx::move_only_function<result_type()> f_wrapper =
                hpx::util::deferred_call(
                    HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
            auto t = std::make_shared<
                async_execute_wrapper_helper<decltype(f_wrapper), result_type>>(
                HPX_MOVE(f_wrapper));
#if defined(HPX_COMPUTE_HOST_CODE)
            pool_->get_io_service().post(hpx::bind_front(
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
        std::vector<hpx::future<
            typename detail::bulk_function_result<F, Shape, Ts...>::type>>
        bulk_async_execute(F&& f, Shape const& shape, Ts&&... ts) const
        {
            std::vector<hpx::future<
                typename detail::bulk_function_result<F, Shape, Ts...>::type>>
                results;
            results.reserve(hpx::util::size(shape));

            for (auto const& elem : shape)
            {
                results.push_back(
                    async_execute(HPX_FORWARD(F, f), elem, ts...));
            }

            return results;
        }

        // This has to be specialized for service executors.
        // bulk_then_execute spawns an intermediate continuation which then
        // spawns the bulk continuations. The intermediate task must be
        // allowed to yield to wait for the bulk continuations. Because of
        // this the intermediate task is spawned on the current thread
        // pool, not the service pool.
        template <typename F, typename Shape, typename Future, typename... Ts>
        hpx::future<typename parallel::execution::detail::
                bulk_then_execute_result<F, Shape, Future, Ts...>::type>
        bulk_then_execute(
            F&& f, Shape const& shape, Future&& predecessor, Ts&&... ts)
        {
            auto func = parallel::execution::detail::
                make_fused_bulk_async_execute_helper(*this, HPX_FORWARD(F, f),
                    shape, hpx::make_tuple(HPX_FORWARD(Ts, ts)...));
            using vector_result_type =
                typename parallel::execution::detail::bulk_then_execute_result<
                    F, Shape, Future, Ts...>::type;
            using result_future_type = hpx::future<vector_result_type>;
            using shared_state_type =
                typename hpx::traits::detail::shared_state_ptr<
                    vector_result_type>::type;
            using future_type = typename std::decay<Future>::type;

            auto exec_current = hpx::this_thread::get_executor();
            shared_state_type p =
                lcos::detail::make_continuation_exec<vector_result_type>(
                    HPX_FORWARD(Future, predecessor), exec_current,
                    [func = HPX_MOVE(func)](future_type&& predecessor) mutable
                    -> vector_result_type {
                        return hpx::unwrap(func(HPX_MOVE(predecessor)));
                    });

            return hpx::traits::future_access<result_future_type>::create(
                HPX_MOVE(p));
        }

    private:
        template <typename F, typename Result>
        struct async_execute_wrapper_helper
        {
            async_execute_wrapper_helper(F&& f)
              : f_(HPX_MOVE(f))
            {
            }

            void invoke()
            {
                hpx::detail::try_catch_exception_ptr(
                    [&]() { invoke_helper(std::is_void<Result>()); },
                    [&](std::exception_ptr ep) {
                        p_.set_exception(HPX_MOVE(ep));
                    });
            }

            void invoke_helper(std::true_type)
            {
                f_();
                p_.set_value();
            }
            void invoke_helper(std::false_type)
            {
                p_.set_value(f_());
            }

            F f_;
            hpx::promise<Result> p_;
        };

        template <typename F>
        struct post_wrapper_helper
        {
            post_wrapper_helper(F&& f)
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
}}}}    // namespace hpx::parallel::execution::detail
