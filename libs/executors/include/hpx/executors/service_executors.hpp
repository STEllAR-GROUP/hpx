//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/service_executors.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/execution/executors/execution.hpp>
#include <hpx/execution/executors/fused_bulk_execute.hpp>
#include <hpx/execution/executors/static_chunk_size.hpp>
#include <hpx/execution/traits/executor_traits.hpp>
#include <hpx/execution_base/execution.hpp>
#include <hpx/executors/current_executor.hpp>
#include <hpx/functional/bind_front.hpp>
#include <hpx/functional/deferred_call.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/futures/traits/future_traits.hpp>
#include <hpx/io_service/io_service_pool.hpp>
#include <hpx/lcos_local/promise.hpp>
#include <hpx/pack_traversal/unwrap.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/runtime_fwd.hpp>

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
        using execution_category = sequenced_execution_tag;

        /// Associate the static_chunk_size executor parameters type as a default
        /// with this executor.
        typedef static_chunk_size executor_parameters_type;

        service_executor(hpx::util::io_service_pool* pool)
          : pool_(pool)
        {
            HPX_ASSERT(pool_);
        }

        template <typename F, typename... Ts>
        void post(F&& f, Ts&&... ts) const
        {
            typedef typename hpx::util::detail::invoke_deferred_result<F,
                Ts...>::type result_type;

            hpx::util::unique_function_nonser<result_type()> f_wrapper =
                hpx::util::deferred_call(
                    std::forward<F>(f), std::forward<Ts>(ts)...);
            auto t = std::make_shared<post_wrapper_helper<decltype(f_wrapper)>>(
                std::move(f_wrapper));
            pool_->get_io_service().post(hpx::util::bind_front(
                &post_wrapper_helper<decltype(f_wrapper)>::invoke,
                std::move(t)));
        }

        template <typename F, typename... Ts>
        hpx::future<
            typename hpx::util::detail::invoke_deferred_result<F, Ts...>::type>
        async_execute(F&& f, Ts&&... ts) const
        {
            typedef typename hpx::util::detail::invoke_deferred_result<F,
                Ts...>::type result_type;

            hpx::util::unique_function_nonser<result_type()> f_wrapper =
                hpx::util::deferred_call(
                    std::forward<F>(f), std::forward<Ts>(ts)...);
            auto t = std::make_shared<
                async_execute_wrapper_helper<decltype(f_wrapper), result_type>>(
                std::move(f_wrapper));
            pool_->get_io_service().post(hpx::util::bind_front(
                &async_execute_wrapper_helper<decltype(f_wrapper),
                    result_type>::invoke,
                t));

            return t->p_.get_future();
        }

        template <typename F, typename Shape, typename... Ts>
        std::vector<hpx::lcos::future<
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
                    async_execute(std::forward<F>(f), elem, ts...));
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
            using func_result_type =
                typename parallel::execution::detail::then_bulk_function_result<
                    F, Shape, Future, Ts...>::type;
            using result_type =
                std::vector<hpx::lcos::future<func_result_type>>;

            auto func = parallel::execution::detail::
                make_fused_bulk_async_execute_helper<result_type>(*this,
                    std::forward<F>(f), shape,
                    hpx::util::make_tuple(std::forward<Ts>(ts)...));
            using vector_result_type =
                typename parallel::execution::detail::bulk_then_execute_result<
                    F, Shape, Future, Ts...>::type;
            using result_future_type = hpx::future<vector_result_type>;
            using shared_state_type =
                typename hpx::traits::detail::shared_state_ptr<
                    vector_result_type>::type;
            using future_type = typename std::decay<Future>::type;

            current_executor exec_current = hpx::this_thread::get_executor();
            shared_state_type p =
                lcos::detail::make_continuation_exec<vector_result_type>(
                    std::forward<Future>(predecessor), exec_current,
                    [func = std::move(func)](future_type&& predecessor) mutable
                    -> vector_result_type {
                        return hpx::util::unwrap(func(std::move(predecessor)));
                    });

            return hpx::traits::future_access<result_future_type>::create(
                std::move(p));
        }

    private:
        template <typename F, typename Result>
        struct async_execute_wrapper_helper
        {
            async_execute_wrapper_helper(F&& f)
              : f_(std::move(f))
            {
            }

            void invoke()
            {
                std::exception_ptr p;

                try
                {
                    invoke_helper(std::is_void<Result>());
                    return;
                }
                catch (...)
                {
                    p = std::current_exception();
                }

                // The exception is set outside the catch block since
                // set_exception may yield. Ending the catch block on a
                // different worker thread than where it was started may lead
                // to segfaults.
                p_.set_exception(std::move(p));
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
            hpx::lcos::local::promise<Result> p_;
        };

        template <typename F>
        struct post_wrapper_helper
        {
            post_wrapper_helper(F&& f)
              : f_(std::move(f))
            {
            }

            void invoke()
            {
                f_();
            }

            F f_;
        };

    private:
        hpx::util::io_service_pool* pool_;
    };
}}}}    // namespace hpx::parallel::execution::detail
