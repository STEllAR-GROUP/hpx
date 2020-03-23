//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/service_executors.hpp

#if !defined(HPX_PARALLEL_EXECUTORS_SERVICE_EXECUTORS_MAY_15_2015_0548PM)
#define HPX_PARALLEL_EXECUTORS_SERVICE_EXECUTORS_MAY_15_2015_0548PM

#include <hpx/config.hpp>
#include <hpx/assertion.hpp>
#include <hpx/execution/executors/current_executor.hpp>
#include <hpx/execution/executors/execution.hpp>
#include <hpx/execution/executors/execution_fwd.hpp>
#include <hpx/execution/executors/fused_bulk_execute.hpp>
#include <hpx/execution/executors/static_chunk_size.hpp>
#include <hpx/execution/traits/executor_traits.hpp>
#include <hpx/functional/bind_front.hpp>
#include <hpx/functional/deferred_call.hpp>
#include <hpx/io_service/io_service_pool.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/runtime_fwd.hpp>
#include <hpx/traits/future_traits.hpp>
#include <hpx/util/unwrap.hpp>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace parallel { namespace execution {
    namespace detail {
        class HPX_EXPORT service_executor
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
                auto t =
                    std::make_shared<post_wrapper_helper<decltype(f_wrapper)>>(
                        std::move(f_wrapper));
                pool_->get_io_service().post(hpx::util::bind_front(
                    &post_wrapper_helper<decltype(f_wrapper)>::invoke,
                    std::move(t)));
            }

            template <typename F, typename... Ts>
            hpx::future<typename hpx::util::detail::invoke_deferred_result<F,
                Ts...>::type>
            async_execute(F&& f, Ts&&... ts) const
            {
                typedef typename hpx::util::detail::invoke_deferred_result<F,
                    Ts...>::type result_type;

                hpx::util::unique_function_nonser<result_type()> f_wrapper =
                    hpx::util::deferred_call(
                        std::forward<F>(f), std::forward<Ts>(ts)...);
                auto t = std::make_shared<async_execute_wrapper_helper<
                    decltype(f_wrapper), result_type>>(std::move(f_wrapper));
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
                std::vector<hpx::future<typename detail::bulk_function_result<F,
                    Shape, Ts...>::type>>
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
            template <typename F, typename Shape, typename Future,
                typename... Ts>
            hpx::future<typename parallel::execution::detail::
                    bulk_then_execute_result<F, Shape, Future, Ts...>::type>
            bulk_then_execute(
                F&& f, Shape const& shape, Future&& predecessor, Ts&&... ts)
            {
                using func_result_type = typename parallel::execution::detail::
                    then_bulk_function_result<F, Shape, Future, Ts...>::type;
                using result_type =
                    std::vector<hpx::lcos::future<func_result_type>>;

                auto func = parallel::execution::detail::
                    make_fused_bulk_async_execute_helper<result_type>(*this,
                        std::forward<F>(f), shape,
                        hpx::util::make_tuple(std::forward<Ts>(ts)...));
                using vector_result_type =
                    typename parallel::execution::detail::
                        bulk_then_execute_result<F, Shape, Future, Ts...>::type;
                using result_future_type = hpx::future<vector_result_type>;
                using shared_state_type =
                    typename hpx::traits::detail::shared_state_ptr<
                        vector_result_type>::type;
                using future_type = typename std::decay<Future>::type;

                current_executor exec_current =
                    hpx::this_thread::get_executor();
                shared_state_type p =
                    lcos::detail::make_continuation_exec<vector_result_type>(
                        std::forward<Future>(predecessor), exec_current,
                        [func = std::move(func)](
                            future_type&& predecessor) mutable
                        -> vector_result_type {
                            return hpx::util::unwrap(
                                func(std::move(predecessor)));
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
                    try
                    {
                        invoke_helper(std::is_void<Result>());
                    }
                    catch (...)
                    {
                        p_.set_exception(std::current_exception());
                    }
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
    }    // namespace detail
    enum class service_executor_type
    {
        io_thread_pool,        ///< Selects creating a service executor using
                               ///< the I/O pool of threads
        parcel_thread_pool,    ///< Selects creating a service executor using
                               ///< the parcel pool of threads
        timer_thread_pool,     ///< Selects creating a service executor using
                               ///< the timer pool of threads
        main_thread            ///< Selects creating a service executor using
                               ///< the main thread
    };

    namespace detail {
        inline hpx::util::io_service_pool* get_service_pool(
            service_executor_type t, char const* name_suffix = "")
        {
            switch (t)
            {
            case service_executor_type::io_thread_pool:
                return get_thread_pool("io-pool");

            case service_executor_type::parcel_thread_pool:
            {
                char const* suffix = *name_suffix ? name_suffix : "-tcp";
                return get_thread_pool("parcel-pool", suffix);
            }

            case service_executor_type::timer_thread_pool:
                return get_thread_pool("timer-pool");

            case service_executor_type::main_thread:
                return get_thread_pool("main-pool");

            default:
                break;
            }

            HPX_THROW_EXCEPTION(bad_parameter,
                "hpx::threads::detail::get_service_pool",
                "unknown pool executor type");
            return nullptr;
        }
    }    // namespace detail

    struct service_executor : public detail::service_executor
    {
        service_executor(service_executor_type t, char const* name_suffix = "")
          : detail::service_executor(detail::get_service_pool(t, name_suffix))
        {
        }
    };

    struct io_pool_executor : public detail::service_executor
    {
        io_pool_executor()
          : detail::service_executor(
                detail::get_service_pool(service_executor_type::io_thread_pool))
        {
        }
    };

    struct parcel_pool_executor : public detail::service_executor
    {
        parcel_pool_executor(char const* name_suffix = "-tcp")
          : detail::service_executor(detail::get_service_pool(
                service_executor_type::parcel_thread_pool, name_suffix))
        {
        }
    };

    struct timer_pool_executor : public detail::service_executor
    {
        timer_pool_executor()
          : detail::service_executor(detail::get_service_pool(
                service_executor_type::timer_thread_pool))
        {
        }
    };

    struct main_pool_executor : public detail::service_executor
    {
        main_pool_executor()
          : detail::service_executor(
                detail::get_service_pool(service_executor_type::main_thread))
        {
        }
    };

    ///  \cond NOINTERNAL
    template <>
    struct is_one_way_executor<detail::service_executor> : std::true_type
    {
    };

    template <>
    struct is_two_way_executor<detail::service_executor> : std::true_type
    {
    };

    template <>
    struct is_bulk_two_way_executor<detail::service_executor> : std::true_type
    {
    };

    template <>
    struct is_one_way_executor<service_executor> : std::true_type
    {
    };

    template <>
    struct is_two_way_executor<service_executor> : std::true_type
    {
    };

    template <>
    struct is_bulk_two_way_executor<service_executor> : std::true_type
    {
    };

    template <>
    struct is_one_way_executor<io_pool_executor> : std::true_type
    {
    };

    template <>
    struct is_two_way_executor<io_pool_executor> : std::true_type
    {
    };

    template <>
    struct is_bulk_two_way_executor<io_pool_executor> : std::true_type
    {
    };

    template <>
    struct is_one_way_executor<parcel_pool_executor> : std::true_type
    {
    };

    template <>
    struct is_two_way_executor<parcel_pool_executor> : std::true_type
    {
    };

    template <>
    struct is_bulk_two_way_executor<parcel_pool_executor> : std::true_type
    {
    };

    template <>
    struct is_one_way_executor<timer_pool_executor> : std::true_type
    {
    };

    template <>
    struct is_two_way_executor<timer_pool_executor> : std::true_type
    {
    };

    template <>
    struct is_bulk_two_way_executor<timer_pool_executor> : std::true_type
    {
    };

    template <>
    struct is_one_way_executor<main_pool_executor> : std::true_type
    {
    };

    template <>
    struct is_two_way_executor<main_pool_executor> : std::true_type
    {
    };

    template <>
    struct is_bulk_two_way_executor<main_pool_executor> : std::true_type
    {
    };
    /// \endcond
}}}    // namespace hpx::parallel::execution

#endif
