//  Copyright (c)      2025 Aditya Sapra
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

// Required HPX execution headers
#include <hpx/async_cuda/target.hpp>
#include <hpx/execution/executors/execution_parameters.hpp>
#include <hpx/execution/executors/rebind_executor.hpp>
#include <hpx/execution/traits/is_execution_policy.hpp>
#include <hpx/execution_base/traits/is_executor.hpp>
#include <hpx/execution_base/traits/is_executor_parameters.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/executors/parallel_executor.hpp>
#include <hpx/executors/sequenced_executor.hpp>
#include <hpx/modules/execution_base.hpp>
#include <hpx/threading_base/execution_agent.hpp>
#include <thrust/execution_policy.h>

#include <memory>
#include <thrust/system/cuda/execution_policy.h>
#include <type_traits>

namespace hpx::thrust {

    struct thrust_task_policy;    //for async ops
    template <typename Executor, typename Parameters>
    struct thrust_task_policy_shim;

    struct thrust_policy;
    template <typename Executor, typename Parameters>
    struct thrust_policy_shim;

    struct thrust_host_policy;
    struct thrust_device_policy;

    struct thrust_task_policy
    {
        using executor_type = hpx::execution::parallel_executor;
        using executor_parameters_type =
            hpx::execution::experimental::extract_executor_parameters<
                executor_type>::type;
        using execution_category = hpx::execution::parallel_execution_tag;

        template <typename Executor_, typename Parameters_>
        struct rebind
        {
            using type = thrust_task_policy_shim<Executor_, Parameters_>;
        };

        constexpr thrust_task_policy() {}

        thrust_task_policy operator()(
            hpx::execution::experimental::to_task_t) const
        {
            return *this;
        }

        thrust_task_policy_shim<executor_type, executor_parameters_type> on(
            hpx::cuda::experimental::target const& t) const;

        // HPX execution policy interface
        executor_type executor() const
        {
            return executor_type{};
        }

        executor_parameters_type& parameters()
        {
            return params_;
        }
        constexpr executor_parameters_type const& parameters() const
        {
            return params_;
        }

        // Async helpers with default-target fallback for base policy
        bool has_target() const
        {
            return false;
        }

        hpx::cuda::experimental::target const& target_or_default() const
        {
            return hpx::cuda::experimental::get_default_target();
        }

        cudaStream_t stream() const
        {
            return target_or_default().native_handle().get_stream();
        }

        auto get() const
        {
            return ::thrust::cuda::par_nosync.on(stream());
        }

        hpx::future<void> get_future() const
        {
            return target_or_default().get_future_with_event();
        }

    private:
        executor_parameters_type params_{};
    };

    template <typename Executor, typename Parameters>
    struct thrust_task_policy_shim : thrust_task_policy
    {
        using executor_type = Executor;
        using executor_parameters_type = Parameters;
        using execution_category =
            typename hpx::traits::executor_execution_category<
                executor_type>::type;

        template <typename Executor_, typename Parameters_>
        struct rebind
        {
            using type = thrust_task_policy_shim<Executor_, Parameters_>;
        };

        thrust_task_policy_shim operator()(
            hpx::execution::experimental::to_task_t) const
        {
            return *this;
        }

        // Bind a CUDA target explicitly for async GPU execution (returns a new shim)
        thrust_task_policy_shim on(
            hpx::cuda::experimental::target const& t) const
        {
            thrust_task_policy_shim copy = *this;
            copy.bound_target_ =
                std::make_shared<hpx::cuda::experimental::target>(t);
            return copy;
        }

        // Async helpers with default-target fallback
        bool has_target() const
        {
            return static_cast<bool>(bound_target_);
        }

        hpx::cuda::experimental::target const& target_or_default() const
        {
            return bound_target_ ?
                *bound_target_ :
                hpx::cuda::experimental::get_default_target();
        }

        cudaStream_t stream() const
        {
            return target_or_default().native_handle().get_stream();
        }

        auto get() const
        {
            return ::thrust::cuda::par_nosync.on(stream());
        }

        hpx::future<void> get_future() const
        {
            return target_or_default().get_future_with_event();
        }

        // HPX execution policy interface for shim
        Executor& executor()
        {
            return exec_;
        }
        Executor const& executor() const
        {
            return exec_;
        }

        Parameters& parameters()
        {
            return params_;
        }
        Parameters const& parameters() const
        {
            return params_;
        }

        template <typename Dependent = void,
            typename Enable = typename std::enable_if<
                std::is_constructible<Executor>::value &&
                    std::is_constructible<Parameters>::value,
                Dependent>::type>
        constexpr thrust_task_policy_shim()
        {
        }

        template <typename Executor_, typename Parameters_>
        constexpr thrust_task_policy_shim(
            Executor_&& exec, Parameters_&& params)
          : exec_(std::forward<Executor_>(exec))
          , params_(std::forward<Parameters_>(params))
        {
        }

        // Construct with an already bound CUDA target
        explicit thrust_task_policy_shim(
            std::shared_ptr<hpx::cuda::experimental::target> tgt)
          : bound_target_(std::move(tgt))
        {
        }

    private:
        Executor exec_{};
        Parameters params_{};
        std::shared_ptr<hpx::cuda::experimental::target> bound_target_{};
    };

    inline thrust_task_policy_shim<thrust_task_policy::executor_type,
        thrust_task_policy::executor_parameters_type>
    thrust_task_policy::on(hpx::cuda::experimental::target const& t) const
    {
        using shim_type =
            thrust_task_policy_shim<executor_type, executor_parameters_type>;
        return shim_type(std::make_shared<hpx::cuda::experimental::target>(t));
    }

    // Base thrust_policy
    struct thrust_policy
    {
        using executor_type = hpx::execution::parallel_executor;
        using executor_parameters_type =
            hpx::execution::experimental::extract_executor_parameters<
                executor_type>::type;
        using execution_category = hpx::execution::parallel_execution_tag;

        template <typename Executor_, typename Parameters_>
        struct rebind
        {
            using type = thrust_policy_shim<Executor_, Parameters_>;
        };

        constexpr thrust_policy() {}

        thrust_task_policy operator()(
            hpx::execution::experimental::to_task_t) const
        {
            return thrust_task_policy();
        }

        template <typename Executor_>
        typename hpx::execution::experimental::rebind_executor<thrust_policy,
            Executor_, executor_parameters_type>::type
        on(Executor_&& exec) const
        {
            using executor_type = typename std::decay<Executor_>::type;
            static_assert(hpx::traits::is_executor_any_v<executor_type>,
                "hpx::traits::is_executor_any_v<Executor>");

            using rebound_type =
                typename hpx::execution::experimental::rebind_executor<
                    thrust_policy, Executor_, executor_parameters_type>::type;
            return rebound_type(std::forward<Executor_>(exec), parameters());
        }

        template <typename... Parameters_,
            typename ParametersType = typename hpx::execution::experimental::
                executor_parameters_join<Parameters_...>::type>
        typename hpx::execution::experimental::rebind_executor<thrust_policy,
            executor_type, ParametersType>::type
        with(Parameters_&&... params) const
        {
            using rebound_type =
                typename hpx::execution::experimental::rebind_executor<
                    thrust_policy, executor_type, ParametersType>::type;
            return rebound_type(executor(),
                hpx::execution::experimental::join_executor_parameters(
                    std::forward<Parameters_>(params)...));
        }

        thrust_policy label(char const* l) const
        {
            auto p = *this;
            p.label_ = l;
            return p;
        }
        char const* label() const
        {
            return label_;
        }

        executor_type executor() const
        {
            return executor_type{};
        }

        executor_parameters_type& parameters()
        {
            return params_;
        }
        constexpr executor_parameters_type const& parameters() const
        {
            return params_;
        }

    private:
        executor_parameters_type params_{};
        char const* label_ = "unnamed kernel";
    };

    template <typename Executor, typename Parameters>
    struct thrust_policy_shim : thrust_policy
    {
        using executor_type = Executor;
        using executor_parameters_type = Parameters;
        using execution_category =
            typename hpx::traits::executor_execution_category<
                executor_type>::type;

        template <typename Executor_, typename Parameters_>
        struct rebind
        {
            using type = thrust_policy_shim<Executor_, Parameters_>;
        };

        thrust_task_policy_shim<Executor, Parameters> operator()(
            hpx::execution::experimental::to_task_t) const
        {
            return thrust_task_policy_shim<Executor, Parameters>(
                exec_, params_);
        }

        template <typename Executor_>
        typename hpx::execution::experimental::rebind_executor<
            thrust_policy_shim, Executor_, executor_parameters_type>::type
        on(Executor_&& exec) const
        {
            using executor_type = typename std::decay<Executor_>::type;
            static_assert(hpx::traits::is_executor_any_v<executor_type>,
                "hpx::traits::is_executor_any_v<Executor>");

            using rebound_type =
                typename hpx::execution::experimental::rebind_executor<
                    thrust_policy_shim, Executor_,
                    executor_parameters_type>::type;
            return rebound_type(std::forward<Executor_>(exec), parameters());
        }

        template <typename... Parameters_,
            typename ParametersType = typename hpx::execution::experimental::
                executor_parameters_join<Parameters_...>::type>
        typename hpx::execution::experimental::rebind_executor<
            thrust_policy_shim, executor_type, ParametersType>::type
        with(Parameters_&&... params) const
        {
            using rebound_type =
                typename hpx::execution::experimental::rebind_executor<
                    thrust_policy_shim, executor_type, ParametersType>::type;
            return rebound_type(executor(),
                hpx::execution::experimental::join_executor_parameters(
                    std::forward<Parameters_>(params)...));
        }

        thrust_policy_shim& label(char const* l)
        {
            label_ = l;
            return *this;
        }
        char const* label() const
        {
            return label_;
        }

        Executor& executor()
        {
            return exec_;
        }

        Executor const& executor() const
        {
            return exec_;
        }

        Parameters& parameters()
        {
            return params_;
        }

        Parameters const& parameters() const
        {
            return params_;
        }

        template <typename Dependent = void,
            typename Enable = typename std::enable_if<
                std::is_constructible<Executor>::value &&
                    std::is_constructible<Parameters>::value,
                Dependent>::type>
        constexpr thrust_policy_shim()
        {
        }

        template <typename Executor_, typename Parameters_>
        constexpr thrust_policy_shim(Executor_&& exec, Parameters_&& params)
          : exec_(std::forward<Executor_>(exec))
          , params_(std::forward<Parameters_>(params))
        {
        }

    private:
        Executor exec_;
        Parameters params_;
        char const* label_ = "unnamed kernel";
    };

    // Host-specific policy that inherits from thrust_policy
    struct thrust_host_policy : thrust_policy
    {
        constexpr thrust_host_policy() = default;

        // Return thrust::host execution policy
        constexpr auto get() const
        {
            return ::thrust::host;
        }
    };

    // Device-specific policy that inherits from thrust_policy
    struct thrust_device_policy : thrust_policy
    {
        constexpr thrust_device_policy() = default;

        // Return thrust::device execution policy
        constexpr auto get() const
        {
            return ::thrust::device;
        }
    };

    // Global policy instances
    inline constexpr thrust_host_policy thrust_host{};
    inline constexpr thrust_device_policy thrust_device{};

    // Legacy support - default thrust policy (keep for backward compatibility)
    static constexpr thrust_policy thrust;

    template <typename ExecutionPolicy>
    struct is_thrust_execution_policy : std::false_type
    {
    };

    template <>
    struct is_thrust_execution_policy<hpx::thrust::thrust_policy>
      : std::true_type
    {
    };

    template <typename Executor, typename Parameters>
    struct is_thrust_execution_policy<
        hpx::thrust::thrust_policy_shim<Executor, Parameters>>
      : std::true_type
    {
    };

    template <>
    struct is_thrust_execution_policy<
        hpx::thrust::thrust_host_policy> : std::true_type
    {
    };

    template <>
    struct is_thrust_execution_policy<
        hpx::thrust::thrust_device_policy> : std::true_type
    {
    };

    template <>
    struct is_thrust_execution_policy<
        hpx::thrust::thrust_task_policy> : std::true_type
    {
    };

    template <typename Executor, typename Parameters>
    struct is_thrust_execution_policy<
        hpx::thrust::thrust_task_policy_shim<Executor, Parameters>>
      : std::true_type
    {
    };

    template <typename T>
    inline constexpr bool is_thrust_execution_policy_v =
        is_thrust_execution_policy<T>::value;

    namespace detail {
        template <typename ExecutionPolicy, typename Enable = void>
        struct get_policy_result;

        template <typename ExecutionPolicy>
        struct get_policy_result<ExecutionPolicy,
            std::enable_if_t<hpx::is_async_execution_policy_v<
                std::decay_t<ExecutionPolicy>>>>
        {
            static_assert(is_thrust_execution_policy<
                              std::decay_t<ExecutionPolicy>>::value,
                "get_policy_result can only be used with Thrust execution "
                "policies");

            using type = hpx::future<void>;

            template <typename Future>
            static constexpr decltype(auto) call(Future&& future)
            {
                return std::forward<Future>(future);
            }
        };

        template <typename ExecutionPolicy>
        struct get_policy_result<ExecutionPolicy,
            std::enable_if_t<!hpx::is_async_execution_policy_v<
                std::decay_t<ExecutionPolicy>>>>
        {
            static_assert(is_thrust_execution_policy<
                              std::decay_t<ExecutionPolicy>>::value,
                "get_policy_result can only be used with Thrust execution "
                "policies");

            template <typename Future>
            static constexpr decltype(auto) call(Future&& future)
            {
                return std::forward<Future>(future).get();
            }
        };
    }    // namespace detail

}    // namespace hpx::thrust

namespace hpx::detail {
    template <typename Executor, typename Parameters>
    struct is_rebound_execution_policy<
        hpx::thrust::thrust_policy_shim<Executor, Parameters>>
      : std::true_type
    {
    };

    template <typename Executor, typename Parameters>
    struct is_rebound_execution_policy<
        hpx::thrust::thrust_task_policy_shim<Executor, Parameters>>
      : std::true_type
    {
    };

    template <>
    struct is_execution_policy<hpx::thrust::thrust_policy>
      : std::true_type
    {
    };

    template <typename Executor, typename Parameters>
    struct is_execution_policy<
        hpx::thrust::thrust_policy_shim<Executor, Parameters>>
      : std::true_type
    {
    };

    template <>
    struct is_execution_policy<hpx::thrust::thrust_host_policy>
      : std::true_type
    {
    };

    template <>
    struct is_execution_policy<hpx::thrust::thrust_device_policy>
      : std::true_type
    {
    };

    template <>
    struct is_execution_policy<hpx::thrust::thrust_task_policy>
      : std::true_type
    {
    };

    template <typename Executor, typename Parameters>
    struct is_execution_policy<
        hpx::thrust::thrust_task_policy_shim<Executor, Parameters>>
      : std::true_type
    {
    };

    template <>
    struct is_parallel_execution_policy<hpx::thrust::thrust_policy>
      : std::true_type
    {
    };

    template <typename Executor, typename Parameters>
    struct is_parallel_execution_policy<
        hpx::thrust::thrust_policy_shim<Executor, Parameters>>
      : std::true_type
    {
    };

    template <>
    struct is_parallel_execution_policy<
        hpx::thrust::thrust_host_policy> : std::true_type
    {
    };

    template <>
    struct is_parallel_execution_policy<
        hpx::thrust::thrust_device_policy> : std::true_type
    {
    };

    template <>
    struct is_parallel_execution_policy<
        hpx::thrust::thrust_task_policy> : std::true_type
    {
    };

    template <typename Executor, typename Parameters>
    struct is_parallel_execution_policy<
        hpx::thrust::thrust_task_policy_shim<Executor, Parameters>>
      : std::true_type
    {
    };

    template <>
    struct is_async_execution_policy<
        hpx::thrust::thrust_task_policy> : std::true_type
    {
    };

    template <typename Executor, typename Parameters>
    struct is_async_execution_policy<
        hpx::thrust::thrust_task_policy_shim<Executor, Parameters>>
      : std::true_type
    {
    };

}    // namespace hpx::detail