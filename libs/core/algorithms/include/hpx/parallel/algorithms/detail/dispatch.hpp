//  Copyright (c) 2007-2021 Hartmut Kaiser
//  Copyright (c) 2021 Giannis Gonidelis
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/algorithms/traits/segmented_iterator_traits.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/execution/executors/execution.hpp>
#include <hpx/executors/exception_list.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/detail/scoped_executor_parameters.hpp>
#include <hpx/parallel/util/result_types.hpp>
#include <hpx/serialization/serialization_fwd.hpp>

#if defined(HPX_HAVE_CXX17_STD_EXECUTION_POLICES)
#include <execution>
#endif
#include <string>
#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { inline namespace v1 { namespace detail {

    ///////////////////////////////////////////////////////////////////////////
    template <typename Result>
    struct local_algorithm_result
    {
        using type = typename hpx::traits::segmented_local_iterator_traits<
            Result>::local_raw_iterator;
    };

    template <typename Result1, typename Result2>
    struct local_algorithm_result<util::in_out_result<Result1, Result2>>
    {
        using type1 = typename hpx::traits::segmented_local_iterator_traits<
            Result1>::local_raw_iterator;
        using type2 = typename hpx::traits::segmented_local_iterator_traits<
            Result2>::local_raw_iterator;

        using type = util::in_out_result<type1, type2>;
    };

    template <typename Result>
    struct local_algorithm_result<util::min_max_result<Result>>
    {
        using type1 = typename hpx::traits::segmented_local_iterator_traits<
            Result>::local_raw_iterator;

        using type = util::min_max_result<type1>;
    };

    template <typename Result1, typename Result2, typename Result3>
    struct local_algorithm_result<
        util::in_in_out_result<Result1, Result2, Result3>>
    {
        using type1 = typename hpx::traits::segmented_local_iterator_traits<
            Result1>::local_raw_iterator;
        using type2 = typename hpx::traits::segmented_local_iterator_traits<
            Result2>::local_raw_iterator;
        using type3 = typename hpx::traits::segmented_local_iterator_traits<
            Result3>::local_raw_iterator;

        using type = util::in_in_out_result<type1, type2, type3>;
    };

    template <>
    struct local_algorithm_result<void>
    {
        using type = void;
    };

    template <typename T>
    using local_algorithm_result_t = typename local_algorithm_result<T>::type;

    ///////////////////////////////////////////////////////////////////////////
    template <typename Derived, typename Result = void>
    struct algorithm
    {
    private:
        constexpr Derived const& derived() const noexcept
        {
            return static_cast<Derived const&>(*this);
        }

    public:
        using result_type = Result;
        using local_result_type = local_algorithm_result_t<result_type>;

        explicit constexpr algorithm(char const* const name) noexcept
          : name_(name)
        {
        }

        ///////////////////////////////////////////////////////////////////////
        // this equivalent to sequential execution
        template <typename ExPolicy, typename... Args>
        HPX_HOST_DEVICE hpx::parallel::util::detail::algorithm_result_t<
            ExPolicy, local_result_type>
        operator()(ExPolicy&& policy, Args&&... args) const
        {
#if !defined(__CUDA_ARCH__)
            try
            {
#endif
                using parameters_type =
                    typename std::decay_t<ExPolicy>::executor_parameters_type;
                using executor_type =
                    typename std::decay_t<ExPolicy>::executor_type;

                hpx::parallel::util::detail::scoped_executor_parameters_ref<
                    parameters_type, executor_type>
                    scoped_param(policy.parameters(), policy.executor());

                return hpx::parallel::util::detail::
                    algorithm_result<ExPolicy, local_result_type>::get(
                        Derived::sequential(HPX_FORWARD(ExPolicy, policy),
                            HPX_FORWARD(Args, args)...));
#if !defined(__CUDA_ARCH__)
            }
            catch (...)
            {
                // this does not return
                return hpx::parallel::v1::detail::handle_exception<ExPolicy,
                    local_result_type>::call();
            }
#endif
        }

    public:
        ///////////////////////////////////////////////////////////////////////
        // main sequential dispatch entry points

        template <typename ExPolicy, typename... Args>
        constexpr hpx::parallel::util::detail::algorithm_result_t<ExPolicy,
            local_result_type>
        call2(ExPolicy&& policy, std::true_type, Args&&... args) const
        {
            using result_handler =
                hpx::parallel::util::detail::algorithm_result<ExPolicy,
                    local_result_type>;

            auto exec = policy.executor();    // avoid move after use
            if constexpr (hpx::is_async_execution_policy_v<
                              std::decay_t<ExPolicy>>)
            {
                // specialization for all task-based (asynchronous) execution
                // policies

                // run the launched task on the requested executor
                return result_handler::get(execution::async_execute(
                    HPX_MOVE(exec), derived(), HPX_FORWARD(ExPolicy, policy),
                    HPX_FORWARD(Args, args)...));
            }
            else if constexpr (std::is_void_v<local_result_type>)
            {
                execution::sync_execute(HPX_MOVE(exec), derived(),
                    HPX_FORWARD(ExPolicy, policy), HPX_FORWARD(Args, args)...);
                return result_handler::get();
            }
            else
            {
                return result_handler::get(execution::sync_execute(
                    HPX_MOVE(exec), derived(), HPX_FORWARD(ExPolicy, policy),
                    HPX_FORWARD(Args, args)...));
            }
        }

        // main parallel dispatch entry point
        template <typename ExPolicy, typename... Args>
        HPX_FORCEINLINE static constexpr hpx::parallel::util::detail::
            algorithm_result_t<ExPolicy, local_result_type>
            call2(ExPolicy&& policy, std::false_type, Args&&... args)
        {
            return Derived::parallel(
                HPX_FORWARD(ExPolicy, policy), HPX_FORWARD(Args, args)...);
        }

        template <typename ExPolicy, typename... Args>
        HPX_FORCEINLINE constexpr hpx::parallel::util::detail::
            algorithm_result_t<ExPolicy, local_result_type>
            call(ExPolicy&& policy, Args&&... args) const
        {
            using is_seq = hpx::is_sequenced_execution_policy<ExPolicy>;
            return call2(HPX_FORWARD(ExPolicy, policy), is_seq(),
                HPX_FORWARD(Args, args)...);
        }

#if defined(HPX_HAVE_CXX17_STD_EXECUTION_POLICES)
        // main dispatch entry points for std execution policies
        template <typename... Args>
        HPX_FORCEINLINE constexpr hpx::parallel::util::detail::
            algorithm_result_t<hpx::execution::sequenced_policy,
                local_result_type>
            call(std::execution::sequenced_policy, Args&&... args) const
        {
            return call2(hpx::execution::seq, std::true_type(),
                HPX_FORWARD(Args, args)...);
        }

        template <typename... Args>
        HPX_FORCEINLINE constexpr hpx::parallel::util::detail::
            algorithm_result_t<hpx::execution::parallel_policy,
                local_result_type>
            call(std::execution::parallel_policy, Args&&... args) const
        {
            return call2(hpx::execution::par, std::false_type(),
                HPX_FORWARD(Args, args)...);
        }

        template <typename... Args>
        HPX_FORCEINLINE constexpr hpx::parallel::util::detail::
            algorithm_result_t<hpx::execution::parallel_unsequenced_policy,
                local_result_type>
            call(std::execution::parallel_unsequenced_policy,
                Args&&... args) const
        {
            return call2(hpx::execution::par_unseq, std::false_type(),
                HPX_FORWARD(Args, args)...);
        }

#if defined(HPX_HAVE_CXX20_STD_EXECUTION_POLICES)
        template <typename... Args>
        HPX_FORCEINLINE constexpr hpx::parallel::util::detail::
            algorithm_result_t<hpx::execution::unsequenced_policy,
                local_result_type>
            call(std::execution::unsequenced_policy, Args&&... args) const
        {
            return call2(hpx::execution::unseq, std::false_type(),
                HPX_FORWARD(Args, args)...);
        }
#endif
#endif

    private:
        char const* const name_;

        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive&, unsigned int)
        {
            // no need to serialize 'name_' as it is always initialized by the
            // constructor
        }
    };
}}}}    // namespace hpx::parallel::v1::detail
