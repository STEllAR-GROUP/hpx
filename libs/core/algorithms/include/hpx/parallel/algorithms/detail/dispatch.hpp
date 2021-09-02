//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c) 2021 Giannis Gonidelis
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/local/config.hpp>
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
    struct local_algorithm_result<std::pair<Result1, Result2>>
    {
        using type1 = typename hpx::traits::segmented_local_iterator_traits<
            Result1>::local_raw_iterator;
        using type2 = typename hpx::traits::segmented_local_iterator_traits<
            Result2>::local_raw_iterator;

        using type = std::pair<type1, type2>;
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

    template <typename Result1, typename Result2, typename Result3>
    struct local_algorithm_result<hpx::tuple<Result1, Result2, Result3>>
    {
        using type1 = typename hpx::traits::segmented_local_iterator_traits<
            Result1>::local_raw_iterator;
        using type2 = typename hpx::traits::segmented_local_iterator_traits<
            Result2>::local_raw_iterator;
        using type3 = typename hpx::traits::segmented_local_iterator_traits<
            Result3>::local_raw_iterator;

        using type = hpx::tuple<type1, type2, type3>;
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

    ///////////////////////////////////////////////////////////////////////////
    template <typename Derived, typename Result = void>
    struct algorithm
    {
    private:
        Derived const& derived() const
        {
            return static_cast<Derived const&>(*this);
        }

    public:
        using result_type = Result;
        using local_result_type =
            typename local_algorithm_result<result_type>::type;

        explicit algorithm(char const* const name)
          : name_(name)
        {
        }

        ///////////////////////////////////////////////////////////////////////
        // this equivalent to sequential execution
        template <typename ExPolicy, typename... Args>
        HPX_HOST_DEVICE
            typename parallel::util::detail::algorithm_result<ExPolicy,
                local_result_type>::type
            operator()(ExPolicy&& policy, Args&&... args) const
        {
#if !defined(__CUDA_ARCH__)
            try
            {
#endif
                using parameters_type = typename std::decay<
                    ExPolicy>::type::executor_parameters_type;
                using executor_type =
                    typename std::decay<ExPolicy>::type::executor_type;

                parallel::util::detail::scoped_executor_parameters_ref<
                    parameters_type, executor_type>
                    scoped_param(policy.parameters(), policy.executor());

                return parallel::util::detail::
                    algorithm_result<ExPolicy, local_result_type>::get(
                        Derived::sequential(std::forward<ExPolicy>(policy),
                            std::forward<Args>(args)...));
#if !defined(__CUDA_ARCH__)
            }
            catch (...)
            {
                // this does not return
                return detail::handle_exception<ExPolicy,
                    local_result_type>::call();
            }
#endif
        }

    protected:
        ///////////////////////////////////////////////////////////////////////
        template <typename ExPolicy, typename... Args>
        constexpr typename parallel::util::detail::algorithm_result<ExPolicy,
            local_result_type>::type
        call_execute(ExPolicy&& policy, std::false_type, Args&&... args) const
        {
            using result = parallel::util::detail::algorithm_result<ExPolicy,
                local_result_type>;

            return result::get(execution::sync_execute(policy.executor(),
                derived(), std::forward<ExPolicy>(policy),
                std::forward<Args>(args)...));
        }

        template <typename ExPolicy, typename... Args>
        constexpr
            typename parallel::util::detail::algorithm_result<ExPolicy>::type
            call_execute(
                ExPolicy&& policy, std::true_type, Args&&... args) const
        {
            execution::sync_execute(policy.executor(), derived(),
                std::forward<ExPolicy>(policy), std::forward<Args>(args)...);

            return parallel::util::detail::algorithm_result<ExPolicy>::get();
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename ExPolicy, typename... Args>
        typename parallel::util::detail::algorithm_result<ExPolicy,
            local_result_type>::type
        call_sequential(ExPolicy&& policy, Args&&... args) const
        {
            try
            {
                // run the launched task on the requested executor
                hpx::future<local_result_type> result =
                    execution::async_execute(policy.executor(), derived(),
                        std::forward<ExPolicy>(policy),
                        std::forward<Args>(args)...);

                return parallel::util::detail::algorithm_result<ExPolicy,
                    local_result_type>::get(std::move(result));
            }
            catch (std::bad_alloc const& ba)
            {
                throw ba;
            }
            catch (...)
            {
                return detail::handle_exception<ExPolicy,
                    local_result_type>::call();
            }
        }

    public:
        ///////////////////////////////////////////////////////////////////////
        // main sequential dispatch entry points

        // specialization for all task-based (asynchronous) execution policies
        // clang-format off
        template <typename ExPolicy, typename... Args,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_async_execution_policy_v<std::decay_t<ExPolicy>>
            )>
        // clang-format on
        constexpr typename parallel::util::detail::algorithm_result<ExPolicy,
            local_result_type>::type
        call2(ExPolicy&& policy, std::true_type, Args&&... args) const
        {
            return call_sequential(
                std::forward<ExPolicy>(policy), std::forward<Args>(args)...);
        }

        // clang-format off
        template <typename ExPolicy, typename... Args,
            HPX_CONCEPT_REQUIRES_(
                !hpx::is_async_execution_policy_v<std::decay_t<ExPolicy>>
            )>
        // clang-format on
        typename parallel::util::detail::algorithm_result<ExPolicy,
            local_result_type>::type
        call2(ExPolicy&& policy, std::true_type, Args&&... args) const
        {
            try
            {
                using is_void = std::is_void<local_result_type>;
                return call_execute(std::forward<ExPolicy>(policy), is_void(),
                    std::forward<Args>(args)...);
            }
            catch (std::bad_alloc const& ba)
            {
                throw ba;
            }
            catch (...)
            {
                return detail::handle_exception<ExPolicy,
                    local_result_type>::call();
            }
        }

        // main parallel dispatch entry point
        template <typename ExPolicy, typename... Args>
        static constexpr
            typename parallel::util::detail::algorithm_result<ExPolicy,
                local_result_type>::type
            call2(ExPolicy&& policy, std::false_type, Args&&... args)
        {
            return Derived::parallel(
                std::forward<ExPolicy>(policy), std::forward<Args>(args)...);
        }

        template <typename ExPolicy, typename... Args>
        HPX_FORCEINLINE constexpr
            typename parallel::util::detail::algorithm_result<ExPolicy,
                local_result_type>::type
            call(ExPolicy&& policy, Args&&... args)
        {
            using is_seq = hpx::is_sequenced_execution_policy<ExPolicy>;
            return call2(std::forward<ExPolicy>(policy), is_seq(),
                std::forward<Args>(args)...);
        }

#if defined(HPX_HAVE_CXX17_STD_EXECUTION_POLICES)
        // main dispatch entry points for std execution policies
        template <typename... Args>
        HPX_FORCEINLINE constexpr
            typename parallel::util::detail::algorithm_result<
                hpx::execution::sequenced_policy, local_result_type>::type
            call(std::execution::sequenced_policy, Args&&... args)
        {
            return call2(hpx::execution::seq, std::true_type(),
                std::forward<Args>(args)...);
        }

        template <typename... Args>
        HPX_FORCEINLINE constexpr
            typename parallel::util::detail::algorithm_result<
                hpx::execution::parallel_policy, local_result_type>::type
            call(std::execution::parallel_policy, Args&&... args)
        {
            return call2(hpx::execution::par, std::false_type(),
                std::forward<Args>(args)...);
        }

        template <typename... Args>
        HPX_FORCEINLINE constexpr
            typename parallel::util::detail::algorithm_result<
                hpx::execution::parallel_unsequenced_policy,
                local_result_type>::type
            call(std::execution::parallel_unsequenced_policy, Args&&... args)
        {
            return call2(hpx::execution::par_unseq, std::false_type(),
                std::forward<Args>(args)...);
        }

#if defined(HPX_HAVE_CXX20_STD_EXECUTION_POLICES)
        template <typename... Args>
        HPX_FORCEINLINE constexpr
            typename parallel::util::detail::algorithm_result<
                hpx::execution::unsequenced_policy, local_result_type>::type
            call(std::execution::unsequenced_policy, Args&&... args)
        {
            return call2(hpx::execution::unseq, std::false_type(),
                std::forward<Args>(args)...);
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
