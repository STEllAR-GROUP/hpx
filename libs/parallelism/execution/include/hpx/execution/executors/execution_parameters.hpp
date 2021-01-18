//  Copyright (c) 2016 Marcin Copik
//  Copyright (c) 2016-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/async_base/traits/is_launch_policy.hpp>
#include <hpx/concepts/has_member_xxx.hpp>
#include <hpx/execution/detail/execution_parameter_callbacks.hpp>
#include <hpx/execution/traits/is_executor.hpp>
#include <hpx/execution/traits/is_executor_parameters.hpp>
#include <hpx/functional/tag_invoke.hpp>
#include <hpx/preprocessor/cat.hpp>
#include <hpx/preprocessor/stringize.hpp>
#include <hpx/serialization/base_object.hpp>
#include <hpx/type_support/decay.hpp>
#include <hpx/type_support/detail/wrap_int.hpp>
#include <hpx/type_support/pack.hpp>

#include <hpx/execution/executors/execution.hpp>
#include <hpx/execution/executors/execution_parameters_fwd.hpp>

#include <cstddef>
#include <functional>
#include <type_traits>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parallel { namespace execution {

    namespace detail {
        /// \cond NOINTERNAL

        ///////////////////////////////////////////////////////////////////////
        // customization point for interface get_chunk_size()
        template <typename Parameters, typename Executor_>
        struct get_chunk_size_fn_helper<Parameters, Executor_,
            typename std::enable_if<
                hpx::traits::is_executor_any<Executor_>::value
#if defined(HPX_HAVE_THREAD_EXECUTORS_COMPATIBILITY)
                || hpx::traits::is_threads_executor<Executor_>::value
#endif
                >::type>
        {
            // Check whether the parameter object implements this function
            template <typename AnyParameters, typename Executor, typename F>
            HPX_FORCEINLINE static std::size_t call_param(
                hpx::traits::detail::wrap_int, AnyParameters&& /* params */,
                Executor&& /* exec */, F&& /* f */, std::size_t /* cores */,
                std::size_t /* num_tasks */)
            {
                // return zero for the chunk-size which will tell the
                // implementation to calculate the chunk size either based
                // on a specified maximum number of chunks or based on some
                // internal rule (if no maximum number of chunks was given)
                return 0;
            }

            template <typename AnyParameters, typename Executor, typename F>
            HPX_FORCEINLINE static auto call_param(int, AnyParameters&& params,
                Executor&& exec, F&& f, std::size_t cores,
                std::size_t num_tasks)
                -> decltype(params.get_chunk_size(std::forward<Executor>(exec),
                    std::forward<F>(f), cores, num_tasks))
            {
                return params.get_chunk_size(std::forward<Executor>(exec),
                    std::forward<F>(f), cores, num_tasks);
            }

            // Check whether this function is implemented by the executor
            template <typename AnyParameters, typename Executor, typename F>
            HPX_FORCEINLINE static std::size_t call(
                hpx::traits::detail::wrap_int, AnyParameters&& params,
                Executor&& exec, F&& f, std::size_t cores,
                std::size_t num_tasks)
            {
                return call_param(0, std::forward<AnyParameters>(params),
                    std::forward<Executor>(exec), std::forward<F>(f), cores,
                    num_tasks);
            }

            template <typename AnyParameters, typename Executor, typename F>
            HPX_FORCEINLINE static auto call(int, AnyParameters&& params,
                Executor&& exec, F&& f, std::size_t cores,
                std::size_t num_tasks)
                -> decltype(
                    exec.get_chunk_size(std::forward<AnyParameters>(params),
                        std::forward<F>(f), cores, num_tasks))
            {
                return exec.get_chunk_size(std::forward<AnyParameters>(params),
                    std::forward<F>(f), cores, num_tasks);
            }

            template <typename Executor, typename F>
            HPX_FORCEINLINE static std::size_t call(Parameters& params,
                Executor&& exec, F&& f, std::size_t cores,
                std::size_t num_tasks)
            {
                return call(0, params, std::forward<Executor>(exec),
                    std::forward<F>(f), cores, num_tasks);
            }

            template <typename AnyParameters, typename Executor, typename F>
            HPX_FORCEINLINE static std::size_t call(AnyParameters params,
                Executor&& exec, F&& f, std::size_t cores,
                std::size_t num_tasks)
            {
                return call(0, static_cast<Parameters&>(params),
                    std::forward<Executor>(exec), std::forward<F>(f), cores,
                    num_tasks);
            }

            template <typename AnyParameters, typename Executor, typename F>
            struct result
            {
                using type = decltype(call(std::declval<AnyParameters>(),
                    std::declval<Executor>(), std::declval<F>(),
                    std::declval<std::size_t>(), std::declval<std::size_t>()));
            };
        };

        HPX_HAS_MEMBER_XXX_TRAIT_DEF(get_chunk_size)

        ///////////////////////////////////////////////////////////////////////
        // customization point for interface maximal_number_of_chunks()
        template <typename Parameters, typename Executor_>
        struct maximal_number_of_chunks_fn_helper<Parameters, Executor_,
            typename std::enable_if<
                hpx::traits::is_executor_any<Executor_>::value
#if defined(HPX_HAVE_THREAD_EXECUTORS_COMPATIBILITY)
                || hpx::traits::is_threads_executor<Executor_>::value
#endif
                >::type>
        {
            // Check whether the parameter object implements this function
            template <typename AnyParameters, typename Executor>
            HPX_FORCEINLINE static std::size_t call_param(
                hpx::traits::detail::wrap_int, AnyParameters&&, Executor&&,
                std::size_t, std::size_t)
            {
                // return zero chunks which will tell the implementation to
                // calculate the number of chunks either based on a
                // specified chunk size or based on some internal rule (if no
                // chunk-size was given)
                return 0;
            }

            template <typename AnyParameters, typename Executor>
            HPX_FORCEINLINE static auto call_param(int, AnyParameters&& params,
                Executor&& exec, std::size_t cores, std::size_t num_tasks)
                -> decltype(params.maximal_number_of_chunks(
                    std::forward<Executor>(exec), cores, num_tasks))
            {
                return params.maximal_number_of_chunks(
                    std::forward<Executor>(exec), cores, num_tasks);
            }

            // Check whether this function is implemented by the executor
            template <typename AnyParameters, typename Executor>
            HPX_FORCEINLINE static std::size_t call(
                hpx::traits::detail::wrap_int, AnyParameters&& params,
                Executor&& exec, std::size_t cores, std::size_t num_tasks)
            {
                return call_param(0, std::forward<AnyParameters>(params),
                    std::forward<Executor>(exec), cores, num_tasks);
            }

            template <typename AnyParameters, typename Executor>
            HPX_FORCEINLINE static auto call(int, AnyParameters&& params,
                Executor&& exec, std::size_t cores, std::size_t num_tasks)
                -> decltype(exec.maximal_number_of_chunks(
                    std::forward<AnyParameters>(params), cores, num_tasks))
            {
                return exec.maximal_number_of_chunks(
                    std::forward<AnyParameters>(params), cores, num_tasks);
            }

            template <typename Executor>
            HPX_FORCEINLINE static std::size_t call(Parameters& params,
                Executor&& exec, std::size_t cores, std::size_t num_tasks)
            {
                return call(
                    0, params, std::forward<Executor>(exec), cores, num_tasks);
            }

            template <typename AnyParameters, typename Executor>
            HPX_FORCEINLINE static std::size_t call(AnyParameters params,
                Executor&& exec, std::size_t cores, std::size_t num_tasks)
            {
                return call(static_cast<Parameters&>(params),
                    std::forward<Executor>(exec), cores, num_tasks);
            }

            template <typename AnyParameters, typename Executor>
            struct result
            {
                using type = decltype(call(std::declval<AnyParameters>(),
                    std::declval<Executor>(), std::declval<std::size_t>(),
                    std::declval<std::size_t>()));
            };
        };

        HPX_HAS_MEMBER_XXX_TRAIT_DEF(maximal_number_of_chunks)

        ///////////////////////////////////////////////////////////////////////
        // customization point for interface reset_thread_distribution()
        template <typename Parameters, typename Executor_>
        struct reset_thread_distribution_fn_helper<Parameters, Executor_,
            typename std::enable_if<
                hpx::traits::is_executor_any<Executor_>::value
#if defined(HPX_HAVE_THREAD_EXECUTORS_COMPATIBILITY)
                || hpx::traits::is_threads_executor<Executor_>::value
#endif
                >::type>
        {
            // Check whether the parameter object implements this function
            template <typename AnyParameters, typename Executor>
            HPX_FORCEINLINE static void call_param(
                hpx::traits::detail::wrap_int, AnyParameters&&, Executor&&)
            {
            }

            template <typename AnyParameters, typename Executor>
            HPX_FORCEINLINE static auto call_param(
                int, AnyParameters&& params, Executor && /* exec */)
                -> decltype(params.reset_thread_distribution())
            {
                params.reset_thread_distribution();
            }

            // Check whether this function is implemented by the executor
            template <typename AnyParameters, typename Executor>
            HPX_FORCEINLINE static void call(hpx::traits::detail::wrap_int,
                AnyParameters&& params, Executor&& exec)
            {
                call_param(0, std::forward<AnyParameters>(params),
                    std::forward<Executor>(exec));
            }

            // Check whether this function is implemented by the executor
            template <typename AnyParameters, typename Executor>
            HPX_FORCEINLINE static auto call(
                int, AnyParameters&& params, Executor&& exec)
                -> decltype(exec.reset_thread_distribution(
                    std::forward<AnyParameters>(params)))
            {
                exec.reset_thread_distribution(
                    std::forward<AnyParameters>(params));
            }

            template <typename Executor>
            HPX_FORCEINLINE static void call(
                Parameters& params, Executor&& exec)
            {
                call(0, params, std::forward<Executor>(exec));
            }

            template <typename AnyParameters, typename Executor>
            HPX_FORCEINLINE static void call(
                AnyParameters params, Executor&& exec)
            {
                call(static_cast<Parameters&>(params),
                    std::forward<Executor>(exec));
            }

            template <typename AnyParameters, typename Executor>
            struct result
            {
                using type = decltype(call(
                    std::declval<AnyParameters>(), std::declval<Executor>()));
            };
        };

        HPX_HAS_MEMBER_XXX_TRAIT_DEF(reset_thread_distribution)

        ///////////////////////////////////////////////////////////////////////
        // customization point for interface processing_units_count()
        template <typename Parameters, typename Executor_>
        struct processing_units_count_fn_helper<Parameters, Executor_,
            typename std::enable_if<
                hpx::traits::is_executor_any<Executor_>::value>::type>
        {
            // Check whether the parameter object implements this function
            template <typename AnyParameters, typename Executor>
            HPX_FORCEINLINE static std::size_t call_param(
                hpx::traits::detail::wrap_int, AnyParameters&&, Executor&&)
            {
                return get_os_thread_count();
            }

            template <typename AnyParameters, typename Executor>
            HPX_FORCEINLINE static auto call_param(
                int, AnyParameters&& params, Executor&& exec)
                -> decltype(
                    params.processing_units_count(std::forward<Executor>(exec)))
            {
                return params.processing_units_count(
                    std::forward<Executor>(exec));
            }

            // Check whether this function is implemented by the executor
            template <typename AnyParameters, typename Executor>
            HPX_FORCEINLINE static std::size_t call(
                hpx::traits::detail::wrap_int, AnyParameters&& params,
                Executor&& exec)
            {
                return call_param(0, std::forward<AnyParameters>(params),
                    std::forward<Executor>(exec));
            }

            template <typename AnyParameters, typename Executor>
            HPX_FORCEINLINE static auto call(int, AnyParameters&& /* params */,
                Executor&& exec) -> decltype(exec.processing_units_count())
            {
                return exec.processing_units_count();
            }

            template <typename Executor>
            HPX_FORCEINLINE static std::size_t call(
                Parameters& params, Executor&& exec)
            {
                return call(0, params, std::forward<Executor>(exec));
            }

            template <typename AnyParameters, typename Executor>
            HPX_FORCEINLINE static std::size_t call(
                AnyParameters params, Executor&& exec)
            {
                return call(static_cast<Parameters&>(params),
                    std::forward<Executor>(exec));
            }

            template <typename AnyParameters, typename Executor>
            struct result
            {
                using type = decltype(call(
                    std::declval<AnyParameters>(), std::declval<Executor>()));
            };
        };

        HPX_HAS_MEMBER_XXX_TRAIT_DEF(processing_units_count)

        ///////////////////////////////////////////////////////////////////////
        // customization point for interface mark_begin_execution()
        template <typename Parameters, typename Executor_>
        struct mark_begin_execution_fn_helper<Parameters, Executor_,
            typename std::enable_if<
                hpx::traits::is_executor_any<Executor_>::value
#if defined(HPX_HAVE_THREAD_EXECUTORS_COMPATIBILITY)
                || hpx::traits::is_threads_executor<Executor_>::value
#endif
                >::type>
        {
            // Check whether the parameter object implements this function
            template <typename AnyParameters, typename Executor>
            HPX_FORCEINLINE static void call_param(
                hpx::traits::detail::wrap_int, AnyParameters&&, Executor&&)
            {
            }

            template <typename AnyParameters, typename Executor>
            HPX_FORCEINLINE static auto call_param(
                int, AnyParameters&& params, Executor&& exec)
                -> decltype(
                    params.mark_begin_execution(std::forward<Executor>(exec)))
            {
                params.mark_begin_execution(std::forward<Executor>(exec));
            }

            // Check whether this function is implemented by the executor
            template <typename AnyParameters, typename Executor>
            HPX_FORCEINLINE static void call(hpx::traits::detail::wrap_int,
                AnyParameters&& params, Executor&& exec)
            {
                call_param(0, std::forward<AnyParameters>(params),
                    std::forward<Executor>(exec));
            }

            template <typename AnyParameters, typename Executor>
            HPX_FORCEINLINE static auto call(
                int, AnyParameters&& params, Executor&& exec)
                -> decltype(exec.mark_begin_execution(
                    std::forward<AnyParameters>(params)))
            {
                exec.mark_begin_execution(std::forward<AnyParameters>(params));
            }

            template <typename Executor>
            HPX_FORCEINLINE static void call(
                Parameters& params, Executor&& exec)
            {
                call(0, params, std::forward<Executor>(exec));
            }

            template <typename AnyParameters, typename Executor>
            HPX_FORCEINLINE static void call(
                AnyParameters params, Executor&& exec)
            {
                call(static_cast<Parameters&>(params),
                    std::forward<Executor>(exec));
            }

            template <typename AnyParameters, typename Executor>
            struct result
            {
                using type = decltype(call(
                    std::declval<AnyParameters>(), std::declval<Executor>()));
            };
        };

        HPX_HAS_MEMBER_XXX_TRAIT_DEF(mark_begin_execution)

        ///////////////////////////////////////////////////////////////////////
        // customization point for interface mark_end_of_scheduling()
        template <typename Parameters, typename Executor_>
        struct mark_end_of_scheduling_fn_helper<Parameters, Executor_,
            typename std::enable_if<
                hpx::traits::is_executor_any<Executor_>::value
#if defined(HPX_HAVE_THREAD_EXECUTORS_COMPATIBILITY)
                || hpx::traits::is_threads_executor<Executor_>::value
#endif
                >::type>
        {
            // Check whether the parameter object implements this function
            template <typename AnyParameters, typename Executor>
            HPX_FORCEINLINE static void call_param(
                hpx::traits::detail::wrap_int, AnyParameters&&, Executor&&)
            {
            }

            template <typename AnyParameters, typename Executor>
            HPX_FORCEINLINE static auto call_param(
                int, AnyParameters&& params, Executor&& exec)
                -> decltype(
                    params.mark_end_of_scheduling(std::forward<Executor>(exec)))
            {
                params.mark_end_of_scheduling(std::forward<Executor>(exec));
            }

            // Check whether this function is implemented by the executor
            template <typename AnyParameters, typename Executor>
            HPX_FORCEINLINE static void call(hpx::traits::detail::wrap_int,
                AnyParameters&& params, Executor&& exec)
            {
                call_param(0, std::forward<AnyParameters>(params),
                    std::forward<Executor>(exec));
            }

            template <typename AnyParameters, typename Executor>
            HPX_FORCEINLINE static auto call(
                int, AnyParameters&& params, Executor&& exec)
                -> decltype(exec.mark_end_of_scheduling(
                    std::forward<AnyParameters>(params)))
            {
                exec.mark_end_of_scheduling(
                    std::forward<AnyParameters>(params));
            }

            template <typename Executor>
            HPX_FORCEINLINE static void call(
                Parameters& params, Executor&& exec)
            {
                call(0, params, std::forward<Executor>(exec));
            }

            template <typename AnyParameters, typename Executor>
            HPX_FORCEINLINE static void call(
                AnyParameters params, Executor&& exec)
            {
                call(static_cast<Parameters&>(params),
                    std::forward<Executor>(exec));
            }

            template <typename AnyParameters, typename Executor>
            struct result
            {
                using type = decltype(call(
                    std::declval<AnyParameters>(), std::declval<Executor>()));
            };
        };

        HPX_HAS_MEMBER_XXX_TRAIT_DEF(mark_end_of_scheduling)

        ///////////////////////////////////////////////////////////////////////
        // customization point for interface mark_end_execution()
        template <typename Parameters, typename Executor_>
        struct mark_end_execution_fn_helper<Parameters, Executor_,
            typename std::enable_if<
                hpx::traits::is_executor_any<Executor_>::value
#if defined(HPX_HAVE_THREAD_EXECUTORS_COMPATIBILITY)
                || hpx::traits::is_threads_executor<Executor_>::value
#endif
                >::type>
        {
            // Check whether the parameter object implements this function
            template <typename AnyParameters, typename Executor>
            HPX_FORCEINLINE static void call_param(
                hpx::traits::detail::wrap_int, AnyParameters&&, Executor&&)
            {
            }

            template <typename AnyParameters, typename Executor>
            HPX_FORCEINLINE static auto call_param(
                int, AnyParameters&& params, Executor&& exec)
                -> decltype(
                    params.mark_end_execution(std::forward<Executor>(exec)))
            {
                params.mark_end_execution(std::forward<Executor>(exec));
            }

            // Check whether this function is implemented by the executor
            template <typename AnyParameters, typename Executor>
            HPX_FORCEINLINE static void call(hpx::traits::detail::wrap_int,
                AnyParameters&& params, Executor&& exec)
            {
                call_param(0, std::forward<AnyParameters>(params),
                    std::forward<Executor>(exec));
            }

            template <typename AnyParameters, typename Executor>
            HPX_FORCEINLINE static auto call(
                int, AnyParameters&& params, Executor&& exec)
                -> decltype(exec.mark_end_execution(
                    std::forward<AnyParameters>(params)))
            {
                exec.mark_end_execution(std::forward<AnyParameters>(params));
            }

            template <typename Executor>
            HPX_FORCEINLINE static void call(
                Parameters& params, Executor&& exec)
            {
                call(0, params, std::forward<Executor>(exec));
            }

            template <typename AnyParameters, typename Executor>
            HPX_FORCEINLINE static void call(
                AnyParameters params, Executor&& exec)
            {
                call(static_cast<Parameters&>(params),
                    std::forward<Executor>(exec));
            }

            template <typename AnyParameters, typename Executor>
            struct result
            {
                using type = decltype(call(
                    std::declval<AnyParameters>(), std::declval<Executor>()));
            };
        };

        HPX_HAS_MEMBER_XXX_TRAIT_DEF(mark_end_execution)

        /// \endcond
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        /// \cond NOINTERNAL
        template <bool... Flags>
        struct parameters_type_counter;

        template <>
        struct parameters_type_counter<>
        {
            static constexpr int value = 0;
        };

        /// Return the number of parameters which are true
        template <bool Flag1, bool... Flags>
        struct parameters_type_counter<Flag1, Flags...>
        {
            static constexpr int value =
                Flag1 + parameters_type_counter<Flags...>::value;
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename T>
        struct unwrapper : T
        {
            // default constructor is needed for serialization purposes
            template <typename Dependent = void,
                typename Enable = typename std::enable_if<
                    std::is_constructible<T>::value, Dependent>::type>
            unwrapper()
              : T()
            {
            }

            // generic poor-man's forwarding constructor
            template <typename U,
                typename Enable = typename std::enable_if<!std::is_same<
                    typename std::decay<U>::type, unwrapper>::value>::type>
            unwrapper(U&& u)
              : T(std::forward<U>(u))
            {
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename T, typename Wrapper, typename Enable = void>
        struct maximal_number_of_chunks_call_helper
        {
        };

        template <typename T, typename Wrapper>
        struct maximal_number_of_chunks_call_helper<T, Wrapper,
            typename std::enable_if<
                has_maximal_number_of_chunks<T>::value>::type>
        {
            template <typename Executor>
            HPX_FORCEINLINE std::size_t maximal_number_of_chunks(
                Executor&& exec, std::size_t cores, std::size_t num_tasks)
            {
                auto& wrapped =
                    static_cast<unwrapper<Wrapper>*>(this)->member_.get();
                return wrapped.maximal_number_of_chunks(
                    std::forward<Executor>(exec), cores, num_tasks);
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename T, typename Wrapper, typename Enable = void>
        struct get_chunk_size_call_helper
        {
        };

        template <typename T, typename Wrapper>
        struct get_chunk_size_call_helper<T, Wrapper,
            typename std::enable_if<has_get_chunk_size<T>::value>::type>
        {
            template <typename Executor, typename F>
            HPX_FORCEINLINE std::size_t get_chunk_size(Executor&& exec, F&& f,
                std::size_t cores, std::size_t num_tasks)
            {
                auto& wrapped =
                    static_cast<unwrapper<Wrapper>*>(this)->member_.get();
                return wrapped.get_chunk_size(std::forward<Executor>(exec),
                    std::forward<F>(f), cores, num_tasks);
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename T, typename Wrapper, typename Enable = void>
        struct mark_begin_execution_call_helper
        {
        };

        template <typename T, typename Wrapper>
        struct mark_begin_execution_call_helper<T, Wrapper,
            typename std::enable_if<has_mark_begin_execution<T>::value>::type>
        {
            template <typename Executor>
            HPX_FORCEINLINE void mark_begin_execution(Executor&& exec)
            {
                auto& wrapped =
                    static_cast<unwrapper<Wrapper>*>(this)->member_.get();
                wrapped.mark_begin_execution(std::forward<Executor>(exec));
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename T, typename Wrapper, typename Enable = void>
        struct mark_end_of_scheduling_call_helper
        {
        };

        template <typename T, typename Wrapper>
        struct mark_end_of_scheduling_call_helper<T, Wrapper,
            typename std::enable_if<has_mark_begin_execution<T>::value>::type>
        {
            template <typename Executor>
            HPX_FORCEINLINE void mark_end_of_scheduling(Executor&& exec)
            {
                auto& wrapped =
                    static_cast<unwrapper<Wrapper>*>(this)->member_.get();
                wrapped.mark_end_of_scheduling(std::forward<Executor>(exec));
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename T, typename Wrapper, typename Enable = void>
        struct mark_end_execution_call_helper
        {
        };

        template <typename T, typename Wrapper>
        struct mark_end_execution_call_helper<T, Wrapper,
            typename std::enable_if<has_mark_begin_execution<T>::value>::type>
        {
            template <typename Executor>
            HPX_FORCEINLINE void mark_end_execution(Executor&& exec)
            {
                auto& wrapped =
                    static_cast<unwrapper<Wrapper>*>(this)->member_.get();
                wrapped.mark_end_execution(std::forward<Executor>(exec));
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename T, typename Wrapper, typename Enable = void>
        struct processing_units_count_call_helper
        {
        };

        template <typename T, typename Wrapper>
        struct processing_units_count_call_helper<T, Wrapper,
            typename std::enable_if<has_processing_units_count<T>::value>::type>
        {
            template <typename Executor>
            HPX_FORCEINLINE std::size_t processing_units_count(Executor&& exec)
            {
                auto& wrapped =
                    static_cast<unwrapper<Wrapper>*>(this)->member_.get();
                return wrapped.processing_units_count(
                    std::forward<Executor>(exec));
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename T, typename Wrapper, typename Enable = void>
        struct reset_thread_distribution_call_helper
        {
        };

        template <typename T, typename Wrapper>
        struct reset_thread_distribution_call_helper<T, Wrapper,
            typename std::enable_if<
                has_reset_thread_distribution<T>::value>::type>
        {
            template <typename Executor>
            HPX_FORCEINLINE void reset_thread_distribution(Executor&& exec)
            {
                auto& wrapped =
                    static_cast<unwrapper<Wrapper>*>(this)->member_.get();
                wrapped.reset_thread_distribution(std::forward<Executor>(exec));
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename T>
        struct base_member_helper
        {
            explicit base_member_helper(T t)
              : member_(t)
            {
            }

            T member_;
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename T>
        struct unwrapper<::std::reference_wrapper<T>>
          : base_member_helper<std::reference_wrapper<T>>
          , maximal_number_of_chunks_call_helper<T, std::reference_wrapper<T>>
          , get_chunk_size_call_helper<T, std::reference_wrapper<T>>
          , mark_begin_execution_call_helper<T, std::reference_wrapper<T>>
          , mark_end_of_scheduling_call_helper<T, std::reference_wrapper<T>>
          , mark_end_execution_call_helper<T, std::reference_wrapper<T>>
          , processing_units_count_call_helper<T, std::reference_wrapper<T>>
          , reset_thread_distribution_call_helper<T, std::reference_wrapper<T>>
        {
            using wrapper_type = std::reference_wrapper<T>;

            unwrapper(wrapper_type wrapped_param)
              : base_member_helper<wrapper_type>(wrapped_param)
            {
            }
        };

        ///////////////////////////////////////////////////////////////////////

#define HPX_STATIC_ASSERT_ON_PARAMETERS_AMBIGUITY(func)                        \
    static_assert(                                                             \
        parameters_type_counter<                                               \
            HPX_PP_CAT(hpx::parallel::execution::detail::has_, func) <         \
            typename hpx::util::decay_unwrap<Params>::type>::value... >        \
            ::value <= 1,                                                      \
        "Passing more than one executor parameters type "                      \
        "exposing " HPX_PP_STRINGIZE(func) " is not possible") /**/

        template <typename... Params>
        struct executor_parameters : public unwrapper<Params>...
        {
            static_assert(hpx::util::all_of<hpx::traits::is_executor_parameters<
                              typename std::decay<Params>::type>...>::value,
                "All passed parameters must be a proper executor parameters "
                "objects");
            static_assert(sizeof...(Params) >= 2,
                "This type is meant to be used with at least 2 parameters "
                "objects");

            HPX_STATIC_ASSERT_ON_PARAMETERS_AMBIGUITY(get_chunk_size);
            HPX_STATIC_ASSERT_ON_PARAMETERS_AMBIGUITY(mark_begin_execution);
            HPX_STATIC_ASSERT_ON_PARAMETERS_AMBIGUITY(mark_end_of_scheduling);
            HPX_STATIC_ASSERT_ON_PARAMETERS_AMBIGUITY(mark_end_execution);
            HPX_STATIC_ASSERT_ON_PARAMETERS_AMBIGUITY(processing_units_count);
            HPX_STATIC_ASSERT_ON_PARAMETERS_AMBIGUITY(maximal_number_of_chunks);
            HPX_STATIC_ASSERT_ON_PARAMETERS_AMBIGUITY(
                reset_thread_distribution);

            template <typename Dependent = void,
                typename Enable = typename std::enable_if<
                    hpx::util::all_of<std::is_constructible<Params>...>::value,
                    Dependent>::type>
            executor_parameters()
              : unwrapper<Params>()...
            {
            }

            template <typename... Params_,
                typename Enable =
                    typename std::enable_if<hpx::util::pack<Params...>::size ==
                        hpx::util::pack<Params_...>::size>::type>
            executor_parameters(Params_&&... params)
              : unwrapper<Params>(std::forward<Params_>(params))...
            {
            }

        private:
            friend class hpx::serialization::access;

            template <typename Archive>
            void serialize(Archive& ar, const unsigned int /* version */)
            {
                int const sequencer[] = {
                    (ar & serialization::base_object<Params>(*this), 0)..., 0};
                (void) sequencer;
            }
        };

#undef HPX_STATIC_ASSERT_ON_PARAMETERS_AMBIGUITY

        /// \endcond
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    template <typename... Params>
    struct executor_parameters_join
    {
        using type =
            detail::executor_parameters<typename std::decay<Params>::type...>;
    };

    template <typename... Params>
    HPX_FORCEINLINE typename executor_parameters_join<Params...>::type
    join_executor_parameters(Params&&... params)
    {
        using joined_params =
            typename executor_parameters_join<Params...>::type;
        return joined_params(std::forward<Params>(params)...);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Param>
    struct executor_parameters_join<Param>
    {
        using type = Param;
    };

    template <typename Param>
    HPX_FORCEINLINE Param&& join_executor_parameters(Param&& param)
    {
        static_assert(hpx::traits::is_executor_parameters<
                          typename std::decay<Param>::type>::value,
            "The passed parameter must be a proper executor parameters object");

        return std::forward<Param>(param);
    }
}}}    // namespace hpx::parallel::execution

namespace hpx { namespace execution { namespace experimental {
    HPX_INLINE_CONSTEXPR_VARIABLE struct make_with_hint_t
      : hpx::functional::tag<make_with_hint_t>
    {
    } make_with_hint{};

    HPX_INLINE_CONSTEXPR_VARIABLE struct get_hint_t
      : hpx::functional::tag<get_hint_t>
    {
    } get_hint{};
}}}    // namespace hpx::execution::experimental
