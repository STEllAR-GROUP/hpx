//  Copyright (c) 2016 Marcin Copik
//  Copyright (c) 2016-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/local/config.hpp>
#include <hpx/async_base/scheduling_properties.hpp>
#include <hpx/async_base/traits/is_launch_policy.hpp>
#include <hpx/concepts/has_member_xxx.hpp>
#include <hpx/execution/detail/execution_parameter_callbacks.hpp>
#include <hpx/execution_base/traits/is_executor.hpp>
#include <hpx/execution_base/traits/is_executor_parameters.hpp>
#include <hpx/functional/tag_fallback_dispatch.hpp>
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
        template <typename Property, template <typename> class CheckForProperty>
        struct with_property_t final
          : hpx::functional::tag_fallback<
                with_property_t<Property, CheckForProperty>>
        {
        private:
            using derived_propery_t =
                with_property_t<Property, CheckForProperty>;

            template <typename T>
            using check_for_property = CheckForProperty<std::decay_t<T>>;

            // clang-format off
            template <typename Executor, typename Parameters,
                HPX_CONCEPT_REQUIRES_(
                    !hpx::traits::is_executor_parameters<Parameters>::value ||
                    !check_for_property<Parameters>::value
                )>
            // clang-format on
            friend HPX_FORCEINLINE decltype(auto) tag_fallback_dispatch(
                derived_propery_t, Executor&& /*exec*/, Parameters&& /*params*/,
                Property prop)
            {
                return std::make_pair(prop, prop);
            }

            ///////////////////////////////////////////////////////////////////
            // Executor directly supports get_chunk_size
            // clang-format off
            template <typename Executor, typename Parameters,
                HPX_CONCEPT_REQUIRES_(
                    hpx::traits::is_executor_parameters<Parameters>::value &&
                    check_for_property<Parameters>::value
                )>
            // clang-format on
            friend HPX_FORCEINLINE decltype(auto) tag_fallback_dispatch(
                derived_propery_t, Executor&& exec, Parameters&& params,
                Property /*prop*/)
            {
                return std::pair<Parameters&&, Executor&&>(
                    std::forward<Parameters>(params),
                    std::forward<Executor>(exec));
            }

            ///////////////////////////////////////////////////////////////////
            // Executor directly supports get_chunk_size
            // clang-format off
            template <typename Executor, typename Parameters,
                HPX_CONCEPT_REQUIRES_(
                    hpx::traits::is_executor_any<Executor>::value &&
                    check_for_property<Executor>::value
                )>
            // clang-format on
            friend HPX_FORCEINLINE decltype(auto) tag_dispatch(
                derived_propery_t, Executor&& exec, Parameters&& params,
                Property /*prop*/)
            {
                return std::pair<Executor&&, Parameters&&>(
                    std::forward<Executor>(exec),
                    std::forward<Parameters>(params));
            }
        };

        ///////////////////////////////////////////////////////////////////////
        // define member traits
        HPX_HAS_MEMBER_XXX_TRAIT_DEF(get_chunk_size);

        ///////////////////////////////////////////////////////////////////////
        // default property implementation allowing to handle get_chunk_size
        struct get_chunk_size_property
        {
            // default implementation
            template <typename Target, typename F>
            HPX_FORCEINLINE static std::size_t get_chunk_size(Target, F&& /*f*/,
                std::size_t /*cores*/, std::size_t /*num_tasks*/)
            {
                // return zero for the chunk-size which will tell the
                // implementation to calculate the chunk size either based
                // on a specified maximum number of chunks or based on some
                // internal rule (if no maximum number of chunks was given)
                return 0;
            }
        };

        //////////////////////////////////////////////////////////////////////
        // Generate a type that is guaranteed to support get_chunk_size
        using with_get_chunk_size_t =
            with_property_t<get_chunk_size_property, has_get_chunk_size_t>;

        HPX_INLINE_CONSTEXPR_VARIABLE with_get_chunk_size_t
            with_get_chunk_size{};

        //////////////////////////////////////////////////////////////////////
        // customization point for interface get_chunk_size()
        template <typename Parameters, typename Executor_>
        struct get_chunk_size_fn_helper<Parameters, Executor_,
            std::enable_if_t<hpx::traits::is_executor_any<Executor_>::value>>
        {
            template <typename Executor, typename F>
            HPX_FORCEINLINE static std::size_t call(Parameters& params,
                Executor&& exec, F&& f, std::size_t cores,
                std::size_t num_tasks)
            {
                auto withprop =
                    with_get_chunk_size(std::forward<Executor>(exec), params,
                        get_chunk_size_property{});

                return withprop.first.get_chunk_size(
                    std::forward<decltype(withprop.second)>(withprop.second),
                    std::forward<F>(f), cores, num_tasks);
            }

            template <typename AnyParameters, typename Executor, typename F>
            HPX_FORCEINLINE static std::size_t call(AnyParameters params,
                Executor&& exec, F&& f, std::size_t cores,
                std::size_t num_tasks)
            {
                return call(static_cast<Parameters&>(params),
                    std::forward<Executor>(exec), std::forward<F>(f), cores,
                    num_tasks);
            }
        };

        ///////////////////////////////////////////////////////////////////////
        // define member traits
        HPX_HAS_MEMBER_XXX_TRAIT_DEF(maximal_number_of_chunks)

        ///////////////////////////////////////////////////////////////////////
        // default property implementation allowing to handle
        // maximal_number_of_chunks
        struct maximal_number_of_chunks_property
        {
            // default implementation
            template <typename Target>
            HPX_FORCEINLINE static std::size_t maximal_number_of_chunks(
                Target, std::size_t, std::size_t)
            {
                // return zero chunks which will tell the implementation to
                // calculate the number of chunks either based on a
                // specified chunk size or based on some internal rule (if no
                // chunk-size was given)
                return 0;
            }
        };

        //////////////////////////////////////////////////////////////////////
        // Generate a type that is guaranteed to support
        // maximal_number_of_chunks
        using with_maximal_number_of_chunks_t =
            with_property_t<maximal_number_of_chunks_property,
                has_maximal_number_of_chunks_t>;

        HPX_INLINE_CONSTEXPR_VARIABLE with_maximal_number_of_chunks_t
            with_maximal_number_of_chunks{};

        ///////////////////////////////////////////////////////////////////////
        // customization point for interface maximal_number_of_chunks()
        template <typename Parameters, typename Executor_>
        struct maximal_number_of_chunks_fn_helper<Parameters, Executor_,
            std::enable_if_t<hpx::traits::is_executor_any<Executor_>::value>>
        {
            template <typename Executor>
            HPX_FORCEINLINE static std::size_t call(Parameters& params,
                Executor&& exec, std::size_t cores, std::size_t num_tasks)
            {
                auto withprop =
                    with_maximal_number_of_chunks(std::forward<Executor>(exec),
                        params, maximal_number_of_chunks_property{});

                return withprop.first.maximal_number_of_chunks(
                    std::forward<decltype(withprop.second)>(withprop.second),
                    cores, num_tasks);
            }

            template <typename AnyParameters, typename Executor>
            HPX_FORCEINLINE static std::size_t call(AnyParameters params,
                Executor&& exec, std::size_t cores, std::size_t num_tasks)
            {
                return call(static_cast<Parameters&>(params),
                    std::forward<Executor>(exec), cores, num_tasks);
            }
        };

        ///////////////////////////////////////////////////////////////////////
        // define member traits
        HPX_HAS_MEMBER_XXX_TRAIT_DEF(reset_thread_distribution)

        ///////////////////////////////////////////////////////////////////////
        // default property implementation allowing to handle
        // reset_thread_distribution
        struct reset_thread_distribution_property
        {
            // default implementation
            template <typename Target>
            HPX_FORCEINLINE static void reset_thread_distribution(Target)
            {
            }
        };

        //////////////////////////////////////////////////////////////////////
        // Generate a type that is guaranteed to support
        // reset_thread_distribution
        using with_reset_thread_distribution_t =
            with_property_t<reset_thread_distribution_property,
                has_reset_thread_distribution_t>;

        HPX_INLINE_CONSTEXPR_VARIABLE with_reset_thread_distribution_t
            with_reset_thread_distribution{};

        ///////////////////////////////////////////////////////////////////////
        // customization point for interface reset_thread_distribution()
        template <typename Parameters, typename Executor_>
        struct reset_thread_distribution_fn_helper<Parameters, Executor_,
            std::enable_if_t<hpx::traits::is_executor_any<Executor_>::value>>
        {
            template <typename Executor>
            HPX_FORCEINLINE static void call(
                Parameters& params, Executor&& exec)
            {
                auto withprop =
                    with_reset_thread_distribution(std::forward<Executor>(exec),
                        params, reset_thread_distribution_property{});

                withprop.first.reset_thread_distribution(
                    std::forward<decltype(withprop.second)>(withprop.second));
            }

            template <typename AnyParameters, typename Executor>
            HPX_FORCEINLINE static void call(
                AnyParameters params, Executor&& exec)
            {
                call(static_cast<Parameters&>(params),
                    std::forward<Executor>(exec));
            }
        };

        ///////////////////////////////////////////////////////////////////////
        // define member traits
        HPX_HAS_MEMBER_XXX_TRAIT_DEF(processing_units_count)

        ///////////////////////////////////////////////////////////////////////
        // default property implementation allowing to handle
        // reset_thread_distribution
        struct processing_units_count_property
        {
            // default implementation
            template <typename Target>
            HPX_FORCEINLINE static std::size_t processing_units_count(Target)
            {
                return get_os_thread_count();
            }
        };

        //////////////////////////////////////////////////////////////////////
        // Generate a type that is guaranteed to support
        // reset_thread_distribution
        using with_processing_units_count_t =
            with_property_t<processing_units_count_property,
                has_processing_units_count_t>;

        HPX_INLINE_CONSTEXPR_VARIABLE with_processing_units_count_t
            with_processing_units_count{};

        ///////////////////////////////////////////////////////////////////////
        // customization point for interface processing_units_count()
        template <typename Parameters, typename Executor_>
        struct processing_units_count_fn_helper<Parameters, Executor_,
            std::enable_if_t<hpx::traits::is_executor_any<Executor_>::value>>
        {
            template <typename Executor>
            HPX_FORCEINLINE static std::size_t call(
                Parameters& params, Executor&& exec)
            {
                auto withprop =
                    with_processing_units_count(std::forward<Executor>(exec),
                        params, processing_units_count_property{});

                return withprop.first.processing_units_count(
                    std::forward<decltype(withprop.second)>(withprop.second));
            }

            template <typename AnyParameters, typename Executor>
            HPX_FORCEINLINE static std::size_t call(
                AnyParameters params, Executor&& exec)
            {
                return call(static_cast<Parameters&>(params),
                    std::forward<Executor>(exec));
            }
        };

        ///////////////////////////////////////////////////////////////////////
        // define member traits
        HPX_HAS_MEMBER_XXX_TRAIT_DEF(mark_begin_execution)

        ///////////////////////////////////////////////////////////////////////
        // default property implementation allowing to handle
        // mark_begin_execution
        struct mark_begin_execution_property
        {
            // default implementation
            template <typename Target>
            HPX_FORCEINLINE static void mark_begin_execution(Target)
            {
            }
        };

        //////////////////////////////////////////////////////////////////////
        // Generate a type that is guaranteed to support
        // mark_begin_execution
        using with_mark_begin_execution_t =
            with_property_t<mark_begin_execution_property,
                has_mark_begin_execution_t>;

        HPX_INLINE_CONSTEXPR_VARIABLE with_mark_begin_execution_t
            with_mark_begin_execution{};

        ///////////////////////////////////////////////////////////////////////
        // customization point for interface mark_begin_execution()
        template <typename Parameters, typename Executor_>
        struct mark_begin_execution_fn_helper<Parameters, Executor_,
            std::enable_if_t<hpx::traits::is_executor_any<Executor_>::value>>
        {
            template <typename Executor>
            HPX_FORCEINLINE static void call(
                Parameters& params, Executor&& exec)
            {
                auto withprop =
                    with_mark_begin_execution(std::forward<Executor>(exec),
                        params, mark_begin_execution_property{});

                withprop.first.mark_begin_execution(
                    std::forward<decltype(withprop.second)>(withprop.second));
            }

            template <typename AnyParameters, typename Executor>
            HPX_FORCEINLINE static void call(
                AnyParameters params, Executor&& exec)
            {
                call(static_cast<Parameters&>(params),
                    std::forward<Executor>(exec));
            }
        };

        ///////////////////////////////////////////////////////////////////////
        // define member traits
        HPX_HAS_MEMBER_XXX_TRAIT_DEF(mark_end_of_scheduling)

        ///////////////////////////////////////////////////////////////////////
        // default property implementation allowing to handle
        // mark_end_of_scheduling
        struct mark_end_of_scheduling_property
        {
            // default implementation
            template <typename Target>
            HPX_FORCEINLINE static void mark_end_of_scheduling(Target)
            {
            }
        };

        //////////////////////////////////////////////////////////////////////
        // Generate a type that is guaranteed to support
        // mark_end_of_scheduling
        using with_mark_end_of_scheduling_t =
            with_property_t<mark_end_of_scheduling_property,
                has_mark_end_of_scheduling_t>;

        HPX_INLINE_CONSTEXPR_VARIABLE with_mark_end_of_scheduling_t
            with_mark_end_of_scheduling{};

        ///////////////////////////////////////////////////////////////////////
        // customization point for interface mark_end_of_scheduling()
        template <typename Parameters, typename Executor_>
        struct mark_end_of_scheduling_fn_helper<Parameters, Executor_,
            std::enable_if_t<hpx::traits::is_executor_any<Executor_>::value>>
        {
            template <typename Executor>
            HPX_FORCEINLINE static void call(
                Parameters& params, Executor&& exec)
            {
                auto withprop =
                    with_mark_end_of_scheduling(std::forward<Executor>(exec),
                        params, mark_end_of_scheduling_property{});

                withprop.first.mark_end_of_scheduling(
                    std::forward<decltype(withprop.second)>(withprop.second));
            }

            template <typename AnyParameters, typename Executor>
            HPX_FORCEINLINE static void call(
                AnyParameters params, Executor&& exec)
            {
                call(static_cast<Parameters&>(params),
                    std::forward<Executor>(exec));
            }
        };

        ///////////////////////////////////////////////////////////////////////
        // define member traits
        HPX_HAS_MEMBER_XXX_TRAIT_DEF(mark_end_execution)

        ///////////////////////////////////////////////////////////////////////
        // default property implementation allowing to handle
        // mark_end_execution
        struct mark_end_execution_property
        {
            // default implementation
            template <typename Target>
            HPX_FORCEINLINE static void mark_end_execution(Target)
            {
            }
        };

        //////////////////////////////////////////////////////////////////////
        // Generate a type that is guaranteed to support
        // mark_end_execution
        using with_mark_end_execution_t =
            with_property_t<mark_end_execution_property,
                has_mark_end_execution_t>;

        HPX_INLINE_CONSTEXPR_VARIABLE with_mark_end_execution_t
            with_mark_end_execution{};

        ///////////////////////////////////////////////////////////////////////
        // customization point for interface mark_end_execution()
        template <typename Parameters, typename Executor_>
        struct mark_end_execution_fn_helper<Parameters, Executor_,
            std::enable_if_t<hpx::traits::is_executor_any<Executor_>::value>>
        {
            template <typename Executor>
            HPX_FORCEINLINE static void call(
                Parameters& params, Executor&& exec)
            {
                auto withprop =
                    with_mark_end_execution(std::forward<Executor>(exec),
                        params, mark_end_execution_property{});

                withprop.first.mark_end_execution(
                    std::forward<decltype(withprop.second)>(withprop.second));
            }

            template <typename AnyParameters, typename Executor>
            HPX_FORCEINLINE static void call(
                AnyParameters params, Executor&& exec)
            {
                call(static_cast<Parameters&>(params),
                    std::forward<Executor>(exec));
            }
        };
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
                typename Enable = std::enable_if_t<
                    std::is_constructible<T>::value, Dependent>>
            unwrapper()
              : T()
            {
            }

            // generic poor-man's forwarding constructor
            template <typename U,
                typename Enable = std::enable_if_t<
                    !std::is_same<std::decay_t<U>, unwrapper>::value>>
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
            std::enable_if_t<has_maximal_number_of_chunks<T>::value>>
        {
            template <typename Executor>
            HPX_FORCEINLINE std::size_t maximal_number_of_chunks(
                Executor&& exec, std::size_t cores, std::size_t num_tasks) const
            {
                auto& wrapped =
                    static_cast<unwrapper<Wrapper> const*>(this)->member_.get();
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
            std::enable_if_t<has_get_chunk_size<T>::value>>
        {
            template <typename Executor, typename F>
            HPX_FORCEINLINE std::size_t get_chunk_size(Executor&& exec, F&& f,
                std::size_t cores, std::size_t num_tasks) const
            {
                auto& wrapped =
                    static_cast<unwrapper<Wrapper> const*>(this)->member_.get();
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
            std::enable_if_t<has_mark_begin_execution<T>::value>>
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
            std::enable_if_t<has_mark_begin_execution<T>::value>>
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
            std::enable_if_t<has_mark_begin_execution<T>::value>>
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
            std::enable_if_t<has_processing_units_count<T>::value>>
        {
            template <typename Executor>
            HPX_FORCEINLINE std::size_t processing_units_count(
                Executor&& exec) const
            {
                auto& wrapped =
                    static_cast<unwrapper<Wrapper> const*>(this)->member_.get();
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
            std::enable_if_t<has_reset_thread_distribution<T>::value>>
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
            explicit constexpr base_member_helper(T t)
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

            constexpr unwrapper(wrapper_type wrapped_param)
              : base_member_helper<wrapper_type>(wrapped_param)
            {
            }
        };

        ///////////////////////////////////////////////////////////////////////

#define HPX_STATIC_ASSERT_ON_PARAMETERS_AMBIGUITY(func)                        \
    static_assert(                                                             \
        parameters_type_counter<                                               \
            HPX_PP_CAT(hpx::parallel::execution::detail::has_, func) <         \
            hpx::util::decay_unwrap_t<Params>>::value... > ::value <= 1,       \
        "Passing more than one executor parameters type "                      \
        "exposing " HPX_PP_STRINGIZE(func) " is not possible") /**/

        template <typename... Params>
        struct executor_parameters : public unwrapper<Params>...
        {
            static_assert(hpx::util::all_of<hpx::traits::is_executor_parameters<
                              std::decay_t<Params>>...>::value,
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
                typename Enable = std::enable_if_t<
                    hpx::util::all_of<std::is_constructible<Params>...>::value,
                    Dependent>>
            constexpr executor_parameters()
              : unwrapper<Params>()...
            {
            }

            template <typename... Params_,
                typename Enable =
                    std::enable_if_t<hpx::util::pack<Params...>::size ==
                        hpx::util::pack<Params_...>::size>>
            constexpr executor_parameters(Params_&&... params)
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
    // specialize trait for the type-combiner
    template <typename... Parameters>
    struct is_executor_parameters<detail::executor_parameters<Parameters...>>
      : hpx::util::all_of<hpx::traits::is_executor_parameters<Parameters>...>
    {
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename... Params>
    struct executor_parameters_join
    {
        using type = detail::executor_parameters<std::decay_t<Params>...>;
    };

    template <typename... Params>
    constexpr HPX_FORCEINLINE typename executor_parameters_join<Params...>::type
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
    constexpr HPX_FORCEINLINE Param&& join_executor_parameters(Param&& param)
    {
        static_assert(
            hpx::traits::is_executor_parameters<std::decay_t<Param>>::value,
            "The passed parameter must be a proper executor parameters object");

        return std::forward<Param>(param);
    }
}}}    // namespace hpx::parallel::execution
