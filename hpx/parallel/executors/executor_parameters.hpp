//  Copyright (c) 2016 Marcin Copik
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/executor_parameters.hpp

#if !defined(HPX_PARALLEL_EXECUTOR_PARAMETERS)
#define HPX_PARALLEL_EXECUTOR_PARAMETERS

#include <hpx/config.hpp>
#include <hpx/traits/is_executor_parameters.hpp>
#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/executors/executor_traits.hpp>
#include <hpx/util/always_void.hpp>

#include <type_traits>
#include <utility>
#include <vector>
#include <cstdarg>
#include <functional>

#include <boost/ref.hpp>
namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v3)
{
    namespace detail {

        template<template<typename> class Condition, typename Arg,
            typename Enable = void>
        struct counter_increment
        {
            HPX_CONSTEXPR static int value = 0;
        };

        template<template<typename> class Condition, typename Arg>
        struct counter_increment<Condition, Arg,
            typename std::enable_if< Condition<Arg>::value>::type >
        {
            HPX_CONSTEXPR static int value = 1;
        };

        template<template<typename> class Condition, typename ...Args>
        struct parameter_type_counter;

        template<template<typename> class Condition>
        struct parameter_type_counter<Condition>
        {
            HPX_CONSTEXPR static int value = 0;
        };

        /// Return the number of parameters which have been derived from type T.
        /// Useful for checking possible duplicates or ensuring that all passed types
        /// are indeed executor parameters.
        ///
        /// \param T                [in] Parameter to look for
        /// \param Parameters       [in] The executor object to use.
        ///
        //
        template<template<typename> class Condition, typename Arg1, typename ...Args>
        struct parameter_type_counter<Condition, Arg1, Args...>
        {
            HPX_CONSTEXPR static int value =
                parameter_type_counter<Condition, Args...>::value +
                counter_increment<Condition, Arg1>::value;
        };

        template<typename T>
        struct unwrapper : T
        {
            unwrapper() {}

            template<typename U>
            unwrapper(U && u) : T(u) {}
        };

        template<typename T>
        struct unwrapper< boost::reference_wrapper<T> >
        {
            template<typename WrapperType>
            unwrapper(WrapperType && wrapped_param) :
                wrap(std::forward<WrapperType>(wrapped_param)) {}

            template<typename Executor, typename U = T, typename std::enable_if<
                is_executor_parameters_chunk_size<U>::value >::type* = nullptr
                >
            auto variable_chunk_size(Executor & exec)
                -> decltype(std::declval<U>().variable_chunk_size(exec))
            {
                return wrap.get().variable_chunk_size(exec);
            }

            template<typename Executor, typename F, typename U = T,
                typename std::enable_if<
                    is_executor_parameters_chunk_size<U>::value
                >::type* = nullptr>
            auto get_chunk_size(Executor & exec, F && f, std::size_t num_tasks)
                -> decltype( std::declval<U>().get_chunk_size(
                            exec, std::forward<F>(f), num_tasks
                        ))
            {
                return wrap.get().variable_chunk_size(exec);
            }

            template<typename U = T, typename std::enable_if<
                is_executor_parameters_mark_begin_end<U>::value >::type* = nullptr
                >
            auto mark_begin_execution() ->
                decltype(std::declval<U>().mark_begin_execution())
            {
                return wrap.get().mark_begin_execution();
            }

            template<typename U = T, typename std::enable_if<
                is_executor_parameters_mark_begin_end<U>::value >::type* = nullptr
                >
            auto mark_end_execution() ->
                decltype(std::declval<U>().mark_end_execution())
            {
                return wrap.get().mark_end_execution();
            }

            template<typename U = T, typename std::enable_if<
                is_executor_parameters_processing_units_count<U>::value >::type* = nullptr
                >
            auto processing_units_count() ->
                decltype(std::declval<U>().processing_units_count())
            {
                return wrap.get().processing_units_count();
            }

            template<typename Executor, typename U = T, typename std::enable_if<
                is_executor_parameters_reset_thread_distr<U>::value >::type* = nullptr
                >
            auto reset_thread_distribution(Executor & exec)
                -> decltype(std::declval<U>().reset_thread_distribution(exec))
            {
                return wrap.get().reset_thread_distribution(exec);
            }
        private:
            boost::reference_wrapper<T> wrap;
        };

#if defined(HPX_HAVE_CXX11_STD_REFERENCE_WRAPPER)
        template<typename T>
        struct unwrapper< std::reference_wrapper<T> >
        {
            template<typename WrapperType>
            unwrapper(WrapperType && wrapped_param) :
                wrap(std::forward<WrapperType>(wrapped_param)) {}

            template<typename Executor, typename U = T, typename std::enable_if<
                is_executor_parameters_chunk_size<U>::value >::type* = nullptr
                >
            auto variable_chunk_size(Executor & exec)
                -> decltype(std::declval<U>().variable_chunk_size(exec))
            {
                return wrap.get().variable_chunk_size(exec);
            }

            template<typename Executor, typename F, typename U = T,
                typename std::enable_if<
                    is_executor_parameters_chunk_size<U>::value
                >::type* = nullptr>
            auto get_chunk_size(Executor & exec, F && f, std::size_t num_tasks)
                -> decltype( std::declval<U>().get_chunk_size(
                            exec, std::forward<F>(f), num_tasks
                        ))
            {
                return wrap.get().variable_chunk_size(exec);
            }

            template<typename U = T, typename std::enable_if<
                is_executor_parameters_mark_begin_end<U>::value >::type* = nullptr
                >
            auto mark_begin_execution()
                -> decltype(std::declval<U>().mark_begin_execution())
            {
                return wrap.get().mark_begin_execution();
            }

            template<typename U = T, typename std::enable_if<
                is_executor_parameters_mark_begin_end<U>::value >::type* = nullptr
                >
            auto mark_end_execution() -> decltype(std::declval<U>().mark_end_execution())
            {
                return wrap.get().mark_end_execution();
            }

            template<typename U = T, typename std::enable_if<
                is_executor_parameters_processing_units_count<U>::value >::type* = nullptr
                >
            auto processing_units_count()
                -> decltype(std::declval<U>().processing_units_count())
            {
                return wrap.get().processing_units_count();
            }

            template<typename Executor, typename U = T, typename std::enable_if<
                is_executor_parameters_reset_thread_distr<U>::value >::type* = nullptr
                >
            auto reset_thread_distribution(Executor & exec)
                -> decltype(std::declval<U>().reset_thread_distribution(exec))
            {
                return wrap.get().reset_thread_distribution(exec);
            }

        private:
            std::reference_wrapper<T> wrap;
        };
#endif

        template<typename... Params>
        struct executor_parameters : public unwrapper<Params>...
        {
        public:

            executor_parameters()
            {
                static_assert(
                    parameter_type_counter<
                            hpx::parallel::is_executor_parameters, Params...
                        >::value == sizeof...(Params),
                    "All passed parameters must be a proper executor parameter!"
                );

                static_assert(
                    parameter_type_counter<
                        hpx::parallel::is_executor_parameters_chunk_size, Params...
                        >::value <= 1,
                    "Passing more than one chunk size policy is prohibited!"
                );
            }

            template<typename... Params_>
            executor_parameters(Params_ &&... params) : unwrapper<Params>(params)...
            {
                static_assert(
                    parameter_type_counter<
                            hpx::parallel::is_executor_parameters, Params...
                        >::value == sizeof...(Params),
                    "All passed parameters must be a proper executor parameter!"
                );

                static_assert(
                    parameter_type_counter<
                            hpx::parallel::is_executor_parameters_chunk_size, Params...
                        >::value <= 1,
                    "Passing more than one chunk size policy is prohibited!"
                );
            }
        };
    }

    template<typename... Params>
    struct executor_parameters_join
    {
        typedef detail::executor_parameters<
                typename hpx::util::decay<Params>::type...
            > type;
    };

}}}

#endif
