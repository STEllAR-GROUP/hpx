//  Copyright (c) 2016 Marcin Copik
//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_EXECUTOR_PARAMETERS)
#define HPX_PARALLEL_EXECUTOR_PARAMETERS

#include <hpx/config.hpp>
#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/executors/executor_parameter_traits.hpp>
#include <hpx/runtime/serialization/base_object.hpp>
#include <hpx/traits/is_executor_parameters.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/detail/pack.hpp>

#include <cstdarg>
#if defined(HPX_HAVE_CXX11_STD_REFERENCE_WRAPPER)
#include <functional>
#endif
#include <type_traits>
#include <utility>
#include <vector>

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/stringize.hpp>
#include <boost/ref.hpp>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v3)
{
    namespace detail
    {
        /// \cond NOINTERNAL
        template <bool ... Flags>
        struct parameters_type_counter;

        template <>
        struct parameters_type_counter<>
        {
            HPX_CONSTEXPR static int value = 0;
        };

        /// Return the number of parameters which are true
        template <bool Flag1, bool ... Flags>
        struct parameters_type_counter<Flag1, Flags...>
        {
            HPX_CONSTEXPR static int value =
                Flag1 + parameters_type_counter<Flags...>::value;
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename T>
        struct unwrapper : T
        {
            // default constructor is needed for serialization purposes
            unwrapper() : T() {}

            // generic poor-mans forwarding constructor
            template <typename U>
            unwrapper(U && u) : T(std::forward<U>(u)) {}
        };

        template <typename T>
        struct unwrapper< ::boost::reference_wrapper<T> >
        {
            template <typename WrapperType>
            unwrapper(WrapperType && wrapped_param)
              : wrap(std::forward<WrapperType>(wrapped_param))
            {}

            template <
                typename Executor, typename U = T,
                typename std::enable_if<
                    has_variable_chunk_size<U>::value
                >::type* = nullptr>
            auto variable_chunk_size(Executor & exec)
            ->  decltype(std::declval<U>().variable_chunk_size(exec))
            {
                return wrap.get().variable_chunk_size(exec);
            }

            template <typename Executor, typename F, typename U = T,
                typename std::enable_if<
                    has_get_chunk_size<U>::value
                >::type* = nullptr>
            auto get_chunk_size(Executor & exec, F && f, std::size_t num_tasks)
            ->  decltype(
                    std::declval<U>().get_chunk_size(
                        exec, std::forward<F>(f), num_tasks)
                )
            {
                return wrap.get().get_chunk_size(exec, std::forward<F>(f),
                    num_tasks);
            }

            template <typename U = T,
                typename std::enable_if<
                    has_mark_begin_execution<U>::value
                >::type* = nullptr>
            auto mark_begin_execution()
            ->  decltype(std::declval<U>().mark_begin_execution())
            {
                return wrap.get().mark_begin_execution();
            }

            template <typename U = T,
                typename std::enable_if<
                    has_mark_end_execution<U>::value
                >::type* = nullptr>
            auto mark_end_execution()
            ->  decltype(std::declval<U>().mark_end_execution())
            {
                return wrap.get().mark_end_execution();
            }

            template <typename U = T,
                typename std::enable_if<
                    has_processing_units_count<U>::value
                >::type* = nullptr>
            auto processing_units_count()
            ->  decltype(std::declval<U>().processing_units_count())
            {
                return wrap.get().processing_units_count();
            }

            template<typename Executor, typename U = T,
                typename std::enable_if<
                    has_reset_thread_distribution<U>::value
                >::type* = nullptr>
            auto reset_thread_distribution(Executor & exec)
            ->  decltype(std::declval<U>().reset_thread_distribution(exec))
            {
                return wrap.get().reset_thread_distribution(exec);
            }

        private:
            boost::reference_wrapper<T> wrap;
        };

#if defined(HPX_HAVE_CXX11_STD_REFERENCE_WRAPPER)
        template <typename T>
        struct unwrapper< ::std::reference_wrapper<T> >
        {
            template <typename WrapperType>
            unwrapper(WrapperType && wrapped_param)
              : wrap(std::forward<WrapperType>(wrapped_param))
            {}

            template <typename Executor, typename U = T,
                typename std::enable_if<
                    has_variable_chunk_size<U>::value
                >::type* = nullptr>
            auto variable_chunk_size(Executor & exec)
            ->  decltype(std::declval<U>().variable_chunk_size(exec))
            {
                return wrap.get().variable_chunk_size(exec);
            }

            template <typename Executor, typename F, typename U = T,
                typename std::enable_if<
                    has_get_chunk_size<U>::value
                >::type* = nullptr>
            auto get_chunk_size(Executor & exec, F && f, std::size_t num_tasks)
            ->  decltype(
                    std::declval<U>().get_chunk_size(
                        exec, std::forward<F>(f), num_tasks)
                )
            {
                return wrap.get().get_chunk_size(exec, std::forward<F>(f),
                    num_tasks);
            }

            template <typename U = T,
                typename std::enable_if<
                    has_mark_begin_execution<U>::value
                >::type* = nullptr>
            auto mark_begin_execution()
            ->  decltype(std::declval<U>().mark_begin_execution())
            {
                return wrap.get().mark_begin_execution();
            }

            template <typename U = T,
                typename std::enable_if<
                    has_mark_end_execution<U>::value
                >::type* = nullptr>
            auto mark_end_execution()
            ->  decltype(std::declval<U>().mark_end_execution())
            {
                return wrap.get().mark_end_execution();
            }

            template <typename U = T,
                typename std::enable_if<
                    has_processing_units_count<U>::value
                >::type* = nullptr>
            auto processing_units_count()
            ->  decltype(std::declval<U>().processing_units_count())
            {
                return wrap.get().processing_units_count();
            }

            template <typename Executor, typename U = T,
                typename std::enable_if<
                    has_reset_thread_distribution<U>::value
                >::type* = nullptr>
            auto reset_thread_distribution(Executor & exec)
            ->  decltype(std::declval<U>().reset_thread_distribution(exec))
            {
                return wrap.get().reset_thread_distribution(exec);
            }

        private:
            std::reference_wrapper<T> wrap;
        };
#endif

        ///////////////////////////////////////////////////////////////////////

#define HPX_STATIC_ASSERT_ON_PARAMETERS_AMBIGUITY(func)                       \
    static_assert(parameters_type_counter<                                    \
            BOOST_PP_CAT(hpx::parallel::v3::detail::has_, func)<              \
                typename hpx::util::decay_unwrap<Params>::type>::value...     \
        >::value <= 1,                                                        \
        "Passing more than one executor parameters type exposing "            \
            BOOST_PP_STRINGIZE(func) " is not possible")                      \
    /**/

        template <typename ... Params>
        struct executor_parameters : public unwrapper<Params>...
        {
            static_assert(
                hpx::util::detail::all_of<
                    hpx::traits::is_executor_parameters<Params>...
                >::value,
                "All passed parameters must be a proper executor parameters "
                "objects"
            );

            HPX_STATIC_ASSERT_ON_PARAMETERS_AMBIGUITY(variable_chunk_size);
            HPX_STATIC_ASSERT_ON_PARAMETERS_AMBIGUITY(get_chunk_size);
            HPX_STATIC_ASSERT_ON_PARAMETERS_AMBIGUITY(mark_begin_execution);
            HPX_STATIC_ASSERT_ON_PARAMETERS_AMBIGUITY(mark_end_execution);
            HPX_STATIC_ASSERT_ON_PARAMETERS_AMBIGUITY(processing_units_count);
            HPX_STATIC_ASSERT_ON_PARAMETERS_AMBIGUITY(reset_thread_distribution);

            executor_parameters()
              : unwrapper<Params>()...
            {}

            template <typename ... Params_>
            executor_parameters(Params_ &&... params)
              : unwrapper<Params>(std::forward<Params_>(params))...
            {}

        private:
            friend class hpx::serialization::access;

            template <typename Archive>
            void serialize(Archive & ar, const unsigned int version)
            {
                int const sequencer[] = {
                    (ar & serialization::base_object<Params>(*this), 0)..., 0
                };
                (void)sequencer;
            }
        };

#undef HPX_STATIC_ASSERT_ON_PARAMETERS_AMBIGUITY

        /// \endcond
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename ... Params>
    struct executor_parameters_join
    {
        typedef detail::executor_parameters<
                typename hpx::util::decay<Params>::type...
            > type;
    };

    template <typename ... Params>
    HPX_FORCEINLINE
    typename executor_parameters_join<Params...>::type
    join_executor_parameters(Params &&... params)
    {
        typedef
            typename executor_parameters_join<Params...>::type
            joined_params;
        return joined_params(std::forward<Params>(params)...);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Param>
    struct executor_parameters_join<Param>
    {
        typedef Param type;
    };

    template <typename Param>
    HPX_FORCEINLINE
    Param && join_executor_parameters(Param && param)
    {
        return std::forward<Param>(param);
    }
}}}

#endif
