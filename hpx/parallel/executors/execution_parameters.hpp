//  Copyright (c) 2016 Marcin Copik
//  Copyright (c) 2016-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// hpxinspect:nodeprecatedinclude:boost/ref.hpp
// hpxinspect:nodeprecatedname:boost::reference_wrapper

#if !defined(HPX_PARALLEL_EXECUTORS_EXECUTION_PARAMETERS_AUG_21_2017_0750PM)
#define HPX_PARALLEL_EXECUTORS_EXECUTION_PARAMETERS_AUG_21_2017_0750PM

#include <hpx/config.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/runtime/serialization/base_object.hpp>
#include <hpx/traits/detail/wrap_int.hpp>
#include <hpx/traits/has_member_xxx.hpp>
#include <hpx/traits/is_executor.hpp>
#include <hpx/traits/is_executor_parameters.hpp>
#include <hpx/traits/is_launch_policy.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/detail/pack.hpp>
#include <hpx/util/detail/pp/cat.hpp>
#include <hpx/util/detail/pp/stringize.hpp>

#include <hpx/parallel/executors/execution.hpp>
#include <hpx/parallel/executors/execution_parameters_fwd.hpp>

#include <boost/ref.hpp>

#include <cstddef>
#include <functional>
#include <type_traits>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parallel { namespace execution
{
    namespace detail
    {
        /// \cond NOINTERNAL

        ///////////////////////////////////////////////////////////////////////
        // customization point for interface get_chunk_size()
        template <typename Parameters, typename Executor_>
        struct get_chunk_size_fn_helper<Parameters, Executor_,
            typename std::enable_if<
                hpx::traits::is_executor_any<Executor_>::value ||
                    hpx::traits::is_threads_executor<Executor_>::value
            >::type>
        {
            template <typename AnyParameters, typename Executor, typename F>
            HPX_FORCEINLINE static std::size_t
            call(hpx::traits::detail::wrap_int, AnyParameters&& params,
                    Executor&& exec, F&& f, std::size_t cores,
                    std::size_t num_tasks)
            {
                return (num_tasks + 4 * cores -1) / (4 * cores);
            }

            template <typename AnyParameters, typename Executor, typename F>
            HPX_FORCEINLINE static auto
            call(int, AnyParameters&& params, Executor&& exec, F&& f,
                    std::size_t cores, std::size_t num_tasks)
            ->  decltype(params.get_chunk_size(
                    std::forward<Executor>(exec), std::forward<F>(f), cores,
                    num_tasks))
            {
                return params.get_chunk_size(
                    std::forward<Executor>(exec), std::forward<F>(f), cores,
                    num_tasks);
            }

            template <typename Executor, typename F>
            HPX_FORCEINLINE static std::size_t
            call(Parameters& params, Executor&& exec, F&& f,
                    std::size_t cores, std::size_t num_tasks)
            {
                return call(0, params, std::forward<Executor>(exec),
                    std::forward<F>(f), cores, num_tasks);
            }

            template <typename AnyParameters, typename Executor, typename F>
            HPX_FORCEINLINE static std::size_t
            call(AnyParameters params, Executor&& exec, F&& f,
                    std::size_t cores, std::size_t num_tasks)
            {
                return call(0, static_cast<Parameters&>(params),
                    std::forward<Executor>(exec), std::forward<F>(f), cores,
                    num_tasks);
            }

            template <typename AnyParameters, typename Executor, typename F>
            struct result
            {
                using type = decltype(call(
                    std::declval<AnyParameters>(),
                    std::declval<Executor>(), std::declval<F>(),
                    std::declval<std::size_t>(), std::declval<std::size_t>()
                ));
            };
        };

        HPX_HAS_MEMBER_XXX_TRAIT_DEF(get_chunk_size)

        ///////////////////////////////////////////////////////////////////////
        // customization point for interface maximal_number_of_chunks()
        template <typename Parameters, typename Executor_>
        struct maximal_number_of_chunks_fn_helper<Parameters, Executor_,
            typename std::enable_if<
                hpx::traits::is_executor_any<Executor_>::value ||
                    hpx::traits::is_threads_executor<Executor_>::value
            >::type>
        {
            template <typename AnyParameters, typename Executor>
            HPX_FORCEINLINE static std::size_t
            call(hpx::traits::detail::wrap_int, AnyParameters &&, Executor &&,
                std::size_t cores, std::size_t num_tasks)
            {
                return 4 * cores;       // assume 4 times the number of cores
            }

            template <typename AnyParameters, typename Executor>
            HPX_FORCEINLINE static auto call(int, AnyParameters&& params,
                    Executor&& exec, std::size_t cores, std::size_t num_tasks)
            ->  decltype(params.maximal_number_of_chunks(
                    std::forward<Executor>(exec), cores, num_tasks))
            {
                return params.maximal_number_of_chunks(
                    std::forward<Executor>(exec), cores, num_tasks);
            }

            template <typename Executor>
            HPX_FORCEINLINE static std::size_t
            call(Parameters& params, Executor && exec, std::size_t cores,
                std::size_t num_tasks)
            {
                return call(0, params, std::forward<Executor>(exec), cores,
                    num_tasks);
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
                using type = decltype(call(
                    std::declval<AnyParameters>(),
                    std::declval<Executor>(),
                    std::declval<std::size_t>(), std::declval<std::size_t>()
                ));
            };
        };

        HPX_HAS_MEMBER_XXX_TRAIT_DEF(maximal_number_of_chunks)

        ///////////////////////////////////////////////////////////////////////
        // customization point for interface reset_thread_distribution()
        template <typename Parameters, typename Executor_>
        struct reset_thread_distribution_fn_helper<Parameters, Executor_,
            typename std::enable_if<
                hpx::traits::is_executor_any<Executor_>::value ||
                    hpx::traits::is_threads_executor<Executor_>::value
            >::type>
        {
            // handle thread executors
            template <typename AnyParameters, typename Executor>
            HPX_FORCEINLINE static void call2(
                hpx::traits::detail::wrap_int, AnyParameters&& params,
                    Executor&& exec)
            {
            }

            template <typename AnyParameters, typename Executor>
            HPX_FORCEINLINE static auto call2(int, AnyParameters&&,
                    Executor&& exec)
            ->  decltype(exec.reset_thread_distribution())
            {
                exec.reset_thread_distribution();
            }

            // handle parameters exposing the required functionality
            template <typename AnyParameters, typename Executor>
            HPX_FORCEINLINE static void call(
                hpx::traits::detail::wrap_int, AnyParameters&& params,
                    Executor&& exec)
            {
                call2(0, std::forward<AnyParameters>(params),
                    std::forward<Executor>(exec));
            }

            template <typename AnyParameters, typename Executor>
            HPX_FORCEINLINE static auto call(int, AnyParameters && params,
                    Executor && exec)
            ->  decltype(params.reset_thread_distribution(
                    std::forward<Executor>(exec)))
            {
                params.reset_thread_distribution(std::forward<Executor>(exec));
            }

            template <typename Executor>
            HPX_FORCEINLINE static void call(Parameters& params,
                Executor && exec)
            {
                call(0, params, std::forward<Executor>(exec));
            }

            template <typename AnyParameters, typename Executor>
            HPX_FORCEINLINE static void call(AnyParameters params,
                Executor && exec)
            {
                call(static_cast<Parameters&>(params),
                    std::forward<Executor>(exec));
            }

            template <typename AnyParameters, typename Executor>
            struct result
            {
                using type = decltype(call(
                    std::declval<AnyParameters>(),
                    std::declval<Executor>()
                ));
            };
        };

        HPX_HAS_MEMBER_XXX_TRAIT_DEF(reset_thread_distribution)

        ///////////////////////////////////////////////////////////////////////
        // customization point for interface count_processing_units()
        template <typename Parameters, typename Executor_>
        struct count_processing_units_fn_helper<Parameters, Executor_,
            typename std::enable_if<
                hpx::traits::is_executor_any<Executor_>::value ||
                    hpx::traits::is_threads_executor<Executor_>::value
            >::type>
        {
            template <typename AnyParameters, typename Executor>
            HPX_FORCEINLINE static std::size_t call(
                hpx::traits::detail::wrap_int,
                AnyParameters && params, Executor&& exec)
            {
                return hpx::get_os_thread_count();
            }

            template <typename AnyParameters, typename Executor>
            HPX_FORCEINLINE static auto call(int, AnyParameters && params,
                    Executor && exec)
            ->  decltype(params.processing_units_count(
                    std::forward<Executor>(exec)))
            {
                return params.processing_units_count(
                    std::forward<Executor>(exec));
            }

            template <typename Executor>
            HPX_FORCEINLINE static std::size_t call(Parameters& params,
                Executor && exec)
            {
                return call(0, params, std::forward<Executor>(exec));
            }

            template <typename AnyParameters, typename Executor>
            HPX_FORCEINLINE static std::size_t call(AnyParameters params,
                Executor && exec)
            {
                return call(static_cast<Parameters&>(params),
                    std::forward<Executor>(exec));
            }

            template <typename AnyParameters, typename Executor>
            struct result
            {
                using type = decltype(call(
                    std::declval<AnyParameters>(),
                    std::declval<Executor>()
                ));
            };
        };

        HPX_HAS_MEMBER_XXX_TRAIT_DEF(count_processing_units)

        ///////////////////////////////////////////////////////////////////////
        // customization point for interface mark_begin_execution()
        template <typename Parameters, typename Executor_>
        struct mark_begin_execution_fn_helper<Parameters, Executor_,
            typename std::enable_if<
                hpx::traits::is_executor_any<Executor_>::value ||
                    hpx::traits::is_threads_executor<Executor_>::value
            >::type>
        {
            template <typename AnyParameters, typename Executor>
            HPX_FORCEINLINE static void call(hpx::traits::detail::wrap_int,
                AnyParameters &&, Executor &&)
            {
            }

            template <typename AnyParameters, typename Executor>
            HPX_FORCEINLINE static auto call(int, AnyParameters && params,
                    Executor && exec)
            ->  decltype(params.mark_begin_execution(
                    std::forward<Executor>(exec)))
            {
                params.mark_begin_execution(std::forward<Executor>(exec));
            }

            template <typename Executor>
            HPX_FORCEINLINE static void call(Parameters& params,
                Executor && exec)
            {
                call(0, params, std::forward<Executor>(exec));
            }

            template <typename AnyParameters, typename Executor>
            HPX_FORCEINLINE static void call(AnyParameters params,
                Executor && exec)
            {
                call(static_cast<Parameters&>(params),
                    std::forward<Executor>(exec));
            }

            template <typename AnyParameters, typename Executor>
            struct result
            {
                using type = decltype(call(
                    std::declval<AnyParameters>(),
                    std::declval<Executor>()
                ));
            };
        };

        HPX_HAS_MEMBER_XXX_TRAIT_DEF(mark_begin_execution)

        ///////////////////////////////////////////////////////////////////////
        // customization point for interface mark_end_execution()
        template <typename Parameters, typename Executor_>
        struct mark_end_execution_fn_helper<Parameters, Executor_,
            typename std::enable_if<
                hpx::traits::is_executor_any<Executor_>::value ||
                    hpx::traits::is_threads_executor<Executor_>::value
            >::type>
        {
            template <typename AnyParameters, typename Executor>
            HPX_FORCEINLINE static void call(hpx::traits::detail::wrap_int,
                AnyParameters &&, Executor &&)
            {
            }

            template <typename AnyParameters, typename Executor>
            HPX_FORCEINLINE static auto call(int, AnyParameters && params,
                    Executor&& exec)
            ->  decltype(params.mark_end_execution(std::forward<Executor>(exec)))
            {
                params.mark_end_execution(std::forward<Executor>(exec));
            }

            template <typename Executor>
            HPX_FORCEINLINE static void call(Parameters& params,
                Executor&& exec)
            {
                call(0, params, std::forward<Executor>(exec));
            }

            template <typename AnyParameters, typename Executor>
            HPX_FORCEINLINE static void call(AnyParameters params,
                Executor&& exec)
            {
                call(static_cast<Parameters&>(params),
                    std::forward<Executor>(exec));
            }

            template <typename AnyParameters, typename Executor>
            struct result
            {
                using type = decltype(call(
                    std::declval<AnyParameters>(),
                    std::declval<Executor>()
                ));
            };
        };

        HPX_HAS_MEMBER_XXX_TRAIT_DEF(mark_end_execution)

        /// \endcond
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        /// \cond NOINTERNAL
        template <bool ... Flags>
        struct parameters_type_counter;

        template <>
        struct parameters_type_counter<>
        {
            static HPX_CONSTEXPR_OR_CONST int value = 0;
        };

        /// Return the number of parameters which are true
        template <bool Flag1, bool ... Flags>
        struct parameters_type_counter<Flag1, Flags...>
        {
            static HPX_CONSTEXPR_OR_CONST int value =
                Flag1 + parameters_type_counter<Flags...>::value;
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename T>
        struct unwrapper : T
        {
            // default constructor is needed for serialization purposes
            template <typename Dependent = void, typename Enable =
                typename std::enable_if<
                    std::is_constructible<T>::value, Dependent
                >::type>
            unwrapper() : T() {}

            // generic poor-man's forwarding constructor
            template <typename U, typename Enable = typename
                std::enable_if<!std::is_same<typename hpx::util::decay<U>::type,
                unwrapper>::value>::type>
            unwrapper(U && u) : T(std::forward<U>(u)) {}
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename T, typename Wrapper, typename Enable = void>
        struct maximal_number_of_chunks_call_helper
        {
            maximal_number_of_chunks_call_helper(Wrapper&) {}
        };

        template <typename T, typename Wrapper>
        struct maximal_number_of_chunks_call_helper<T, Wrapper,
            typename std::enable_if<
                has_maximal_number_of_chunks<T>::value
            >::type>
        {
            maximal_number_of_chunks_call_helper(Wrapper& wrap)
              : wrap_(wrap)
            {}

            template <typename Executor>
            HPX_FORCEINLINE std::size_t maximal_number_of_chunks(
                Executor && exec, std::size_t cores, std::size_t num_tasks)
            {
                return wrap_.get().maximal_number_of_chunks(
                    std::forward<Executor>(exec), cores, num_tasks);
            }

        private:
            Wrapper& wrap_;
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename T, typename Wrapper, typename Enable = void>
        struct get_chunk_size_call_helper
        {
            get_chunk_size_call_helper(Wrapper&) {}
        };

        template <typename T, typename Wrapper>
        struct get_chunk_size_call_helper<T, Wrapper,
            typename std::enable_if<has_get_chunk_size<T>::value>::type>
        {
            get_chunk_size_call_helper(Wrapper& wrap)
              : wrap_(wrap)
            {}

            template <typename Executor, typename F>
            HPX_FORCEINLINE std::size_t get_chunk_size(Executor && exec, F && f,
                std::size_t cores, std::size_t num_tasks)
            {
                return wrap_.get().get_chunk_size(std::forward<Executor>(exec),
                    std::forward<F>(f), cores, num_tasks);
            }

        private:
            Wrapper& wrap_;
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename T, typename Wrapper, typename Enable = void>
        struct mark_begin_execution_call_helper
        {
            mark_begin_execution_call_helper(Wrapper&) {}
        };

        template <typename T, typename Wrapper>
        struct mark_begin_execution_call_helper<T, Wrapper,
            typename std::enable_if<has_mark_begin_execution<T>::value>::type>
        {
            mark_begin_execution_call_helper(Wrapper& wrap)
              : wrap_(wrap)
            {}

            template <typename Executor>
            HPX_FORCEINLINE void mark_begin_execution(Executor && exec)
            {
                wrap_.get().mark_begin_execution(std::forward<Executor>(exec));
            }

        private:
            Wrapper& wrap_;
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename T, typename Wrapper, typename Enable = void>
        struct mark_end_execution_call_helper
        {
            mark_end_execution_call_helper(Wrapper&) {}
        };

        template <typename T, typename Wrapper>
        struct mark_end_execution_call_helper<T, Wrapper,
            typename std::enable_if<has_mark_begin_execution<T>::value>::type>
        {
            mark_end_execution_call_helper(Wrapper& wrap)
              : wrap_(wrap)
            {}

            template <typename Executor>
            HPX_FORCEINLINE void mark_end_execution(Executor && exec)
            {
                wrap_.get().mark_end_execution(std::forward<Executor>(exec));
            }

        private:
            Wrapper& wrap_;
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename T, typename Wrapper, typename Enable = void>
        struct processing_units_count_call_helper
        {
            processing_units_count_call_helper(Wrapper&) {}
        };

        template <typename T, typename Wrapper>
        struct processing_units_count_call_helper<T, Wrapper,
            typename std::enable_if<
                has_count_processing_units<T>::value
            >::type>
        {
            processing_units_count_call_helper(Wrapper& wrap)
              : wrap_(wrap)
            {}

            template <typename Executor>
            HPX_FORCEINLINE std::size_t
            processing_units_count(Executor && exec)
            {
                return wrap_.get().processing_units_count(
                    std::forward<Executor>(exec));
            }

        private:
            Wrapper& wrap_;
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename T, typename Wrapper, typename Enable = void>
        struct reset_thread_distribution_call_helper
        {
            reset_thread_distribution_call_helper(Wrapper&) {}
        };

        template <typename T, typename Wrapper>
        struct reset_thread_distribution_call_helper<T, Wrapper,
            typename std::enable_if<
                has_reset_thread_distribution<T>::value
            >::type>
        {
            reset_thread_distribution_call_helper(Wrapper& wrap)
              : wrap_(wrap)
            {}

            template <typename Executor>
            HPX_FORCEINLINE void reset_thread_distribution(Executor && exec)
            {
                wrap_.get().reset_thread_distribution(
                    std::forward<Executor>(exec));
            }

        private:
            Wrapper& wrap_;
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename T>
        struct base_member_helper
        {
            base_member_helper(T && t)
              : member_(std::move(t))
            {}

            T member_;
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename T>
        struct unwrapper< ::boost::reference_wrapper<T> >
          : base_member_helper<boost::reference_wrapper<T> >
          , maximal_number_of_chunks_call_helper<T, boost::reference_wrapper<T> >
          , get_chunk_size_call_helper<T, boost::reference_wrapper<T> >
          , mark_begin_execution_call_helper<T, boost::reference_wrapper<T> >
          , mark_end_execution_call_helper<T, boost::reference_wrapper<T> >
          , processing_units_count_call_helper<T, boost::reference_wrapper<T> >
          , reset_thread_distribution_call_helper<T, boost::reference_wrapper<T> >
        {
            typedef boost::reference_wrapper<T> wrapper_type;

            unwrapper(wrapper_type wrapped_param)
              : base_member_helper<wrapper_type>(std::move(wrapped_param))
              , maximal_number_of_chunks_call_helper<T, wrapper_type>(this->member_)
              , get_chunk_size_call_helper<T, wrapper_type>(this->member_)
              , mark_begin_execution_call_helper<T, wrapper_type>(this->member_)
              , mark_end_execution_call_helper<T, wrapper_type>(this->member_)
              , processing_units_count_call_helper<T, wrapper_type>(this->member_)
              , reset_thread_distribution_call_helper<T, wrapper_type>(this->member_)
            {}
        };

        template <typename T>
        struct unwrapper< ::std::reference_wrapper<T> >
          : base_member_helper<std::reference_wrapper<T> >
          , maximal_number_of_chunks_call_helper<T, std::reference_wrapper<T> >
          , get_chunk_size_call_helper<T, std::reference_wrapper<T> >
          , mark_begin_execution_call_helper<T, std::reference_wrapper<T> >
          , mark_end_execution_call_helper<T, std::reference_wrapper<T> >
          , processing_units_count_call_helper<T, std::reference_wrapper<T> >
          , reset_thread_distribution_call_helper<T, std::reference_wrapper<T> >
        {
            typedef std::reference_wrapper<T> wrapper_type;

            unwrapper(wrapper_type wrapped_param)
              : base_member_helper<wrapper_type>(std::move(wrapped_param))
              , maximal_number_of_chunks_call_helper<T, wrapper_type>(this->member_)
              , get_chunk_size_call_helper<T, wrapper_type>(this->member_)
              , mark_begin_execution_call_helper<T, wrapper_type>(this->member_)
              , mark_end_execution_call_helper<T, wrapper_type>(this->member_)
              , processing_units_count_call_helper<T, wrapper_type>(this->member_)
              , reset_thread_distribution_call_helper<T, wrapper_type>(this->member_)
            {}
        };

        ///////////////////////////////////////////////////////////////////////

#define HPX_STATIC_ASSERT_ON_PARAMETERS_AMBIGUITY(func)                       \
    static_assert(                                                            \
        parameters_type_counter<                                              \
            HPX_PP_CAT(hpx::parallel::execution::detail::has_, func)<         \
                typename hpx::util::decay_unwrap<Params>::type>::value...     \
        >::value <= 1,                                                        \
        "Passing more than one executor parameters type exposing "            \
            HPX_PP_STRINGIZE(func) " is not possible")                        \
    /**/

        template <typename ... Params>
        struct executor_parameters : public unwrapper<Params>...
        {
            static_assert(
                hpx::util::detail::all_of<
                    hpx::traits::is_executor_parameters<
                        typename std::decay<Params>::type
                    >...
                >::value,
                "All passed parameters must be a proper executor parameters "
                "objects");
            static_assert(sizeof...(Params) >= 2,
                "This type is meant to be used with at least 2 parameters "
                "objects");

            HPX_STATIC_ASSERT_ON_PARAMETERS_AMBIGUITY(get_chunk_size);
            HPX_STATIC_ASSERT_ON_PARAMETERS_AMBIGUITY(mark_begin_execution);
            HPX_STATIC_ASSERT_ON_PARAMETERS_AMBIGUITY(mark_end_execution);
            HPX_STATIC_ASSERT_ON_PARAMETERS_AMBIGUITY(count_processing_units);
            HPX_STATIC_ASSERT_ON_PARAMETERS_AMBIGUITY(maximal_number_of_chunks);
            HPX_STATIC_ASSERT_ON_PARAMETERS_AMBIGUITY(reset_thread_distribution);

            template <typename Dependent = void, typename Enable =
                typename std::enable_if<
                    hpx::util::detail::all_of<
                        std::is_constructible<Params>...
                    >::value, Dependent
                >::type>
            executor_parameters()
              : unwrapper<Params>()...
            {}

            template <typename ... Params_, typename Enable =
                typename std::enable_if<
                    hpx::util::detail::pack<Params...>::size ==
                        hpx::util::detail::pack<Params_...>::size
                >::type>
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
        static_assert(
            hpx::traits::is_executor_parameters<
                typename std::decay<Param>::type
            >::value,
            "The passed parameter must be a proper executor parameters object");

        return std::forward<Param>(param);
    }
}}}

#endif
