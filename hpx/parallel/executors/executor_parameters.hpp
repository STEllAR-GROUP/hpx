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
#include <functional>
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
            unwrapper() : T() {}

            // generic poor-mans forwarding constructor
            template <typename U>
            unwrapper(U && u) : T(std::forward<U>(u)) {}
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename T, typename Wrapper, typename Enable = void>
        struct variable_chunk_size_call_helper
        {
            variable_chunk_size_call_helper(Wrapper&) {}
        };

        template <typename T, typename Wrapper>
        struct variable_chunk_size_call_helper<T, Wrapper,
            typename std::enable_if<has_variable_chunk_size<T>::value>::type>
        {
            variable_chunk_size_call_helper(Wrapper& wrap)
              : wrap_(wrap)
            {}

            template <typename Executor>
            HPX_FORCEINLINE bool variable_chunk_size(Executor && exec)
            {
                return wrap_.get().variable_chunk_size(
                    std::forward<Executor>(exec));
            }

        private:
            Wrapper& wrap_;
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
                Executor && exec, std::size_t cores)
            {
                return wrap_.get().maximal_number_of_chunks(
                    std::forward<Executor>(exec), cores);
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
                std::size_t num_tasks)
            {
                return wrap_.get().get_chunk_size(std::forward<Executor>(exec),
                    std::forward<F>(f), num_tasks);
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

            HPX_FORCEINLINE void mark_begin_execution()
            {
                wrap_.get().mark_begin_execution();
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

            HPX_FORCEINLINE void mark_end_execution()
            {
                wrap_.get().mark_end_execution();
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
                has_processing_units_count<T>::value
            >::type>
        {
            processing_units_count_call_helper(Wrapper& wrap)
              : wrap_(wrap)
            {}

            HPX_FORCEINLINE std::size_t processing_units_count()
            {
                return wrap_.get().processing_units_count();
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
          , variable_chunk_size_call_helper<T, boost::reference_wrapper<T> >
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
              , variable_chunk_size_call_helper<T, wrapper_type>(this->member_)
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
          , variable_chunk_size_call_helper<T, std::reference_wrapper<T> >
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
              , variable_chunk_size_call_helper<T, wrapper_type>(this->member_)
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
            BOOST_PP_CAT(hpx::parallel::v3::detail::has_, func)<              \
                typename hpx::util::decay_unwrap<Params>::type>::value...     \
        >::value <= 1,                                                        \
        "Passing more than one executor parameters type exposing "            \
            BOOST_PP_STRINGIZE(func) " is not possible")                      \
    /**/

#if defined(HPX_MSVC) && HPX_MSVC < 1900
// for MSVC 12 disable: warning C4520: '...' : multiple default constructors specified
#pragma warning(push)
#pragma warning(disable: 4520)
#endif

        template <typename ... Params>
        struct executor_parameters : public unwrapper<Params>...
        {
            static_assert(
                hpx::util::detail::all_of<
                    hpx::traits::is_executor_parameters<Params>...
                >::value,
                "All passed parameters must be proper executor parameters "
                "objects"
            );
            static_assert(sizeof...(Params) >= 2,
                "This type is meant to be used with at least 2 parameters "
                "objects");

            HPX_STATIC_ASSERT_ON_PARAMETERS_AMBIGUITY(variable_chunk_size);
            HPX_STATIC_ASSERT_ON_PARAMETERS_AMBIGUITY(get_chunk_size);
            HPX_STATIC_ASSERT_ON_PARAMETERS_AMBIGUITY(mark_begin_execution);
            HPX_STATIC_ASSERT_ON_PARAMETERS_AMBIGUITY(mark_end_execution);
            HPX_STATIC_ASSERT_ON_PARAMETERS_AMBIGUITY(processing_units_count);
            HPX_STATIC_ASSERT_ON_PARAMETERS_AMBIGUITY(maximal_number_of_chunks);
            HPX_STATIC_ASSERT_ON_PARAMETERS_AMBIGUITY(reset_thread_distribution);

            executor_parameters()
              : unwrapper<Params>()...
            {}

            template <typename ... Params_, typename T =
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

#if defined(HPX_MSVC) && HPX_MSVC < 1900
#pragma warning(pop)
#endif

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
