//  Copyright (c) 2011-2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !BOOST_PP_IS_ITERATING

#ifndef HPX_UTIL_BIND_HPP
#define HPX_UTIL_BIND_HPP

#include <hpx/config.hpp>
#include <hpx/traits/is_action.hpp>
#include <hpx/util/tuple.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/detail/remove_reference.hpp>

#include <boost/get_pointer.hpp>

#include <boost/fusion/include/at_c.hpp>
#include <boost/fusion/include/vector.hpp>
#include <boost/fusion/functional/adapter/fused.hpp>

#include <boost/detail/workaround.hpp>
#include <boost/utility/enable_if.hpp>

#include <boost/preprocessor/iteration/iterate.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/repetition/enum.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>
#include <boost/preprocessor/repetition/repeat_from_to.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util {
    namespace detail
    {
        struct not_enough_parameters {};

        namespace result_of
        {
            template <typename Env, typename T>
            struct eval
            {
                typedef T & type;
            };

            template <typename Env, typename T>
            struct eval<Env, boost::reference_wrapper<T const> >
            {
                typedef T const & t;
            };

            template <typename Env, typename T>
            struct eval<Env, boost::reference_wrapper<T> >
            {
                typedef T & t;
            };
        }

        template <typename Env, typename T>
        T & eval(Env &, T & t)
        {
            return t;
        }

        template <typename Env, typename T>
        T const & eval(Env &, T const & t)
        {
            return t;
        }

        template <typename Env, typename T>
        T & eval(Env &, boost::reference_wrapper<T> const & r)
        {
            return r.get();
        }

        template <typename Env, typename T>
        T const & eval(Env &, boost::reference_wrapper<T const> const & r)
        {
            return r.get();
        }

        template <typename Env, typename T>
        T & eval(Env &, boost::reference_wrapper<T> & r)
        {
            return r.get();
        }

        template <typename Env, typename T>
        T const & eval(Env &, boost::reference_wrapper<T const> & r)
        {
            return r.get();
        }

        ///////////////////////////////////////////////////////////////////////
        using boost::get_pointer;

        template <typename T>
        T* get_pointer(boost::shared_ptr<T>& p)
        {
            return p.get();
        }

        template <typename T>
        T* get_pointer(boost::intrusive_ptr<T>& p)
        {
            return p.get();
        }

        template <typename T>
        T * get_pointer(T &t)
        {
            return &t;
        }

        template <typename T>
        T const * get_pointer(T const &t)
        {
            return &t;
        }
    }

    namespace placeholders
    {
        template <int N>
        struct arg {};

#define HPX_UTIL_BIND_PLACEHOLDERS(Z, N, D)                                     \
        typedef arg<N> BOOST_PP_CAT(BOOST_PP_CAT(arg_, BOOST_PP_INC(N)), _type);\
        BOOST_PP_CAT(BOOST_PP_CAT(arg_, BOOST_PP_INC(N)), _type) const          \
            BOOST_PP_CAT(arg_, BOOST_PP_INC(N)) = {};                           \
        BOOST_PP_CAT(BOOST_PP_CAT(arg_, BOOST_PP_INC(N)), _type) const          \
            BOOST_PP_CAT(_, BOOST_PP_INC(N)) = {};                              \
/**/
        BOOST_PP_REPEAT(HPX_FUNCTION_ARGUMENT_LIMIT, HPX_UTIL_BIND_PLACEHOLDERS, _)
#undef HPX_UTIL_BIND_PLACEHOLDERS

        template <typename>
        struct is_placeholder
          : boost::mpl::false_
        {};

        template <typename T>
        struct is_placeholder<T const>
          : is_placeholder<T>
        {};

        template <int N>
        struct is_placeholder<arg<N> >
          : boost::mpl::true_
        {};
    }

    namespace detail
    {
        namespace result_of
        {
            template <typename Env, int N>
            struct eval<Env, util::placeholders::arg<N> >
            {
                typedef typename boost::fusion::result_of::at_c<Env, N>::type type;
            };
            
            template <typename Env, int N>
            struct eval<Env, util::placeholders::arg<N> const>
            {
                typedef typename boost::fusion::result_of::at_c<Env, N>::type type;
            };
        }

        template <typename Env, int N>
        typename boost::fusion::result_of::at_c<Env, N>::type
        eval(Env & env, util::placeholders::arg<N>)
        {
            return boost::fusion::at_c<N>(env);
        }
    }

#define HPX_UTIL_BIND_EVAL(Z, N, D)                                             \
    ::hpx::util::detail::eval(env, BOOST_PP_CAT(arg, N))                        \
/**/

#define HPX_UTIL_BIND_REMOVE_REFERENCE(Z, N, D)                                 \
        typename detail::remove_reference<BOOST_PP_CAT(D, N)>::type             \
/**/

#define HPX_UTIL_BIND_REFERENCE(Z, N, D)                                        \
    typename detail::env_value_type<BOOST_PP_CAT(D, N)>::type                   \
/**/

#define HPX_UTIL_BIND_FUNCTION_OPERATOR(Z, N, D)                                \
    template <BOOST_PP_ENUM_PARAMS(N, typename A)>                              \
    BOOST_FORCEINLINE result_type operator()(HPX_ENUM_FWD_ARGS(N, A, a)) const  \
    {                                                                           \
        typedef                                                                 \
            BOOST_PP_CAT(hpx::util::tuple, N)<                                  \
                BOOST_PP_ENUM(N, HPX_UTIL_BIND_REFERENCE, A)                    \
            >                                                                   \
            env_type;                                                           \
        env_type env(HPX_ENUM_FORWARD_ARGS(N, A, a));                           \
        return eval(env, f) D;                                                  \
    }                                                                           \
                                                                                \
    template <BOOST_PP_ENUM_PARAMS(N, typename A)>                              \
    BOOST_FORCEINLINE result_type operator()(HPX_ENUM_FWD_ARGS(N, A, a))        \
    {                                                                           \
        typedef                                                                 \
            BOOST_PP_CAT(hpx::util::tuple, N)<                                  \
                BOOST_PP_ENUM(N, HPX_UTIL_BIND_REFERENCE, A)                    \
            >                                                                   \
            env_type;                                                           \
        env_type env(HPX_ENUM_FORWARD_ARGS(N, A, a));                           \
        return eval(env, f) D;                                                  \
    }                                                                           \
/**/

    ///////////////////////////////////////////////////////////////////////////
    // free functions
    namespace detail
    {
        template <typename R>
        struct bound_function0
        {
            typedef R result_type;
            R(*f)();

            bound_function0(R(*f_)())
                : f(f_)
            {}

            BOOST_FORCEINLINE R operator()() const
            {
                return f();
            }

            BOOST_PP_REPEAT_FROM_TO(1, HPX_FUNCTION_ARGUMENT_LIMIT,
                HPX_UTIL_BIND_FUNCTION_OPERATOR, ())
        };

        namespace result_of
        {
            template <typename Env, typename R>
            struct eval<Env, util::detail::bound_function0<R> >
            {
                typedef R type;
            };
        }

        template <typename Env, typename R>
        R
        eval(Env & env, util::detail::bound_function0<R> const & f)
        {
            return f();
        }

        template <typename Env, typename R>
        R
        eval(Env & env, util::detail::bound_function0<R> & f)
        {
            return f();
        }
    }

    template <typename R>
    detail::bound_function0<R>
    bind(R(*f)())
    {
        return detail::bound_function0<R>(f);
    }

    ///////////////////////////////////////////////////////////////////////////
    // functor
    namespace detail
    {
        template <typename F>
        struct bound_functor0
        {
            typedef typename boost::remove_const<F>::type functor_type;

            functor_type f;

            typedef
                typename boost::result_of<F()>::type
                result_type;

            // default constructor is needed for serialization
            bound_functor0()
            {}

            bound_functor0(bound_functor0 const & other)
                : f(other.f)
            {}

            bound_functor0(BOOST_RV_REF(bound_functor0) other)
                : f(boost::move(other.f))
            {}

            bound_functor0 & operator=(BOOST_COPY_ASSIGN_REF(bound_functor0) other)
            {
                f = other.f;
                return *this;
            }

            bound_functor0 & operator=(BOOST_RV_REF(bound_functor0) other)
            {
                f = boost::move(other.f);
                return *this;
            }

            bound_functor0(BOOST_RV_REF(F) f_)
                : f(boost::move(f_))
            {}

            bound_functor0(F const & f_)
                : f(f_)
            {}

            BOOST_FORCEINLINE result_type operator()() const
            {
                return f();
            }

            BOOST_FORCEINLINE result_type operator()()
            {
                return f();
            }

            BOOST_PP_REPEAT_FROM_TO(1, HPX_FUNCTION_ARGUMENT_LIMIT,
                HPX_UTIL_BIND_FUNCTION_OPERATOR, ())

        private:
            BOOST_COPYABLE_AND_MOVABLE(bound_functor0)
        };

        namespace result_of
        {
            template <typename Env, typename F>
            struct eval<Env, util::detail::bound_functor0<F> >
            {
                typedef typename boost::result_of<F()>::type type;
            };
        }

        template <typename Env, typename F>
        typename boost::result_of<F()>::type
        eval(Env & env, util::detail::bound_functor0<F> const & f)
        {
            return f();
        }
    }

    template <typename F>
    typename boost::disable_if<
        hpx::traits::is_action<typename detail::remove_reference<F>::type>,
        detail::bound_functor0<F>
    >::type
    bind(BOOST_FWD_REF(F) f)
    {
        return detail::bound_functor0<F>(boost::forward<F>(f));
    }
}}

///////////////////////////////////////////////////////////////////////////////
namespace boost { namespace serialization
{
    // serialization of the bound object, just serialize the function object
    template <typename F>
    void serialize(hpx::util::portable_binary_iarchive& ar
      , hpx::util::detail::bound_functor0<F>& bound
      , unsigned int const)
    {
        ar & bound.f;
    }

    template <typename F>
    void serialize(hpx::util::portable_binary_oarchive& ar
      , hpx::util::detail::bound_functor0<F>& bound
      , unsigned int const)
    {
        ar & bound.f;
    }
}}

#if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
#  include <hpx/util/preprocessed/bind.hpp>
#else

#if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(preserve: 1, line: 0, output: "preprocessed/bind_" HPX_LIMIT_STR ".hpp")
#endif

#define BOOST_PP_ITERATION_PARAMS_1                                             \
    (                                                                           \
        3                                                                       \
      , (                                                                       \
            1                                                                   \
          , HPX_FUNCTION_ARGUMENT_LIMIT                                         \
          , <hpx/util/bind.hpp>                                                 \
        )                                                                       \
    )                                                                           \
/**/
#include BOOST_PP_ITERATE()

#if defined(__WAVE__) && defined (HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(output: null)
#endif

#endif // !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)

#undef HPX_UTIL_BIND_MOVE_PARAMS
#undef HPX_UTIL_BIND_FWD_PARAMS
#undef HPX_UTIL_BIND_EVAL
#undef HPX_UTIL_BIND_REMOVE_REFERENCE
#undef HPX_UTIL_BIND_REFERENCE
#undef HPX_UTIL_BIND_FUNCTION_OPERATOR
#undef HPX_UTIL_BIND_FUNCTOR_OPERATOR

#endif

#else  // !BOOST_PP_IS_ITERATING

#define N BOOST_PP_FRAME_ITERATION(1)
#define NN BOOST_PP_FRAME_ITERATION(1)

#define HPX_UTIL_BIND_INIT_MEMBER(Z, N, D)                                      \
    BOOST_PP_CAT(arg, N)(boost::forward<BOOST_PP_CAT(A, N)>(BOOST_PP_CAT(a, N)))\
/**/
#define HPX_UTIL_BIND_MEMBER_TYPE(Z, NN, D)                                           \
    typename boost::remove_const<                                               \
        typename decay<BOOST_PP_CAT(Arg, NN)>::type                              \
    >::type                                               \
/**/
#define HPX_UTIL_BIND_MEMBER(Z, N, D)                                           \
    typename boost::remove_const<                                               \
        typename decay<BOOST_PP_CAT(Arg, N)>::type                              \
    >::type BOOST_PP_CAT(arg, N);                                               \
/**/

#define HPX_UTIL_BIND_INIT_COPY_MEMBER(Z, N, D)                                 \
    BOOST_PP_CAT(arg, N)(other.BOOST_PP_CAT(arg, N))                            \
/**/

#define HPX_UTIL_BIND_INIT_MOVE_MEMBER(Z, N, D)                                 \
    BOOST_PP_CAT(arg, N)(boost::move(other.BOOST_PP_CAT(arg, N)))               \
/**/

#define HPX_UTIL_BIND_ASSIGN_MEMBER(Z, N, D)                                    \
    BOOST_PP_CAT(arg, N) = other.BOOST_PP_CAT(arg, N);                          \
/**/

#define HPX_UTIL_BIND_MOVE_MEMBER(Z, N, D)                                      \
    BOOST_PP_CAT(arg, N) = boost::move(other.BOOST_PP_CAT(arg, N));             \
/**/

///////////////////////////////////////////////////////////////////////////////
// free functions
namespace hpx { namespace util
{
    namespace detail
    {
        template <
            typename R
          , BOOST_PP_ENUM_PARAMS(N, typename T)
          , BOOST_PP_ENUM_PARAMS(N, typename Arg)
        >
        struct BOOST_PP_CAT(bound_function, N)
        {
            typedef R result_type;
            typedef R(*function_pointer_type)(BOOST_PP_ENUM_PARAMS(N, T));
            function_pointer_type f;

            BOOST_PP_CAT(bound_function, N)(
                    BOOST_PP_CAT(bound_function, N) const & other)
                : f(other.f)
                , BOOST_PP_ENUM(N, HPX_UTIL_BIND_INIT_COPY_MEMBER, _)
            {}

            BOOST_PP_CAT(bound_function, N)(
                    BOOST_RV_REF(BOOST_PP_CAT(bound_function, N)) other)
                : f(boost::move(other.f))
                , BOOST_PP_ENUM(N, HPX_UTIL_BIND_INIT_MOVE_MEMBER, _)
            {}

            BOOST_PP_CAT(bound_function, N) & operator=(
                BOOST_COPY_ASSIGN_REF(BOOST_PP_CAT(bound_function, N)) other)
            {
                f = other.f;
                BOOST_PP_REPEAT(N, HPX_UTIL_BIND_ASSIGN_MEMBER, _)
                return *this;
            }

            BOOST_PP_CAT(bound_function, N) & operator=(
                BOOST_RV_REF(BOOST_PP_CAT(bound_function, N)) other)
            {
                f = boost::move(other.f);
                BOOST_PP_REPEAT(N, HPX_UTIL_BIND_MOVE_MEMBER, _)
                return *this;
            }

            template <BOOST_PP_ENUM_PARAMS(N, typename A)>
            BOOST_PP_CAT(bound_function, N)(
                function_pointer_type f
              , HPX_ENUM_FWD_ARGS(N, A, a)
            )
                : f(f)
                , BOOST_PP_ENUM(N, HPX_UTIL_BIND_INIT_MEMBER, _)
            {}

            BOOST_FORCEINLINE R operator()() const
            {
                typedef hpx::util::tuple0<> env_type;
                env_type env;
                return f(BOOST_PP_ENUM(N, HPX_UTIL_BIND_EVAL, _));
            }

            BOOST_FORCEINLINE R operator()()
            {
                typedef hpx::util::tuple0<> env_type;
                env_type env;
                return f(BOOST_PP_ENUM(N, HPX_UTIL_BIND_EVAL, _));
            }

            BOOST_PP_REPEAT_FROM_TO(1, HPX_FUNCTION_ARGUMENT_LIMIT,
                HPX_UTIL_BIND_FUNCTION_OPERATOR,
                (BOOST_PP_ENUM(N, HPX_UTIL_BIND_EVAL, _)))

            BOOST_PP_REPEAT(N, HPX_UTIL_BIND_MEMBER, _)

        private:
            BOOST_COPYABLE_AND_MOVABLE(BOOST_PP_CAT(bound_function, N));
        };

        namespace result_of
        {
            template <
                typename Env
              , typename R
              , BOOST_PP_ENUM_PARAMS(N, typename T)
              , BOOST_PP_ENUM_PARAMS(N, typename Arg)
            >
            struct eval<Env, 
                BOOST_PP_CAT(detail::bound_function, N)<
                    R
                  , BOOST_PP_ENUM_PARAMS(N, T)
                  , BOOST_PP_ENUM_PARAMS(N, Arg)
                >
            >
            {
                typedef R type;
            };
        }

        template <
            typename Env
          , typename R
          , BOOST_PP_ENUM_PARAMS(N, typename T)
          , BOOST_PP_ENUM_PARAMS(N, typename Arg)
        >
        R
        eval(
            Env & env
          , BOOST_PP_CAT(detail::bound_function, N)<
                R
              , BOOST_PP_ENUM_PARAMS(N, T)
              , BOOST_PP_ENUM_PARAMS(N, Arg)
            > const & f
        )
        {
            return
                boost::fusion::fused<
                    BOOST_PP_CAT(detail::bound_function, N)<
                        R
                      , BOOST_PP_ENUM_PARAMS(N, T)
                      , BOOST_PP_ENUM_PARAMS(N, Arg)
                    >
                >(f)(
                    env
                 );
        }
    }

    template <
        typename R
      , BOOST_PP_ENUM_PARAMS(N, typename T)
      , BOOST_PP_ENUM_PARAMS(N, typename A)
    >
    BOOST_PP_CAT(detail::bound_function, N)<
        R
      , BOOST_PP_ENUM_PARAMS(N, T)
      , BOOST_PP_ENUM(N, HPX_UTIL_BIND_REMOVE_REFERENCE, A)
    >
    bind(
        R(*f)(BOOST_PP_ENUM_PARAMS(N, T))
      , HPX_ENUM_FWD_ARGS(N, A, a)
    )
    {
        return
            BOOST_PP_CAT(detail::bound_function, N)<
                R
              , BOOST_PP_ENUM_PARAMS(N, T)
              , BOOST_PP_ENUM(N, HPX_UTIL_BIND_REMOVE_REFERENCE, A)
            >
            (f, HPX_ENUM_FORWARD_ARGS(N, A, a));
    }
}}

///////////////////////////////////////////////////////////////////////////////
// member function pointers
namespace hpx { namespace util
{
    namespace detail
    {
        template <
            typename R
          , typename C
          , BOOST_PP_ENUM_PARAMS(BOOST_PP_DEC(N), typename T)
          BOOST_PP_COMMA_IF(BOOST_PP_DEC(N)) BOOST_PP_ENUM_PARAMS(N, typename Arg)
        >
        struct BOOST_PP_CAT(bound_member_function, N)
        {
            typedef R result_type;
            typedef R(C::*function_pointer_type)(
                BOOST_PP_ENUM_PARAMS(BOOST_PP_DEC(N), T));
            function_pointer_type f;

            template <BOOST_PP_ENUM_PARAMS(N, typename A)>
            BOOST_PP_CAT(bound_member_function, N)(
                function_pointer_type f
              , HPX_ENUM_FWD_ARGS(N, A, a)
            )
                : f(f)
                , BOOST_PP_ENUM(N, HPX_UTIL_BIND_INIT_MEMBER, _)
            {}

            BOOST_PP_CAT(bound_member_function, N)(
                    BOOST_PP_CAT(bound_member_function, N) const & other)
                : f(other.f)
                , BOOST_PP_ENUM(N, HPX_UTIL_BIND_INIT_COPY_MEMBER, _)
            {}

            BOOST_PP_CAT(bound_member_function, N)(BOOST_RV_REF(
                    BOOST_PP_CAT(bound_member_function, N)) other)
                : f(boost::move(other.f))
                , BOOST_PP_ENUM(N, HPX_UTIL_BIND_INIT_MOVE_MEMBER, _)
            {}

            BOOST_PP_CAT(bound_member_function, N) & operator=(
                BOOST_COPY_ASSIGN_REF(BOOST_PP_CAT(bound_member_function, N)) other)
            {
                f = other.f;
                BOOST_PP_REPEAT(N, HPX_UTIL_BIND_ASSIGN_MEMBER, _)
                return *this;
            }

            BOOST_PP_CAT(bound_member_function, N) & operator=(
                BOOST_RV_REF(BOOST_PP_CAT(bound_member_function, N)) other)
            {
                f = boost::move(other.f);
                BOOST_PP_REPEAT(N, HPX_UTIL_BIND_MOVE_MEMBER, _)
                return *this;
            }

            BOOST_FORCEINLINE R operator()() const
            {
                using detail::get_pointer;
                typedef hpx::util::tuple0<> env_type;
                env_type env;
                return
                    (get_pointer(detail::eval(env, arg0))->*f)
                        (BOOST_PP_ENUM_SHIFTED(N, HPX_UTIL_BIND_EVAL, _));
            }

            BOOST_FORCEINLINE R operator()()
            {
                using detail::get_pointer;
                typedef hpx::util::tuple0<> env_type;
                env_type env;
                return
                    (get_pointer(detail::eval(env, arg0))->*f)
                        (BOOST_PP_ENUM_SHIFTED(N, HPX_UTIL_BIND_EVAL, _));
            }
#define BOOST_PP_ITERATION_PARAMS_2                                             \
    (                                                                           \
        3                                                                       \
      , (                                                                       \
            1                                                                   \
          , HPX_FUNCTION_ARGUMENT_LIMIT                                         \
          , <hpx/util/detail/bind_functor_operator.hpp>                         \
        )                                                                       \
    )                                                                           \
 /**/
#include BOOST_PP_ITERATE()

            BOOST_PP_REPEAT(N, HPX_UTIL_BIND_MEMBER, _)
        };

        template <
            typename R
          , typename C
          , BOOST_PP_ENUM_PARAMS(BOOST_PP_DEC(N), typename T)
          BOOST_PP_COMMA_IF(BOOST_PP_DEC(N)) BOOST_PP_ENUM_PARAMS(N, typename Arg)
        >
        struct BOOST_PP_CAT(bound_member_function, N)<
            R
          , C const
          , BOOST_PP_ENUM_PARAMS(BOOST_PP_DEC(N), T)
          BOOST_PP_COMMA_IF(BOOST_PP_DEC(N)) BOOST_PP_ENUM_PARAMS(N, Arg)
        >
        {
            typedef R result_type;
            typedef R(C::*function_pointer_type)(
                BOOST_PP_ENUM_PARAMS(BOOST_PP_DEC(N), T)) const;
            function_pointer_type f;

            template <BOOST_PP_ENUM_PARAMS(N, typename A)>
            BOOST_PP_CAT(bound_member_function, N)(
                function_pointer_type f
              , HPX_ENUM_FWD_ARGS(N, A, a)
            )
                : f(f)
                , BOOST_PP_ENUM(N, HPX_UTIL_BIND_INIT_MEMBER, _)
            {}

            BOOST_PP_CAT(bound_member_function, N)(
                    BOOST_PP_CAT(bound_member_function, N) const & other)
                : f(other.f)
                , BOOST_PP_ENUM(N, HPX_UTIL_BIND_INIT_COPY_MEMBER, _)
            {}

            BOOST_PP_CAT(bound_member_function, N)(
                    BOOST_RV_REF(BOOST_PP_CAT(bound_member_function, N)) other)
                : f(boost::move(other.f))
                , BOOST_PP_ENUM(N, HPX_UTIL_BIND_INIT_MOVE_MEMBER, _)
            {}

            BOOST_PP_CAT(bound_member_function, N) & operator=(
                BOOST_COPY_ASSIGN_REF(BOOST_PP_CAT(bound_member_function, N)) other)
            {
                f = other.f;
                BOOST_PP_REPEAT(N, HPX_UTIL_BIND_ASSIGN_MEMBER, _)
                return *this;
            }

            BOOST_PP_CAT(bound_member_function, N) & operator=(
                BOOST_RV_REF(BOOST_PP_CAT(bound_member_function, N)) other)
            {
                f = boost::move(other.f);
                BOOST_PP_REPEAT(N, HPX_UTIL_BIND_MOVE_MEMBER, _)
                return *this;
            }

            BOOST_FORCEINLINE R operator()() const
            {
                using detail::get_pointer;
                typedef hpx::util::tuple0<> env_type;
                env_type env;
                return
                    (get_pointer(detail::eval(env, arg0))->*f)
                        (BOOST_PP_ENUM_SHIFTED(N, HPX_UTIL_BIND_EVAL, _));
            }

            BOOST_FORCEINLINE R operator()()
            {
                using detail::get_pointer;
                typedef hpx::util::tuple0<> env_type;
                env_type env;
                return
                    (get_pointer(detail::eval(env, arg0))->*f)
                        (BOOST_PP_ENUM_SHIFTED(N, HPX_UTIL_BIND_EVAL, _));
            }

#define HPX_UTIL_BIND_MEMBER_FUNCTION_OPERATOR(Z, N, D)                         \
    template <BOOST_PP_ENUM_PARAMS(N, typename A)>                              \
    BOOST_FORCEINLINE result_type operator()(HPX_ENUM_FWD_ARGS(N, A, a))        \
    {                                                                           \
        using detail::get_pointer;                                              \
        typedef                                                                 \
            BOOST_PP_CAT(hpx::util::tuple, N)<                                  \
                BOOST_PP_ENUM(N, HPX_UTIL_BIND_REFERENCE, A)                    \
            >                                                                   \
            env_type;                                                           \
        env_type env(HPX_ENUM_FORWARD_ARGS(N, A, a));                           \
        return                                                                  \
            (get_pointer(detail::eval(env, a0))->*f)                            \
                (BOOST_PP_ENUM_SHIFTED(NN, HPX_UTIL_BIND_EVAL, _));             \
    }                                                                           \
    template <BOOST_PP_ENUM_PARAMS(N, typename A)>                              \
    BOOST_FORCEINLINE result_type operator()(HPX_ENUM_FWD_ARGS(N, A, a)) const  \
    {                                                                           \
        using detail::get_pointer;                                              \
        typedef                                                                 \
            BOOST_PP_CAT(hpx::util::tuple, N)<                                  \
                BOOST_PP_ENUM(N, HPX_UTIL_BIND_REFERENCE, A)                    \
            >                                                                   \
            env_type;                                                           \
        env_type env(HPX_ENUM_FORWARD_ARGS(N, A, a));                           \
        return                                                                  \
            (get_pointer(detail::eval(env, a0))->*f)                            \
                (BOOST_PP_ENUM_SHIFTED(NN, HPX_UTIL_BIND_EVAL, _));             \
    }                                                                           \
/**/

            BOOST_PP_REPEAT_FROM_TO(
                1
              , HPX_FUNCTION_ARGUMENT_LIMIT
              , HPX_UTIL_BIND_MEMBER_FUNCTION_OPERATOR, _
            )
#undef HPX_UTIL_BIND_MEMBER_FUNCTION_OPERATOR
            BOOST_PP_REPEAT(N, HPX_UTIL_BIND_MEMBER, _)
        };

        namespace result_of
        {
            template <
                typename Env
              , typename R
              , typename C
              , BOOST_PP_ENUM_PARAMS(BOOST_PP_DEC(N), typename T)
              BOOST_PP_COMMA_IF(BOOST_PP_DEC(N))
                  BOOST_PP_ENUM_PARAMS(N, typename Arg)
            >
            struct eval<Env, 
                BOOST_PP_CAT(detail::bound_member_function, N)<
                    R
                  , C
                  , BOOST_PP_ENUM_PARAMS(BOOST_PP_DEC(N), T)
                  BOOST_PP_COMMA_IF(BOOST_PP_DEC(N)) BOOST_PP_ENUM_PARAMS(N, Arg)
                >
            >
            {
                typedef R type;
            };
        }

        template <
            typename Env
          , typename R
          , typename C
          , BOOST_PP_ENUM_PARAMS(BOOST_PP_DEC(N), typename T)
          BOOST_PP_COMMA_IF(BOOST_PP_DEC(N))
              BOOST_PP_ENUM_PARAMS(N, typename Arg)
        >
        R
        eval(
            Env & env
          , BOOST_PP_CAT(detail::bound_member_function, N)<
                R
              , C
              , BOOST_PP_ENUM_PARAMS(BOOST_PP_DEC(N), T)
              BOOST_PP_COMMA_IF(BOOST_PP_DEC(N)) BOOST_PP_ENUM_PARAMS(N, Arg)
            > const & f
        )
        {
            return
                boost::fusion::fused<
                    BOOST_PP_CAT(detail::bound_member_function, N)<
                        R
                      , C
                      , BOOST_PP_ENUM_PARAMS(BOOST_PP_DEC(N), T)
                        BOOST_PP_COMMA_IF(BOOST_PP_DEC(N))
                        BOOST_PP_ENUM_PARAMS(N, Arg)
                    >
                >(f)(
                    env
                 );
        }
    }

    template <
        typename R
      , typename C
      , BOOST_PP_ENUM_PARAMS(BOOST_PP_DEC(N), typename T)
      BOOST_PP_COMMA_IF(BOOST_PP_DEC(N)) BOOST_PP_ENUM_PARAMS(N, typename A)
    >
    BOOST_PP_CAT(detail::bound_member_function, N)<
        R
      , C
      , BOOST_PP_ENUM_PARAMS(BOOST_PP_DEC(N), T)
      BOOST_PP_COMMA_IF(BOOST_PP_DEC(N))
        BOOST_PP_ENUM(N, HPX_UTIL_BIND_REMOVE_REFERENCE, A)
    >
    bind(
        R(C::*f)(BOOST_PP_ENUM_PARAMS(BOOST_PP_DEC(N), T))
      , HPX_ENUM_FWD_ARGS(N, A, a)
    )
    {
        return
            BOOST_PP_CAT(detail::bound_member_function, N)<
                R
              , C
              , BOOST_PP_ENUM_PARAMS(BOOST_PP_DEC(N), T)
                BOOST_PP_COMMA_IF(BOOST_PP_DEC(N))
                BOOST_PP_ENUM(N, HPX_UTIL_BIND_REMOVE_REFERENCE, A)
            >
            (f, HPX_ENUM_FORWARD_ARGS(N, A, a));
    }

    template <
        typename R
      , typename C
      , BOOST_PP_ENUM_PARAMS(BOOST_PP_DEC(N), typename T)
      BOOST_PP_COMMA_IF(BOOST_PP_DEC(N)) BOOST_PP_ENUM_PARAMS(N, typename A)
    >
    BOOST_PP_CAT(detail::bound_member_function, N)<
        R
      , C const
      , BOOST_PP_ENUM_PARAMS(BOOST_PP_DEC(N), T)
      BOOST_PP_COMMA_IF(BOOST_PP_DEC(N))
        BOOST_PP_ENUM(N, HPX_UTIL_BIND_REMOVE_REFERENCE, A)
    >
    bind(
        R(C::*f)(BOOST_PP_ENUM_PARAMS(BOOST_PP_DEC(N), T)) const
      , HPX_ENUM_FWD_ARGS(N, A, a)
    )
    {
        return
            BOOST_PP_CAT(detail::bound_member_function, N)<
                R
              , C const
              , BOOST_PP_ENUM_PARAMS(BOOST_PP_DEC(N), T)
              BOOST_PP_COMMA_IF(BOOST_PP_DEC(N))
                  BOOST_PP_ENUM(N, HPX_UTIL_BIND_REMOVE_REFERENCE, A)
            >
            (f, HPX_ENUM_FORWARD_ARGS(N, A, a));
    }
}}

///////////////////////////////////////////////////////////////////////////////
// functor
namespace hpx { namespace util
{
    namespace detail
    {
#define HPX_UTIL_BIND_RESULT_OF_BOUND_FUNCTOR_ARGS(Z, NN, D)                    \
    typename detail::result_of::eval<                                           \
        HPX_UTIL_STRIP(D)                                                       \
      , HPX_UTIL_BIND_MEMBER_TYPE(Z, NN, D)                                     \
    >::type                                                                     \

        namespace result_of
        {
            template <
                typename F
              , BOOST_PP_ENUM_PARAMS(N, typename Arg)
              , typename Enable = typename boost::mpl::fold<
                    boost::mpl::vector<BOOST_PP_ENUM_PARAMS(N, Arg)>
                  , boost::mpl::false_
                  , boost::mpl::or_<
                        placeholders::is_placeholder<boost::mpl::_2>
                      , boost::mpl::_1
                    >
                >::type
            >
            struct BOOST_PP_CAT(bound_functor, N);

#define HPX_UTIL_BIND_RESULT_OF_BOUND_FUNCTOR(Z, NN, D)                         \
            template <                                                          \
                typename F                                                      \
              , BOOST_PP_ENUM_PARAMS(NN, typename A)                            \
              , BOOST_PP_ENUM_PARAMS(N, typename Arg)                           \
            >                                                                   \
            struct BOOST_PP_CAT(bound_functor, N)<                              \
                F(BOOST_PP_ENUM_PARAMS(NN, A))                                  \
              , BOOST_PP_ENUM_PARAMS(N, Arg)                                    \
              , boost::mpl::false_                                              \
            >                                                                   \
            {                                                                   \
                typedef                                                         \
                    typename boost::result_of<                                  \
                        typename boost::remove_reference<                       \
                            typename detail::result_of::eval<                   \
                                BOOST_PP_CAT(hpx::util::tuple, NN)<             \
                                    BOOST_PP_ENUM(NN, HPX_UTIL_BIND_REFERENCE, A)\
                                >                                               \
                              , F                                               \
                            >::type                                             \
                        >::type(                                                \
                            BOOST_PP_ENUM(                                      \
                                N                                               \
                              , HPX_UTIL_BIND_RESULT_OF_BOUND_FUNCTOR_ARGS      \
                              , (BOOST_PP_CAT(hpx::util::tuple, NN)<            \
                                    BOOST_PP_ENUM(NN, HPX_UTIL_BIND_REFERENCE, A)\
                                >)                                              \
                            )                                                   \
                        )                                                       \
                    >::type                                                     \
                    type;                                                       \
            };                                                                  \
/**/

            BOOST_PP_REPEAT_FROM_TO(
                1
              , HPX_FUNCTION_ARGUMENT_LIMIT
              , HPX_UTIL_BIND_RESULT_OF_BOUND_FUNCTOR
              , _
            )

            template <
                typename F
              , BOOST_PP_ENUM_PARAMS(N, typename Arg)
            >
            struct BOOST_PP_CAT(bound_functor, N)<
                F()
              , BOOST_PP_ENUM_PARAMS(N, Arg), boost::mpl::false_>
            {
                typedef
                    typename boost::result_of<
                        typename boost::remove_reference<
                            typename detail::result_of::eval<
                                hpx::util::tuple0<>
                              , F
                            >::type
                        >::type(
                            BOOST_PP_ENUM(
                                N
                              , HPX_UTIL_BIND_RESULT_OF_BOUND_FUNCTOR_ARGS
                              , hpx::util::tuple0<>
                            )
                        )
                    >::type
                    type;
            };
            
            template <
                typename F
              , BOOST_PP_ENUM_PARAMS(N, typename Arg)
            >
            struct BOOST_PP_CAT(bound_functor, N)<
                F()
              , BOOST_PP_ENUM_PARAMS(N, Arg)
              , boost::mpl::true_
            >
            {
                typedef
                    detail::not_enough_parameters
                    type;
            };
        }

        template <
            typename F
          , BOOST_PP_ENUM_PARAMS(N, typename Arg)
        >
        struct BOOST_PP_CAT(bound_functor, N)
        {
            typedef typename boost::remove_const<F>::type functor_type;

            functor_type f;

            template <typename Sig>
            struct result;

            template <typename This>
            struct result<This()>
            {
                typedef
                    typename result_of::BOOST_PP_CAT(bound_functor, N)<
                        F()
                      , BOOST_PP_ENUM_PARAMS(N, Arg)
                    >::type
                    type;
            };


            // default constructor is needed for serialization
            BOOST_PP_CAT(bound_functor, N)()
            {}

            template <BOOST_PP_ENUM_PARAMS(N, typename A)>
            BOOST_PP_CAT(bound_functor, N)(
                BOOST_RV_REF(F) f_
              , HPX_ENUM_FWD_ARGS(N, A, a)
            )
                : f(boost::move(f_))
                , BOOST_PP_ENUM(N, HPX_UTIL_BIND_INIT_MEMBER, _)
            {}

            template <BOOST_PP_ENUM_PARAMS(N, typename A)>
            BOOST_PP_CAT(bound_functor, N)(
                F const & f_
              , HPX_ENUM_FWD_ARGS(N, A, a)
            )
                : f(f_)
                , BOOST_PP_ENUM(N, HPX_UTIL_BIND_INIT_MEMBER, _)
            {}

            BOOST_PP_CAT(bound_functor, N)(
                    BOOST_PP_CAT(bound_functor, N) const & other)
                : f(other.f)
                , BOOST_PP_ENUM(N, HPX_UTIL_BIND_INIT_COPY_MEMBER, _)
            {}

            BOOST_PP_CAT(bound_functor, N)(
                    BOOST_RV_REF(BOOST_PP_CAT(bound_functor, N)) other)
                : f(boost::move(other.f))
                , BOOST_PP_ENUM(N, HPX_UTIL_BIND_INIT_MOVE_MEMBER, _)
            {}

            BOOST_PP_CAT(bound_functor, N) & operator=(
                BOOST_COPY_ASSIGN_REF(BOOST_PP_CAT(bound_functor, N)) other)
            {
                f = other.f;
                BOOST_PP_REPEAT(N, HPX_UTIL_BIND_ASSIGN_MEMBER, _)
                return *this;
            }

            BOOST_PP_CAT(bound_functor, N) & operator=(
                BOOST_RV_REF(BOOST_PP_CAT(bound_functor, N)) other)
            {
                f = boost::move(other.f);
                BOOST_PP_REPEAT(N, HPX_UTIL_BIND_MOVE_MEMBER, _)
                return *this;
            }

            BOOST_FORCEINLINE
            typename result_of::BOOST_PP_CAT(bound_functor, N)<
                F()
              , BOOST_PP_ENUM_PARAMS(N, Arg)
            >::type
            operator()() const
            {
                typedef hpx::util::tuple0<> env_type;
                env_type env;
                return eval(env, f)(BOOST_PP_ENUM(N, HPX_UTIL_BIND_EVAL, _));
            }

#define HPX_UTIL_BIND_FUNCTOR_OPERATOR(Z, NN, D)                                 \
    template <typename This, BOOST_PP_ENUM_PARAMS(NN, typename A)>                                                    \
    struct result<This(BOOST_PP_ENUM_PARAMS(NN, A))>                                                       \
    {                                                                           \
        typedef                                                                 \
        typename result_of::BOOST_PP_CAT(bound_functor, N)<F(BOOST_PP_ENUM_PARAMS(NN, A)), BOOST_PP_ENUM_PARAMS(N, Arg), boost::mpl::false_>::type \
            type;                                                               \
    };                                                                          \
    template <BOOST_PP_ENUM_PARAMS(NN, typename A)>                              \
    BOOST_FORCEINLINE                                                           \
    typename result_of::BOOST_PP_CAT(bound_functor, N)<F(BOOST_PP_ENUM_PARAMS(NN, A)), BOOST_PP_ENUM_PARAMS(N, Arg), boost::mpl::false_>::type \
    operator()(HPX_ENUM_FWD_ARGS(NN, A, a)) const  \
    {                                                                           \
        typedef                                                                 \
            BOOST_PP_CAT(hpx::util::tuple, NN)<                                  \
                BOOST_PP_ENUM(NN, HPX_UTIL_BIND_REFERENCE, A)                    \
            >                                                                   \
            env_type;                                                           \
        env_type env(HPX_ENUM_FORWARD_ARGS(NN, A, a));                           \
        return eval(env, f) D;                                                  \
    }                                                                           \
                                                                                \
    template <BOOST_PP_ENUM_PARAMS(NN, typename A)>                              \
    BOOST_FORCEINLINE                                                           \
    typename result_of::BOOST_PP_CAT(bound_functor, N)<F(BOOST_PP_ENUM_PARAMS(NN, A)), BOOST_PP_ENUM_PARAMS(N, Arg), boost::mpl::false_>::type \
    operator()(HPX_ENUM_FWD_ARGS(NN, A, a))                                      \
    {                                                                           \
        typedef                                                                 \
            BOOST_PP_CAT(hpx::util::tuple, NN)<                                  \
                BOOST_PP_ENUM(NN, HPX_UTIL_BIND_REFERENCE, A)                    \
            >                                                                   \
            env_type;                                                           \
        env_type env(HPX_ENUM_FORWARD_ARGS(NN, A, a));                            \
        return eval(env, f) D;                                                  \
    }                                                                           \
/**/

            BOOST_PP_REPEAT_FROM_TO(1, HPX_FUNCTION_ARGUMENT_LIMIT,
                HPX_UTIL_BIND_FUNCTOR_OPERATOR,
                (BOOST_PP_ENUM(N, HPX_UTIL_BIND_EVAL, _)))

            BOOST_PP_REPEAT(N, HPX_UTIL_BIND_MEMBER, _)
        };

        namespace result_of
        {
            template <typename Env
              , typename F
              , BOOST_PP_ENUM_PARAMS(N, typename Arg)
            >
            struct eval<
                Env
              , BOOST_PP_CAT(detail::bound_functor, N)<
                    F
                  , BOOST_PP_ENUM_PARAMS(N, Arg)
                >
            >
            {
                typedef
                    typename boost::result_of<
                        boost::fusion::fused<
                            BOOST_PP_CAT(detail::bound_functor, N)<
                                F
                              , BOOST_PP_ENUM_PARAMS(N, Arg)
                            >
                        >(Env &)
                    >::type
                    type;
            };
        }


        template <
            typename Env
          , typename F
          , BOOST_PP_ENUM_PARAMS(N, typename Arg)
        >
        typename boost::result_of<
            boost::fusion::fused<
                BOOST_PP_CAT(detail::bound_functor, N)<
                    F
                  , BOOST_PP_ENUM_PARAMS(N, Arg)
                >
            >(Env &)
        >::type
        eval(
            Env & env
          , BOOST_PP_CAT(detail::bound_functor, N)<
                F
              , BOOST_PP_ENUM_PARAMS(N, Arg)
            > const & f
        )
        {
            return
                boost::fusion::fused<
                    BOOST_PP_CAT(detail::bound_functor, N)<
                        F
                      , BOOST_PP_ENUM_PARAMS(N, Arg)
                    >
                >(f)(
                    env
                 );
        }
    }

    template <typename F
      , BOOST_PP_ENUM_PARAMS(N, typename A)
    >
    typename boost::disable_if<
        hpx::traits::is_action<typename detail::remove_reference<F>::type>,
        BOOST_PP_CAT(detail::bound_functor, N)<
            typename detail::remove_reference<F>::type
          , BOOST_PP_ENUM(N, HPX_UTIL_BIND_REMOVE_REFERENCE, A)>
    >::type
    bind(
        BOOST_FWD_REF(F) f
      , HPX_ENUM_FWD_ARGS(N, A, a)
    )
    {
        return
            BOOST_PP_CAT(detail::bound_functor, N)<
                typename detail::remove_reference<F>::type
              , BOOST_PP_ENUM(N, HPX_UTIL_BIND_REMOVE_REFERENCE, A)
            >(
                boost::forward<F>(f)
              , HPX_ENUM_FORWARD_ARGS(N, A, a)
            );
    }
}}

namespace boost {
    template <
        typename F
      , BOOST_PP_ENUM_PARAMS(N, typename Arg)
    >
    struct result_of<hpx::util::detail::BOOST_PP_CAT(bound_functor, N)<F, BOOST_PP_ENUM_PARAMS(N, Arg)>()>
    {
        typedef
            typename hpx::util::detail::result_of::BOOST_PP_CAT(bound_functor, N)<F(), BOOST_PP_ENUM_PARAMS(N, Arg)>::type
            type;
    };
}

///////////////////////////////////////////////////////////////////////////////
namespace boost { namespace serialization
{
#define HPX_UTIL_BIND_SERIALIZE_MEMBER(Z, NNN, _) ar & BOOST_PP_CAT(bound.arg, NNN);

    // serialization of the bound object, just serialize function object and
    // members
    template <typename F, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    void serialize(hpx::util::portable_binary_iarchive& ar
      , BOOST_PP_CAT(hpx::util::detail::bound_functor, N)<
            F, BOOST_PP_ENUM_PARAMS(N, Arg)
        >& bound
      , unsigned int const)
    {
        ar & bound.f;
        BOOST_PP_REPEAT(N, HPX_UTIL_BIND_SERIALIZE_MEMBER, _)
    }

    template <typename F, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    void serialize(hpx::util::portable_binary_oarchive& ar
      , BOOST_PP_CAT(hpx::util::detail::bound_functor, N)<
            F, BOOST_PP_ENUM_PARAMS(N, Arg)
        >& bound
      , unsigned int const)
    {
        ar & bound.f;
        BOOST_PP_REPEAT(N, HPX_UTIL_BIND_SERIALIZE_MEMBER, _)
    }

#undef HPX_UTIL_BIND_SERIALIZE_MEMBER
}}

#undef HPX_UTIL_BIND_ASSIGN_MEMBER
#undef HPX_UTIL_BIND_INIT_MOVE_MEMBER
#undef HPX_UTIL_BIND_INIT_COPY_MEMBER
#undef HPX_UTIL_BIND_MEMBER
#undef HPX_UTIL_BIND_INIT_MEMBER

#undef NN
#undef N

#endif
